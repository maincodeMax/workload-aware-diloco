from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path
from typing import Any, Iterable

from wa_diloco.config import ExperimentConfig, ensure_workspace, load_config
from wa_diloco.state import compute_delta, load_state, save_state
from wa_diloco.telemetry import append_jsonl, read_json, write_json


def _assignment_path(config: ExperimentConfig, control_cycle: int, learner_id: int) -> Path:
    return (
        config.runtime.workspace
        / "assignments"
        / f"cycle_{control_cycle:05d}"
        / f"learner_{learner_id:02d}.json"
    )


def _report_path(config: ExperimentConfig, control_cycle: int, learner_id: int) -> Path:
    return (
        config.runtime.workspace
        / "reports"
        / f"cycle_{control_cycle:05d}"
        / f"learner_{learner_id:02d}.json"
    )


def _delta_path(config: ExperimentConfig, control_cycle: int, learner_id: int) -> Path:
    return (
        config.runtime.workspace
        / "updates"
        / f"cycle_{control_cycle:05d}"
        / f"learner_{learner_id:02d}"
        / "delta.pt"
    )


def _local_state_path(config: ExperimentConfig, learner_id: int) -> Path:
    return config.runtime.workspace / "learners" / f"learner_{learner_id:02d}" / "local_state.pt"


def _wait_assignment(config: ExperimentConfig, control_cycle: int, learner_id: int) -> dict[str, Any]:
    path = _assignment_path(config, control_cycle, learner_id)
    while not path.exists():
        time.sleep(config.runtime.poll_interval_sec)
    return read_json(path)


def _message_to_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return str(messages)
    lines = []
    for item in messages:
        if isinstance(item, dict):
            role = item.get("role", "unknown")
            content = item.get("content", "")
            lines.append(f"{role}: {content}")
        else:
            lines.append(str(item))
    return "\n".join(lines)


def _example_text(example: dict[str, Any], text_column: str) -> str:
    if text_column in example:
        return str(example[text_column])
    if "messages" in example:
        return _message_to_text(example["messages"])
    raise KeyError(f"dataset example has no {text_column!r} or 'messages' column")


def _dataset_iter(config: ExperimentConfig, learner_id: int) -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(
        config.dataset.name,
        split=config.dataset.split,
        streaming=config.dataset.streaming,
    )
    if config.dataset.streaming:
        ds = ds.shuffle(seed=config.dataset.shuffle_seed + learner_id, buffer_size=10_000)
    return iter(ds)


def _batch_iter(config: ExperimentConfig, tokenizer, learner_id: int, device: str):
    import torch

    source = _dataset_iter(config, learner_id)
    while True:
        texts = []
        for example in itertools.islice(source, config.learners.per_device_batch_size):
            texts.append(_example_text(example, config.dataset.text_column))
        if not texts:
            source = _dataset_iter(config, learner_id)
            continue
        encoded = tokenizer(
            texts,
            max_length=config.model.seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        labels = input_ids.clone()
        if tokenizer.pad_token_id is not None:
            labels[input_ids == tokenizer.pad_token_id] = -100
        yield {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "tokens": int(torch.sum(attention_mask).item()),
        }


def _load_model(config: ExperimentConfig, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if config.model.dtype == "bf16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.train()
    return model, tokenizer


def _build_optimizer(config: ExperimentConfig, model):
    import torch

    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learners.learning_rate,
        weight_decay=config.learners.weight_decay,
    )


def _train_chunk(config: ExperimentConfig, model, optimizer, batches, local_steps: int, device: str):
    import torch

    loss_start = None
    loss_end = None
    tokens = 0
    start = time.time()
    for _ in range(local_steps):
        batch = next(batches)
        optimizer.zero_grad(set_to_none=True)
        autocast_device = "cuda" if device.startswith("cuda") else "cpu"
        with torch.autocast(
            device_type=autocast_device,
            dtype=torch.bfloat16,
            enabled=device.startswith("cuda"),
        ):
            loss = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            ).loss
        if loss_start is None:
            loss_start = float(loss.detach().to("cpu"))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.learners.max_grad_norm)
        optimizer.step()
        loss_end = float(loss.detach().to("cpu"))
        tokens += int(batch["tokens"])
    return {
        "loss_start": loss_start,
        "loss_end": loss_end,
        "tokens": tokens,
        "wall_time_sec": time.time() - start,
    }


def run(config_path: str | Path, learner_id: int) -> None:
    import torch

    config = load_config(config_path)
    ensure_workspace(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = _load_model(config, device)
    optimizer = _build_optimizer(config, model)
    batches = _batch_iter(config, tokenizer, learner_id, device)
    events = config.runtime.workspace / "events" / f"learner_{learner_id:02d}.jsonl"

    sync_base_state: dict[str, Any] | None = None
    local_steps_since_sync = 0
    tokens_since_sync = 0
    last_global_round = -1
    sync_started_at = time.time()
    local_state_path = _local_state_path(config, learner_id)

    append_jsonl(events, {"event": "learner_start", "learner_id": learner_id, "device": device})

    for control_cycle in range(config.runtime.max_control_cycles + 1):
        assignment = _wait_assignment(config, control_cycle, learner_id)
        if assignment["type"] == "stop":
            break

        global_round = int(assignment["global_round"])
        continue_local = bool(assignment["continue_local"])
        base_state_path = Path(assignment["base_state_path"])

        if not continue_local or sync_base_state is None or global_round != last_global_round:
            sync_base_state = load_state(base_state_path, map_location="cpu")
            model.load_state_dict(sync_base_state, strict=True)
            model.to(device)
            optimizer = _build_optimizer(config, model)
            local_steps_since_sync = 0
            tokens_since_sync = 0
            sync_started_at = time.time()
            last_global_round = global_round
        elif local_state_path.exists():
            model.load_state_dict(load_state(local_state_path, map_location="cpu"), strict=True)
            model.to(device)

        local_steps = int(assignment["local_steps"])
        metrics = _train_chunk(config, model, optimizer, batches, local_steps, device)
        local_steps_since_sync += local_steps
        tokens_since_sync += int(metrics["tokens"])

        current_state = {key: value.detach().to("cpu") for key, value in model.state_dict().items()}
        assert sync_base_state is not None
        delta, update_norm = compute_delta(sync_base_state, current_state)

        delta_path = _delta_path(config, control_cycle, learner_id)
        save_state(delta_path, delta)
        save_state(local_state_path, current_state)

        report = {
            "learner_id": learner_id,
            "control_cycle": control_cycle,
            "global_round": global_round,
            "local_steps": local_steps,
            "local_steps_since_sync": local_steps_since_sync,
            "tokens_since_sync": tokens_since_sync,
            "loss_start": metrics["loss_start"],
            "loss_end": metrics["loss_end"],
            "update_norm": update_norm,
            "wall_time_sec": metrics["wall_time_sec"],
            "staleness_sec": time.time() - sync_started_at,
            "delta_path": str(delta_path),
            "local_state_path": str(local_state_path),
            "domain": config.dataset.domain_name,
        }
        write_json(_report_path(config, control_cycle, learner_id), report)
        append_jsonl(events, {"event": "report", **report})

    append_jsonl(events, {"event": "learner_stop", "learner_id": learner_id})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to experiment YAML")
    parser.add_argument("--learner-id", type=int, required=True)
    args = parser.parse_args(argv)
    run(args.config, args.learner_id)
    # Avoid CUDA/extension teardown aborts after the learner has written its
    # final report and stop event. Exceptions before this point still propagate.
    import os

    os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

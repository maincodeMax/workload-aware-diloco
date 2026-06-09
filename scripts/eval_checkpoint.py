#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

from wa_diloco.config import load_config
from wa_diloco.state import load_state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Experiment YAML used for the checkpoint")
    parser.add_argument("--workspace", help="Override workspace path")
    parser.add_argument("--checkpoint", help="Override checkpoint path")
    parser.add_argument("--output", required=True, help="JSON output path")
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset name")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=64)
    args = parser.parse_args()

    config = load_config(args.config)
    workspace = Path(args.workspace).expanduser().resolve() if args.workspace else config.runtime.workspace
    checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else _latest_checkpoint(workspace)
    payload = evaluate(
        config_path=args.config,
        model_name=config.model.name_or_path,
        dtype_name=config.model.dtype,
        seq_len=config.model.seq_len,
        workspace=workspace,
        checkpoint=checkpoint,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        text_column=args.text_column,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def evaluate(
    *,
    config_path: str,
    model_name: str,
    dtype_name: str,
    seq_len: int,
    workspace: Path,
    checkpoint: Path,
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    text_column: str,
    batch_size: int,
    max_batches: int,
) -> dict[str, Any]:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dtype_name == "bf16" and device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    model.load_state_dict(load_state(checkpoint, map_location="cpu"), strict=True)
    model.to(device)
    model.eval()

    dataset_kwargs = {"split": split}
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, **dataset_kwargs)
    else:
        dataset = load_dataset(dataset_name, **dataset_kwargs)

    total_loss = 0.0
    total_tokens = 0
    batches = 0
    examples = 0
    with torch.no_grad():
        for texts in _text_batches(dataset, text_column=text_column, batch_size=batch_size):
            encoded = tokenizer(
                texts,
                max_length=seq_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            labels = input_ids.clone()
            if tokenizer.pad_token_id is not None:
                labels[input_ids == tokenizer.pad_token_id] = -100
            valid_tokens = int((labels != -100).sum().detach().to("cpu"))
            if valid_tokens <= 0:
                continue
            loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            ).loss
            total_loss += float(loss.detach().to("cpu")) * valid_tokens
            total_tokens += valid_tokens
            examples += len(texts)
            batches += 1
            if batches >= max_batches:
                break

    eval_loss = total_loss / total_tokens if total_tokens else None
    return {
        "checkpoint": str(checkpoint),
        "config": str(config_path),
        "dataset": dataset_name,
        "dataset_config": dataset_config,
        "device": device,
        "eval_batches": batches,
        "eval_examples": examples,
        "eval_loss": eval_loss,
        "eval_perplexity": math.exp(eval_loss) if eval_loss is not None and eval_loss < 50 else None,
        "eval_tokens": total_tokens,
        "split": split,
        "text_column": text_column,
        "workspace": str(workspace),
    }


def _latest_checkpoint(workspace: Path) -> Path:
    checkpoints = sorted((workspace / "checkpoints").glob("global_round_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no global_round_*.pt checkpoints under {workspace / 'checkpoints'}")
    return checkpoints[-1]


def _text_batches(dataset: Iterable[dict[str, Any]], *, text_column: str, batch_size: int):
    batch: list[str] = []
    for example in dataset:
        text = _example_text(example, text_column).strip()
        if not text:
            continue
        batch.append(text)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _example_text(example: dict[str, Any], text_column: str) -> str:
    if text_column in example:
        return str(example[text_column])
    if "messages" in example and isinstance(example["messages"], list):
        return "\n".join(
            f"{item.get('role', 'unknown')}: {item.get('content', '')}"
            if isinstance(item, dict)
            else str(item)
            for item in example["messages"]
        )
    raise KeyError(f"dataset example has no {text_column!r} or 'messages' column")


if __name__ == "__main__":
    raise SystemExit(main())

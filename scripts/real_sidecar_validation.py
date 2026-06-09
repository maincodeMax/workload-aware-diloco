#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import random
import statistics
import time
from pathlib import Path


PROMPTS = [
    "Summarize how a distributed training scheduler should react to inference load.",
    "Explain why fixed synchronization intervals can be inefficient in production.",
    "Write a short incident note about serving latency during model training.",
    "List the signals a workload-aware training controller should observe.",
]

LONG_CONTEXT = (
    "You are analyzing telemetry from a shared accelerator fleet running "
    "language-model training and inference. The training job uses low-"
    "communication outer synchronization. The serving workload has bursty "
    "arrivals, mixed prompt lengths, and latency SLOs for interactive users. "
    "Given the following synthetic incident notes, identify likely contention "
    "points, explain how synchronization timing could affect queueing, and "
    "recommend what metrics should be inspected before claiming a production "
    "SLO improvement. Incident notes: "
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/stage2/sidecar-validation")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--duration-sec", type=float, default=240.0)
    parser.add_argument("--period-sec", type=float, default=90.0)
    parser.add_argument("--sync-period-sec", type=float, default=37.0)
    parser.add_argument("--sync-window-sec", type=float, default=8.0)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--slo-ms", type=float, default=0.0)
    parser.add_argument("--backend", choices=["auto", "vllm", "sglang", "transformers"], default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--stress-mode", choices=["pressure", "sync", "both", "off"], default="both")
    parser.add_argument("--stress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sync-events-jsonl", default="")
    parser.add_argument("--sync-events-repeat", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sync-windows", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--load-profile", choices=["steady", "bursty", "trace"], default="steady")
    parser.add_argument("--quiet-concurrency", type=int, default=2)
    parser.add_argument("--busy-concurrency", type=int, default=12)
    parser.add_argument("--burst-period-sec", type=float, default=120.0)
    parser.add_argument("--burst-window-sec", type=float, default=30.0)
    parser.add_argument("--burst-phase-sec", type=float, default=0.0)
    parser.add_argument("--prompt-style", choices=["short", "mixed", "long"], default="short")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--stress-matmul-size", type=int, default=4096)
    parser.add_argument("--stress-steps", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.sync_schedule = _load_sync_schedule(
        args.sync_events_jsonl,
        default_window_sec=args.sync_window_sec,
    )

    start = time.time()
    stop_at = start + args.duration_sec
    stress_proc = None
    if args.stress:
        stress_proc = mp.Process(
            target=_stress_loop,
            args=(
                start,
                stop_at,
                args.period_sec,
                args.sync_period_sec,
                args.sync_window_sec,
                args.sync_schedule,
                args.sync_events_repeat,
                args.stress_mode,
                args.stress_matmul_size,
                args.stress_steps,
            ),
            daemon=True,
        )
        stress_proc.start()

    try:
        real_events = _run_real_sidecar(args, start=start, stop_at=stop_at)
    finally:
        if stress_proc is not None:
            stress_proc.join(timeout=2)
            if stress_proc.is_alive():
                stress_proc.terminate()

    sim_events = _simulate_events(
        count=len(real_events),
        duration_sec=args.duration_sec,
        period_sec=args.period_sec,
        seed=args.seed,
    )
    if args.slo_ms > 0:
        for event in real_events:
            event["slo_ms"] = args.slo_ms
        for event in sim_events:
            event["slo_ms"] = args.slo_ms

    _write_jsonl(output_dir / "real_events.jsonl", real_events)
    _write_jsonl(output_dir / "sim_events.jsonl", sim_events)
    summary = {
        "backend": real_events[0]["backend"] if real_events else args.backend,
        "duration_sec": args.duration_sec,
        "model": args.model,
        "period_sec": args.period_sec,
        "concurrency": args.concurrency,
        "load_profile": args.load_profile,
        "quiet_concurrency": args.quiet_concurrency,
        "busy_concurrency": args.busy_concurrency,
        "burst_period_sec": args.burst_period_sec,
        "burst_window_sec": args.burst_window_sec,
        "burst_phase_sec": args.burst_phase_sec,
        "prompt_style": args.prompt_style,
        "max_new_tokens": args.max_new_tokens,
        "slo_ms": args.slo_ms if args.slo_ms > 0 else None,
        "real": _summary(real_events),
        "simulator": _summary(sim_events),
        "sync_period_sec": args.sync_period_sec,
        "sync_window_sec": args.sync_window_sec,
        "sync_events_jsonl": args.sync_events_jsonl or None,
        "sync_event_count": len(args.sync_schedule["starts"]) if args.sync_schedule else 0,
        "sync_events_repeat": args.sync_events_repeat,
        "sync_windows_enabled": args.sync_windows,
        "sync_trace_duration_sec": args.sync_schedule["duration_sec"] if args.sync_schedule else None,
        "stress_mode": args.stress_mode,
        "stress_enabled": args.stress,
        "stress_matmul_size": args.stress_matmul_size,
        "stress_steps": args.stress_steps,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _run_real_sidecar(args, *, start: float, stop_at: float) -> list[dict]:
    if args.backend == "sglang":
        return _run_sglang(args, start=start, stop_at=stop_at)
    if args.backend in {"auto", "vllm"}:
        try:
            return _run_vllm(args, start=start, stop_at=stop_at)
        except Exception:
            if args.backend == "vllm":
                raise
    return _run_transformers(args, start=start, stop_at=stop_at)


def _run_vllm(args, *, start: float, stop_at: float) -> list[dict]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    sampling = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0)
    events = []
    index = 0
    while time.time() < stop_at:
        elapsed = time.time() - start
        concurrency = _concurrency_for_args(elapsed, args)
        prompts = _prompt_batch(index, concurrency, args.prompt_style)
        pressure = _pressure_for_args(elapsed, args)
        sync_active = _sync_active_for_args(elapsed, args)
        begin = time.perf_counter()
        outputs = llm.generate(prompts, sampling)
        batch_latency_ms = (time.perf_counter() - begin) * 1000.0
        output_tokens = [
            len(item.outputs[0].token_ids) if item.outputs else 0
            for item in outputs
        ]
        for offset, prompt in enumerate(prompts):
            events.append(
                _event(
                    "vllm",
                    elapsed,
                    pressure,
                    batch_latency_ms,
                    sync_active=sync_active,
                    batch_size=len(prompts),
                    load_concurrency=concurrency,
                    prompt_chars=len(prompt),
                    output_tokens=output_tokens[offset] if offset < len(output_tokens) else None,
                )
            )
        index += len(prompts)
    return events


def _run_sglang(args, *, start: float, stop_at: float) -> list[dict]:
    import sglang as sgl

    engine = _build_sglang_engine(args, sgl)
    sampling = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0.0,
    }
    events = []
    index = 0
    try:
        while time.time() < stop_at:
            elapsed = time.time() - start
            concurrency = _concurrency_for_args(elapsed, args)
            prompts = _prompt_batch(index, concurrency, args.prompt_style)
            pressure = _pressure_for_args(elapsed, args)
            sync_active = _sync_active_for_args(elapsed, args)
            begin = time.perf_counter()
            outputs = engine.generate(prompts, sampling)
            batch_latency_ms = (time.perf_counter() - begin) * 1000.0
            output_tokens = [_sglang_output_tokens(item) for item in outputs]
            for offset, prompt in enumerate(prompts):
                events.append(
                    _event(
                        "sglang",
                        elapsed,
                        pressure,
                        batch_latency_ms,
                        sync_active=sync_active,
                        batch_size=len(prompts),
                        load_concurrency=concurrency,
                        prompt_chars=len(prompt),
                        output_tokens=output_tokens[offset] if offset < len(output_tokens) else None,
                    )
                )
            index += len(prompts)
    finally:
        shutdown = getattr(engine, "shutdown", None)
        if callable(shutdown):
            shutdown()
    return events


def _build_sglang_engine(args, sgl):
    kwargs = {
        "model_path": args.model,
        "dtype": "bfloat16",
        "mem_fraction_static": args.gpu_memory_utilization,
        "context_length": args.max_model_len,
    }
    try:
        return sgl.Engine(**kwargs)
    except TypeError:
        kwargs.pop("context_length", None)
        try:
            return sgl.Engine(**kwargs)
        except TypeError:
            return sgl.Engine(model_path=args.model)


def _sglang_output_tokens(item) -> int | None:
    if isinstance(item, dict):
        meta = item.get("meta_info") or item.get("metadata") or {}
        for key in ("completion_tokens", "output_tokens", "num_output_tokens"):
            value = meta.get(key) if isinstance(meta, dict) else None
            if value is not None:
                return int(value)
        text = item.get("text") or item.get("output") or item.get("generated_text")
        return len(str(text).split()) if text is not None else None
    text = getattr(item, "text", None) or getattr(item, "output", None)
    return len(str(text).split()) if text is not None else None


def _run_transformers(args, *, start: float, stop_at: float) -> list[dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()

    events = []
    index = 0
    with torch.no_grad():
        while time.time() < stop_at:
            elapsed = time.time() - start
            concurrency = _concurrency_for_args(elapsed, args)
            prompts = _prompt_batch(index, concurrency, args.prompt_style)
            pressure = _pressure_for_args(elapsed, args)
            sync_active = _sync_active_for_args(elapsed, args)
            encoded = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
            if device == "cuda":
                torch.cuda.synchronize()
            begin = time.perf_counter()
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
            if device == "cuda":
                torch.cuda.synchronize()
            batch_latency_ms = (time.perf_counter() - begin) * 1000.0
            prompt_len = int(encoded["input_ids"].shape[1])
            for offset, prompt in enumerate(prompts):
                output_tokens = int(generated[offset].shape[0]) - prompt_len
                events.append(
                    _event(
                        "transformers",
                        elapsed,
                        pressure,
                        batch_latency_ms,
                        sync_active=sync_active,
                        batch_size=len(prompts),
                        load_concurrency=concurrency,
                        prompt_chars=len(prompt),
                        output_tokens=output_tokens,
                    )
                )
            index += len(prompts)
    return events


def _stress_loop(
    start: float,
    stop_at: float,
    period_sec: float,
    sync_period_sec: float,
    sync_window_sec: float,
    sync_schedule: dict | None,
    sync_events_repeat: bool,
    stress_mode: str,
    matmul_size: int,
    stress_steps: int,
) -> None:
    import torch

    if not torch.cuda.is_available():
        return
    device = "cuda"
    matmul_size = max(1024, int(matmul_size))
    stress_steps = max(1, int(stress_steps))
    a = torch.randn((matmul_size, matmul_size), device=device, dtype=torch.bfloat16)
    b = torch.randn((matmul_size, matmul_size), device=device, dtype=torch.bfloat16)
    while time.time() < stop_at:
        elapsed = time.time() - start
        pressure = _pressure(elapsed, period_sec)
        pressure_active = max(pressure.values()) >= 0.6
        sync_active = _sync_active_for_schedule(
            elapsed,
            sync_period_sec=sync_period_sec,
            sync_window_sec=sync_window_sec,
            sync_schedule=sync_schedule,
            repeat=sync_events_repeat,
        )
        should_stress = (
            stress_mode == "both" and (pressure_active or sync_active)
        ) or (
            stress_mode == "pressure" and pressure_active
        ) or (
            stress_mode == "sync" and sync_active
        )
        if should_stress:
            for _ in range(stress_steps):
                _ = a @ b
            torch.cuda.synchronize()
        else:
            time.sleep(0.05)


def _prompt_batch(index: int, concurrency: int, style: str) -> list[str]:
    concurrency = max(1, int(concurrency))
    prompts = []
    for offset in range(concurrency):
        base = PROMPTS[(index + offset) % len(PROMPTS)]
        if style == "long":
            prompt = LONG_CONTEXT + " ".join([base] * 24)
        elif style == "mixed":
            repeat = [1, 4, 12, 24][(index + offset) % 4]
            prompt = (LONG_CONTEXT if repeat >= 12 else "") + " ".join([base] * repeat)
        else:
            prompt = base
        prompts.append(prompt)
    return prompts


def _simulate_events(*, count: int, duration_sec: float, period_sec: float, seed: int) -> list[dict]:
    rng = random.Random(seed)
    events = []
    count = max(1, count)
    for index in range(count):
        elapsed = duration_sec * index / count
        pressure = _pressure(elapsed, period_sec)
        high_pressure = max(pressure.values()) >= 0.6
        sync_active = _sync_active(elapsed, period_sec / 2.5, 8.0)
        latency_ms = _sim_latency_ms(rng, pressure, sync_active=high_pressure)
        events.append(_event("simulator", elapsed, pressure, latency_ms, sync_active=sync_active))
    return events


def _pressure(elapsed: float, period_sec: float) -> dict[str, float]:
    phase = (math.sin(2 * math.pi * elapsed / period_sec) + 1.0) / 2.0
    return {
        "checkpoint_pressure": 0.8 if int(elapsed // period_sec) % 4 == 3 else 0.0,
        "inference_pressure": 0.1 + 0.8 * max(0.0, phase - 0.35) / 0.65,
        "network_pressure": 0.2 + 0.6 * phase,
    }


def _pressure_for_args(elapsed: float, args) -> dict[str, float]:
    if args.load_profile == "steady":
        return _pressure(elapsed, args.period_sec)

    busy = _serving_busy(elapsed, args)
    if args.load_profile == "bursty":
        inference = 0.95 if busy else 0.08
        network = 0.70 if busy else 0.20
        checkpoint = 0.0
    else:
        phase = ((elapsed + args.burst_phase_sec) % args.burst_period_sec) / args.burst_period_sec
        spike = busy or 0.48 <= phase <= 0.58
        inference = 0.90 if spike else 0.15
        network = 0.75 if spike else 0.25
        checkpoint = 0.75 if 0.78 <= phase <= 0.90 else 0.0
    return {
        "checkpoint_pressure": checkpoint,
        "inference_pressure": inference,
        "network_pressure": network,
    }


def _concurrency_for_args(elapsed: float, args) -> int:
    if args.load_profile == "steady":
        return max(1, int(args.concurrency))
    return max(
        1,
        int(args.busy_concurrency if _serving_busy(elapsed, args) else args.quiet_concurrency),
    )


def _serving_busy(elapsed: float, args) -> bool:
    period = max(1.0, float(args.burst_period_sec))
    window = max(0.0, min(float(args.burst_window_sec), period))
    phase = (elapsed + float(args.burst_phase_sec)) % period
    return phase < window


def _sync_active(elapsed: float, period_sec: float, window_sec: float) -> bool:
    if period_sec <= 0 or window_sec <= 0:
        return False
    return (elapsed % period_sec) <= window_sec


def _sync_active_for_args(elapsed: float, args) -> bool:
    if not args.sync_windows:
        return False
    return _sync_active_for_schedule(
        elapsed,
        sync_period_sec=args.sync_period_sec,
        sync_window_sec=args.sync_window_sec,
        sync_schedule=args.sync_schedule,
        repeat=args.sync_events_repeat,
    )


def _sync_active_for_schedule(
    elapsed: float,
    *,
    sync_period_sec: float,
    sync_window_sec: float,
    sync_schedule: dict | None,
    repeat: bool,
) -> bool:
    if not sync_schedule:
        return _sync_active(elapsed, sync_period_sec, sync_window_sec)

    starts = sync_schedule["starts"]
    window_sec = sync_schedule["window_sec"]
    if repeat:
        duration_sec = sync_schedule["duration_sec"]
        if duration_sec and duration_sec > window_sec:
            elapsed = elapsed % duration_sec

    return any(start <= elapsed <= start + window_sec for start in starts)


def _load_sync_schedule(path: str, *, default_window_sec: float) -> dict | None:
    if not path:
        return None
    records = _read_jsonl(Path(path))
    if not records:
        raise ValueError(f"sync event file is empty or missing: {path}")

    elapsed_starts = [
        float(record["elapsed_sec"])
        for record in records
        if record.get("sync_active") or record.get("event") in {"sync", "merged"}
        if record.get("elapsed_sec") is not None
    ]
    if elapsed_starts:
        duration = max(float(record.get("elapsed_sec", 0.0)) for record in records)
        return {
            "starts": sorted(elapsed_starts),
            "window_sec": float(default_window_sec),
            "duration_sec": max(duration, max(elapsed_starts) + default_window_sec),
        }

    start_ts = next(
        (int(record["ts_ms"]) for record in records if record.get("event") == "coordinator_start" and record.get("ts_ms")),
        None,
    )
    ts_values = [int(record["ts_ms"]) for record in records if record.get("ts_ms")]
    if start_ts is None and ts_values:
        start_ts = min(ts_values)
    if start_ts is None:
        raise ValueError(f"sync event file has no ts_ms or elapsed_sec values: {path}")

    starts = [
        (int(record["ts_ms"]) - start_ts) / 1000.0
        for record in records
        if record.get("event") == "merged" and record.get("ts_ms")
    ]
    if not starts:
        raise ValueError(f"sync event file has no merged events: {path}")
    duration = (max(ts_values) - start_ts) / 1000.0 if ts_values else max(starts) + default_window_sec
    return {
        "starts": sorted(starts),
        "window_sec": float(default_window_sec),
        "duration_sec": max(duration, max(starts) + default_window_sec),
    }


def _sim_latency_ms(rng: random.Random, pressure: dict[str, float], *, sync_active: bool) -> float:
    network = pressure["network_pressure"]
    inference = pressure["inference_pressure"]
    checkpoint = pressure["checkpoint_pressure"]
    sync_penalty = 35.0 * (0.35 + 0.65 * max(network, inference)) if sync_active else 0.0
    jitter = rng.lognormvariate(0.0, 0.18) * 6.0
    return 55.0 + 24.0 * network + 34.0 * inference + 16.0 * checkpoint + sync_penalty + jitter


def _event(
    backend: str,
    elapsed: float,
    pressure: dict[str, float],
    latency_ms: float,
    *,
    sync_active: bool,
    batch_size: int = 1,
    load_concurrency: int | None = None,
    prompt_chars: int | None = None,
    output_tokens: int | None = None,
) -> dict:
    high_pressure = max(pressure.values()) >= 0.6
    return {
        "backend": backend,
        "elapsed_sec": elapsed,
        "high_pressure": high_pressure,
        "latency_ms": latency_ms,
        "batch_size": batch_size,
        "load_concurrency": load_concurrency if load_concurrency is not None else batch_size,
        "max_pressure": max(pressure.values()),
        "output_tokens": output_tokens,
        "prompt_chars": prompt_chars,
        "sync_active": sync_active,
        **pressure,
    }


def _summary(events: list[dict]) -> dict:
    latencies = [float(event["latency_ms"]) for event in events]
    high = [float(event["latency_ms"]) for event in events if event["high_pressure"]]
    low = [float(event["latency_ms"]) for event in events if not event["high_pressure"]]
    sync_active = [float(event["latency_ms"]) for event in events if event.get("sync_active")]
    sync_inactive = [float(event["latency_ms"]) for event in events if not event.get("sync_active")]
    high_p95 = _percentile(high, 95)
    low_p95 = _percentile(low, 95)
    sync_active_p95 = _percentile(sync_active, 95)
    sync_inactive_p95 = _percentile(sync_inactive, 95)
    elapsed = [float(event["elapsed_sec"]) for event in events]
    output_tokens = [
        int(event["output_tokens"])
        for event in events
        if event.get("output_tokens") is not None
    ]
    slo_events = [
        event
        for event in events
        if event.get("slo_ms") is not None
    ]
    slo_ms = float(slo_events[0]["slo_ms"]) if slo_events else None
    slo_violation_count = (
        sum(1 for event in slo_events if float(event["latency_ms"]) > float(event["slo_ms"]))
        if slo_events
        else None
    )
    duration = max(elapsed) - min(elapsed) if len(elapsed) >= 2 else None
    return {
        "correlation_pressure_latency": _correlation(
            [float(event["max_pressure"]) for event in events],
            latencies,
        ),
        "high_pressure_count": len(high),
        "high_pressure_p95_ms": high_p95,
        "high_low_p95_ratio": high_p95 / low_p95 if low_p95 else None,
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p95_ms": _percentile(latencies, 95),
        "latency_p99_ms": _percentile(latencies, 99),
        "low_pressure_count": len(low),
        "low_pressure_p95_ms": low_p95,
        "output_tokens": sum(output_tokens) if output_tokens else None,
        "requests": len(events),
        "slo_ms": slo_ms,
        "slo_violation_count": slo_violation_count,
        "slo_violation_rate": (
            slo_violation_count / len(slo_events) if slo_events and slo_violation_count is not None else None
        ),
        "sync_active_count": len(sync_active),
        "sync_active_p95_ms": sync_active_p95,
        "sync_inactive_count": len(sync_inactive),
        "sync_inactive_p95_ms": sync_inactive_p95,
        "sync_inactive_p95_ratio": (
            sync_active_p95 / sync_inactive_p95 if sync_active_p95 and sync_inactive_p95 else None
        ),
        "tokens_per_sec": (
            sum(output_tokens) / duration if output_tokens and duration and duration > 0 else None
        ),
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def _write_jsonl(path: Path, events: list[dict]) -> None:
    with path.open("w") as f:
        for event in events:
            f.write(json.dumps(event, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with path.open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


if __name__ == "__main__":
    raise SystemExit(main())

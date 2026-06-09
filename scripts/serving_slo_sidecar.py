#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", help="Experiment workspace directory")
    parser.add_argument("--period-sec", type=float, default=300.0)
    parser.add_argument("--interval-sec", type=float, default=1.0)
    parser.add_argument("--mode", choices=["clean", "mixed", "bursty"], default="mixed")
    parser.add_argument("--requests-per-interval", type=int, default=32)
    parser.add_argument("--slo-ms", type=float, default=120.0)
    parser.add_argument("--base-latency-ms", type=float, default=55.0)
    parser.add_argument("--sync-penalty-ms", type=float, default=35.0)
    parser.add_argument("--sync-window-sec", type=float, default=8.0)
    parser.add_argument("--burst-window-sec", type=float, default=30.0)
    parser.add_argument("--burst-phase-sec", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    workspace = Path(args.workspace)
    pressure_target = workspace / "pressure" / "latest.json"
    serving_dir = workspace / "serving"
    events_path = serving_dir / "events.jsonl"
    pressure_target.parent.mkdir(parents=True, exist_ok=True)
    serving_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    start = time.time()
    seen_merges = 0
    last_merge_time = 0.0

    with events_path.open("a", buffering=1) as events:
        while True:
            now = time.time()
            elapsed = now - start
            pressure = _pressure(
                elapsed,
                args.period_sec,
                args.mode,
                sync_window_sec=args.sync_window_sec,
                burst_window_sec=args.burst_window_sec,
                burst_phase_sec=args.burst_phase_sec,
            )
            _write_json_atomic(pressure_target, {"ts": now, **pressure})

            seen_merges, last_merge_time = _observe_latest_merge(
                workspace / "events" / "coordinator.jsonl",
                seen_merges,
                last_merge_time,
                now,
            )
            sync_active = (now - last_merge_time) <= args.sync_window_sec

            for _ in range(args.requests_per_interval):
                latency_ms = _latency_ms(
                    rng,
                    args.base_latency_ms,
                    pressure,
                    sync_active=sync_active,
                    sync_penalty_ms=args.sync_penalty_ms,
                )
                events.write(
                    json.dumps(
                        {
                            "ts": now,
                            "elapsed_sec": elapsed,
                            "latency_ms": latency_ms,
                            "slo_ms": args.slo_ms,
                            "slo_violation": latency_ms > args.slo_ms,
                            "sync_active": sync_active,
                            **pressure,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            time.sleep(args.interval_sec)


def _pressure(
    elapsed: float,
    period_sec: float,
    mode: str,
    *,
    sync_window_sec: float,
    burst_window_sec: float,
    burst_phase_sec: float,
) -> dict[str, float]:
    if mode == "clean":
        return {
            "network_pressure": 0.0,
            "inference_pressure": 0.0,
            "checkpoint_pressure": 0.0,
            "serving_sync_cost": 0.0,
        }
    if mode == "bursty":
        busy_now = _burst_fraction(
            elapsed,
            window_sec=0.5,
            period_sec=period_sec,
            burst_window_sec=burst_window_sec,
            burst_phase_sec=burst_phase_sec,
        )
        sync_cost = _burst_fraction(
            elapsed,
            window_sec=sync_window_sec,
            period_sec=period_sec,
            burst_window_sec=burst_window_sec,
            burst_phase_sec=burst_phase_sec,
        )
        return {
            "network_pressure": 0.15 + 0.20 * busy_now,
            "inference_pressure": 0.15 + 0.75 * busy_now,
            "checkpoint_pressure": 0.0,
            "serving_sync_cost": sync_cost,
        }
    phase = (math.sin(2 * math.pi * elapsed / period_sec) + 1.0) / 2.0
    return {
        "network_pressure": 0.2 + 0.6 * phase,
        "inference_pressure": 0.1 + 0.8 * max(0.0, phase - 0.35) / 0.65,
        "checkpoint_pressure": 0.8 if int(elapsed // period_sec) % 4 == 3 else 0.0,
        "serving_sync_cost": 0.0,
    }


def _burst_fraction(
    start_sec: float,
    *,
    window_sec: float,
    period_sec: float,
    burst_window_sec: float,
    burst_phase_sec: float,
) -> float:
    if period_sec <= 0 or burst_window_sec <= 0 or window_sec <= 0:
        return 0.0
    samples = 32
    busy = 0
    for index in range(samples):
        t = start_sec + window_sec * (index + 0.5) / samples
        phase = (t + burst_phase_sec) % period_sec
        if phase < burst_window_sec:
            busy += 1
    return busy / samples


def _observe_latest_merge(
    coordinator_path: Path,
    seen_merges: int,
    last_merge_time: float,
    now: float,
) -> tuple[int, float]:
    if not coordinator_path.exists():
        return seen_merges, last_merge_time

    merges = 0
    with coordinator_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "merged":
                merges += 1
    if merges > seen_merges:
        return merges, now
    return seen_merges, last_merge_time


def _latency_ms(
    rng: random.Random,
    base_latency_ms: float,
    pressure: dict[str, float],
    *,
    sync_active: bool,
    sync_penalty_ms: float,
) -> float:
    network = pressure["network_pressure"]
    inference = pressure["inference_pressure"]
    checkpoint = pressure["checkpoint_pressure"]
    penalty = sync_penalty_ms * (0.35 + 0.65 * max(network, inference)) if sync_active else 0.0
    jitter = rng.lognormvariate(0.0, 0.18) * 6.0
    return (
        base_latency_ms
        + 24.0 * network
        + 34.0 * inference
        + 16.0 * checkpoint
        + penalty
        + jitter
    )


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())

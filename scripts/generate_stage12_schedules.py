#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


SEEDS = [23, 29, 31, 37, 41]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/stage12/schedules")
    parser.add_argument("--duration-sec", type=float, default=600.0)
    parser.add_argument("--sync-window-sec", type=float, default=8.0)
    parser.add_argument("--burst-period-sec", type=float, default=120.0)
    parser.add_argument("--burst-window-sec", type=float, default=30.0)
    parser.add_argument("--burst-phase-sec", type=float, default=0.0)
    parser.add_argument("--random-draws", type=int, default=5)
    parser.add_argument("--min-gap-sec", type=float, default=18.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for seed in SEEDS:
        wa_path = _policy_workspace("wa_diloco", seed)
        wa_starts, _ = _read_sync_starts(wa_path)
        count = len(wa_starts)
        if count <= 0:
            raise ValueError(f"no WA sync starts for seed {seed}: {wa_path}")

        calibrated = _calibrated_starts(
            count=count,
            duration=args.duration_sec,
            window=args.sync_window_sec,
            min_gap=args.min_gap_sec,
            burst_period=args.burst_period_sec,
            burst_window=args.burst_window_sec,
            burst_phase=args.burst_phase_sec,
        )
        calibrated_path = output_dir / f"calibrated_wa-s{seed}.jsonl"
        _write_schedule(calibrated_path, calibrated, duration=args.duration_sec)
        manifest.append(
            {
                "policy": "calibrated_wa",
                "seed": seed,
                "draw": None,
                "count": count,
                "path": str(calibrated_path),
            }
        )

        for draw in range(args.random_draws):
            rng = random.Random(seed * 1009 + draw)
            starts = _random_starts(
                rng,
                count=count,
                duration=args.duration_sec,
                window=args.sync_window_sec,
                min_gap=args.min_gap_sec,
            )
            path = output_dir / f"random_matched_d{draw}-s{seed}.jsonl"
            _write_schedule(path, starts, duration=args.duration_sec)
            manifest.append(
                {
                    "policy": f"random_matched_d{draw}",
                    "seed": seed,
                    "draw": draw,
                    "count": count,
                    "path": str(path),
                }
            )

        no_sync_path = output_dir / f"no_sync-s{seed}.jsonl"
        _write_schedule(no_sync_path, [], duration=args.duration_sec)
        manifest.append(
            {
                "policy": "no_sync",
                "seed": seed,
                "draw": None,
                "count": 0,
                "path": str(no_sync_path),
            }
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(manifest_path)
    return 0


def _policy_workspace(policy: str, seed: int) -> Path:
    match policy, seed:
        case "wa_diloco", 23 | 29 | 31:
            return Path(f"workspaces/stage2/wa-current-s{seed}/events/coordinator.jsonl")
        case "wa_diloco", 37 | 41:
            return Path(f"workspaces/stage5/wa-current-s{seed}/events/coordinator.jsonl")
    raise KeyError((policy, seed))


def _read_sync_starts(path: Path) -> tuple[list[float], float]:
    records = []
    with path.open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    elapsed = [
        float(record["elapsed_sec"])
        for record in records
        if (record.get("sync_active") or record.get("event") in {"sync", "merged"})
        and record.get("elapsed_sec") is not None
    ]
    if elapsed:
        duration = max(float(record.get("elapsed_sec", 0.0)) for record in records)
        return sorted(elapsed), duration

    ts_values = [int(record["ts_ms"]) for record in records if record.get("ts_ms")]
    start_ts = next(
        (int(record["ts_ms"]) for record in records if record.get("event") == "coordinator_start" and record.get("ts_ms")),
        min(ts_values) if ts_values else None,
    )
    if start_ts is None:
        return [], 0.0
    starts = [
        (int(record["ts_ms"]) - start_ts) / 1000.0
        for record in records
        if record.get("event") == "merged" and record.get("ts_ms")
    ]
    duration = (max(ts_values) - start_ts) / 1000.0 if ts_values else 0.0
    return sorted(starts), duration


def _write_schedule(path: Path, starts: list[float], *, duration: float) -> None:
    with path.open("w") as f:
        f.write(json.dumps({"event": "coordinator_start", "elapsed_sec": 0.0, "ts_ms": 0}) + "\n")
        for start in starts:
            f.write(
                json.dumps(
                    {
                        "event": "merged",
                        "elapsed_sec": round(float(start), 3),
                        "ts_ms": int(round(float(start) * 1000.0)),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        f.write(
            json.dumps(
                {
                    "event": "coordinator_end",
                    "elapsed_sec": round(float(duration), 3),
                    "ts_ms": int(round(float(duration) * 1000.0)),
                },
                sort_keys=True,
            )
            + "\n"
        )


def _random_starts(
    rng: random.Random,
    *,
    count: int,
    duration: float,
    window: float,
    min_gap: float,
) -> list[float]:
    starts: list[float] = []
    attempts = 0
    limit = max(window, duration - window)
    while len(starts) < count and attempts < 10000:
        attempts += 1
        candidate = rng.uniform(0.0, limit)
        if all(abs(candidate - existing) >= min_gap for existing in starts):
            starts.append(candidate)
    if len(starts) < count:
        grid = [i * (duration - window) / max(1, count - 1) for i in range(count)]
        starts = grid[:count]
    return sorted(starts)


def _calibrated_starts(
    *,
    count: int,
    duration: float,
    window: float,
    min_gap: float,
    burst_period: float,
    burst_window: float,
    burst_phase: float,
) -> list[float]:
    candidates = [i * 0.5 for i in range(int(max(0.0, duration - window) * 2) + 1)]
    chosen: list[float] = []
    targets = [(index + 0.5) * duration / count for index in range(count)]
    for target in targets:
        scored = [
            (
                _sync_cost(
                    start,
                    window=window,
                    burst_period=burst_period,
                    burst_window=burst_window,
                    burst_phase=burst_phase,
                ),
                abs(start - target) / duration,
                start,
            )
            for start in candidates
            if all(abs(start - existing) >= min_gap for existing in chosen)
        ]
        scored.sort()
        if scored:
            chosen.append(scored[0][2])
    if len(chosen) < count:
        scored = [
            (
                _sync_cost(
                    start,
                    window=window,
                    burst_period=burst_period,
                    burst_window=burst_window,
                    burst_phase=burst_phase,
                ),
                start,
            )
            for start in candidates
        ]
        scored.sort()
        for _, start in scored:
            if start not in chosen:
                chosen.append(start)
            if len(chosen) == count:
                break
    return sorted(chosen)


def _sync_cost(
    start: float,
    *,
    window: float,
    burst_period: float,
    burst_window: float,
    burst_phase: float,
) -> float:
    samples = 32
    busy = 0
    for index in range(samples):
        t = start + window * (index + 0.5) / samples
        phase = (t + burst_phase) % burst_period
        if phase < burst_window:
            busy += 1
    return busy / samples


if __name__ == "__main__":
    raise SystemExit(main())

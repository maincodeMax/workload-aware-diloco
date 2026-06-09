#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

from wa_diloco.summarize_results import _serving_metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-events", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--replay-workspaces", nargs="*", default=[])
    parser.add_argument("--period-sec", type=float, default=300.0)
    parser.add_argument("--interval-sec", type=float, default=5.0)
    parser.add_argument("--requests-per-interval", type=int, default=32)
    parser.add_argument("--sync-window-sec", type=float, default=8.0)
    parser.add_argument("--slo-multiplier", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    real_events = _read_jsonl(Path(args.real_events))
    calibration = _fit_calibration(real_events, slo_multiplier=args.slo_multiplier)
    calibrated_sidecar_events = _calibrated_events_for_trace(real_events, calibration, seed=args.seed)
    original_sidecar_events = _original_events_for_trace(real_events, seed=args.seed)

    _write_jsonl(output_dir / "calibrated_sidecar_events.jsonl", calibrated_sidecar_events)
    _write_jsonl(output_dir / "original_sidecar_events.jsonl", original_sidecar_events)

    replay_summaries = {}
    for workspace_raw in args.replay_workspaces:
        workspace = Path(workspace_raw)
        events = _replay_workspace(
            workspace,
            calibration,
            period_sec=args.period_sec,
            interval_sec=args.interval_sec,
            requests_per_interval=args.requests_per_interval,
            sync_window_sec=args.sync_window_sec,
            seed=args.seed,
        )
        target = workspace / "serving_calibrated" / "events.jsonl"
        _write_jsonl(target, events)
        replay_summaries[workspace.name] = _serving_metrics(target)

    summary = {
        "calibration": calibration,
        "original_simulator": _summary(original_sidecar_events),
        "calibrated_simulator": _summary(calibrated_sidecar_events),
        "real": _summary(real_events),
        "replay_workspaces": replay_summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_markdown(output_dir / "summary.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _fit_calibration(events: list[dict], *, slo_multiplier: float) -> dict:
    low_inactive = [
        float(event["latency_ms"])
        for event in events
        if not event.get("high_pressure") and not event.get("sync_active")
    ]
    high_inactive = [
        float(event["latency_ms"])
        for event in events
        if event.get("high_pressure") and not event.get("sync_active")
    ]
    sync_active = [float(event["latency_ms"]) for event in events if event.get("sync_active")]
    sync_inactive = [float(event["latency_ms"]) for event in events if not event.get("sync_active")]

    low_p50 = _percentile(low_inactive or [float(event["latency_ms"]) for event in events], 50) or 0.0
    low_p95 = _percentile(low_inactive or sync_inactive or [float(event["latency_ms"]) for event in events], 95) or low_p50
    high_p95 = _percentile(high_inactive, 95) or low_p95
    sync_p95 = _percentile(sync_active, 95) or low_p95
    inactive_p95 = _percentile(sync_inactive, 95) or low_p95

    pressure_penalty_ms = max(0.0, high_p95 - low_p95)
    sync_penalty_ms = max(0.0, sync_p95 - inactive_p95)
    residuals = []
    for event in events:
        predicted = _predict_latency_no_jitter(
            event,
            base_latency_ms=low_p50,
            pressure_penalty_ms=pressure_penalty_ms,
            sync_penalty_ms=sync_penalty_ms,
        )
        residuals.append(float(event["latency_ms"]) - predicted)
    residuals.sort()

    return {
        "base_latency_ms": low_p50,
        "low_inactive_p95_ms": low_p95,
        "pressure_penalty_ms": pressure_penalty_ms,
        "sync_penalty_ms": sync_penalty_ms,
        "slo_ms": low_p95 * slo_multiplier,
        "slo_multiplier": slo_multiplier,
        "residuals_ms": residuals,
    }


def _calibrated_events_for_trace(events: list[dict], calibration: dict, *, seed: int) -> list[dict]:
    rng = random.Random(seed)
    return [
        _with_latency(
            event,
            _predict_latency_no_jitter(
                event,
                base_latency_ms=float(calibration["base_latency_ms"]),
                pressure_penalty_ms=float(calibration["pressure_penalty_ms"]),
                sync_penalty_ms=float(calibration["sync_penalty_ms"]),
            )
            + _sample_residual(rng, calibration["residuals_ms"]),
            slo_ms=float(calibration["slo_ms"]),
            backend="calibrated_simulator",
        )
        for event in events
    ]


def _original_events_for_trace(events: list[dict], *, seed: int) -> list[dict]:
    rng = random.Random(seed)
    output = []
    for event in events:
        pressure = _pressure_from_event(event)
        latency = _original_latency_ms(rng, pressure, sync_active=bool(event.get("sync_active")))
        output.append(_with_latency(event, latency, slo_ms=120.0, backend="original_simulator"))
    return output


def _replay_workspace(
    workspace: Path,
    calibration: dict,
    *,
    period_sec: float,
    interval_sec: float,
    requests_per_interval: int,
    sync_window_sec: float,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed + sum(ord(ch) for ch in workspace.name))
    coordinator = _read_jsonl(workspace / "events" / "coordinator.jsonl")
    ts_values = [int(event["ts_ms"]) for event in coordinator if event.get("ts_ms")]
    if not ts_values:
        return []
    start_ms = min(ts_values)
    end_ms = max(ts_values)
    merge_ms = [
        int(event["ts_ms"])
        for event in coordinator
        if event.get("event") == "merged" and event.get("ts_ms")
    ]
    events = []
    now_ms = start_ms
    while now_ms <= end_ms:
        elapsed = (now_ms - start_ms) / 1000.0
        pressure = _pressure(elapsed, period_sec)
        sync_active = any(0 <= (now_ms - merged) / 1000.0 <= sync_window_sec for merged in merge_ms)
        template = {
            "elapsed_sec": elapsed,
            "high_pressure": max(pressure.values()) >= 0.6,
            "max_pressure": max(pressure.values()),
            "sync_active": sync_active,
            "ts": now_ms / 1000.0,
            **pressure,
        }
        for _ in range(requests_per_interval):
            latency = _predict_latency_no_jitter(
                template,
                base_latency_ms=float(calibration["base_latency_ms"]),
                pressure_penalty_ms=float(calibration["pressure_penalty_ms"]),
                sync_penalty_ms=float(calibration["sync_penalty_ms"]),
            ) + _sample_residual(rng, calibration["residuals_ms"])
            events.append(
                _with_latency(
                    template,
                    latency,
                    slo_ms=float(calibration["slo_ms"]),
                    backend="calibrated_replay",
                )
            )
        now_ms += int(interval_sec * 1000)
    return events


def _predict_latency_no_jitter(
    event: dict,
    *,
    base_latency_ms: float,
    pressure_penalty_ms: float,
    sync_penalty_ms: float,
) -> float:
    max_pressure = float(event.get("max_pressure", max(_pressure_from_event(event).values())))
    pressure_scale = max(0.0, min(1.0, (max_pressure - 0.6) / 0.4))
    sync_scale = (0.35 + 0.65 * max_pressure) if event.get("sync_active") else 0.0
    return base_latency_ms + pressure_penalty_ms * pressure_scale + sync_penalty_ms * sync_scale


def _pressure(elapsed: float, period_sec: float) -> dict[str, float]:
    phase = (math.sin(2 * math.pi * elapsed / period_sec) + 1.0) / 2.0
    return {
        "checkpoint_pressure": 0.8 if int(elapsed // period_sec) % 4 == 3 else 0.0,
        "inference_pressure": 0.1 + 0.8 * max(0.0, phase - 0.35) / 0.65,
        "network_pressure": 0.2 + 0.6 * phase,
    }


def _pressure_from_event(event: dict) -> dict[str, float]:
    return {
        "checkpoint_pressure": float(event.get("checkpoint_pressure", 0.0)),
        "inference_pressure": float(event.get("inference_pressure", 0.0)),
        "network_pressure": float(event.get("network_pressure", 0.0)),
    }


def _original_latency_ms(rng: random.Random, pressure: dict[str, float], *, sync_active: bool) -> float:
    network = pressure["network_pressure"]
    inference = pressure["inference_pressure"]
    checkpoint = pressure["checkpoint_pressure"]
    penalty = 35.0 * (0.35 + 0.65 * max(network, inference)) if sync_active else 0.0
    jitter = rng.lognormvariate(0.0, 0.18) * 6.0
    return 55.0 + 24.0 * network + 34.0 * inference + 16.0 * checkpoint + penalty + jitter


def _with_latency(template: dict, latency_ms: float, *, slo_ms: float, backend: str) -> dict:
    latency_ms = max(0.0, latency_ms)
    return {
        **template,
        "backend": backend,
        "latency_ms": latency_ms,
        "slo_ms": slo_ms,
        "slo_violation": latency_ms > slo_ms,
    }


def _sample_residual(rng: random.Random, residuals: list[float]) -> float:
    if not residuals:
        return 0.0
    return residuals[rng.randrange(len(residuals))]


def _summary(events: list[dict]) -> dict:
    latencies = [float(event["latency_ms"]) for event in events]
    high = [float(event["latency_ms"]) for event in events if event.get("high_pressure")]
    low = [float(event["latency_ms"]) for event in events if not event.get("high_pressure")]
    sync_active = [float(event["latency_ms"]) for event in events if event.get("sync_active")]
    sync_inactive = [float(event["latency_ms"]) for event in events if not event.get("sync_active")]
    violations = sum(1 for event in events if event.get("slo_violation"))
    return {
        "high_low_p95_ratio": _ratio(_percentile(high, 95), _percentile(low, 95)),
        "high_pressure_p95_ms": _percentile(high, 95),
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p95_ms": _percentile(latencies, 95),
        "latency_p99_ms": _percentile(latencies, 99),
        "low_pressure_p95_ms": _percentile(low, 95),
        "requests": len(events),
        "slo_ms": events[0].get("slo_ms") if events else None,
        "slo_violation_rate": violations / len(events) if events else None,
        "sync_inactive_p95_ratio": _ratio(_percentile(sync_active, 95), _percentile(sync_inactive, 95)),
        "sync_active_p95_ms": _percentile(sync_active, 95),
        "sync_inactive_p95_ms": _percentile(sync_inactive, 95),
    }


def _write_markdown(path: Path, summary: dict) -> None:
    lines = ["## Serving Simulator Calibration", ""]
    lines.append("| trace | p50_ms | p95_ms | p99_ms | high/low p95 | sync/inactive p95 | slo_viol_% |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name in ["real", "original_simulator", "calibrated_simulator"]:
        item = summary[name]
        lines.append(
            f"| {name} | {_fmt(item.get('latency_p50_ms'))} | {_fmt(item.get('latency_p95_ms'))} | "
            f"{_fmt(item.get('latency_p99_ms'))} | {_fmt(item.get('high_low_p95_ratio'))} | "
            f"{_fmt(item.get('sync_inactive_p95_ratio'))} | "
            f"{_fmt(100.0 * item.get('slo_violation_rate', 0.0))} |"
        )
    lines.extend(["", "## Replay Workspaces", ""])
    lines.append("| run | p99_ms | slo_viol_% | sync_active_viol_% |")
    lines.append("| --- | ---: | ---: | ---: |")
    for run, item in sorted(summary["replay_workspaces"].items()):
        lines.append(
            f"| {run} | {_fmt(item.get('serving_latency_p99_ms'))} | "
            f"{_fmt(100.0 * item.get('serving_slo_violation_rate', 0.0))} | "
            f"{_fmt(100.0 * item.get('serving_sync_active_violation_rate', 0.0))} |"
        )
    path.write_text("\n".join(lines) + "\n")


def _ratio(numerator, denominator):
    return numerator / denominator if numerator is not None and denominator else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _fmt(value) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())

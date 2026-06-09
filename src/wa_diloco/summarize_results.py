from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _read_jsonl(path: Path):
    if not path.exists():
        return []
    records = []
    with path.open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def run(workspace: str | Path) -> None:
    root = Path(workspace)
    coordinator_events = _read_jsonl(root / "events" / "coordinator.jsonl")
    merged = [event for event in coordinator_events if event.get("event") == "merged"]
    cycles = [event for event in coordinator_events if event.get("event") == "control_cycle"]
    reasons = Counter()
    for cycle in cycles:
        for decision in cycle.get("decisions", []):
            reasons[decision.get("reason", "unknown")] += 1
    sync_pressure = _sync_pressure_metrics(cycles, coordinator_events)

    learner_reports = []
    for path in sorted((root / "reports").glob("cycle_*/learner_*.json")):
        with path.open() as f:
            learner_reports.append(json.load(f))

    reported_cumulative_tokens = sum(
        int(report.get("tokens_since_sync", 0)) for report in learner_reports
    )
    actual_tokens = _actual_tokens_from_reports(learner_reports)
    mean_loss_end = None
    loss_values = [report.get("loss_end") for report in learner_reports if report.get("loss_end")]
    if loss_values:
        mean_loss_end = sum(float(value) for value in loss_values) / len(loss_values)

    duration_sec = None
    if coordinator_events:
        ts_values = [event.get("ts_ms") for event in coordinator_events if event.get("ts_ms")]
        if ts_values:
            duration_sec = (max(ts_values) - min(ts_values)) / 1000.0

    payload = {
        "workspace": str(root),
        "control_cycles": len(cycles),
        "global_rounds": len(merged),
        "decision_reasons": dict(reasons),
        "reports": len(learner_reports),
        "actual_tokens_processed_estimate": actual_tokens,
        "total_reported_tokens_since_sync_sum": reported_cumulative_tokens,
        "mean_loss_end": mean_loss_end,
        "coordinator_duration_sec": duration_sec,
        **sync_pressure,
        **_serving_metrics(root / "serving" / "events.jsonl"),
    }

    print(json.dumps(payload, indent=2, sort_keys=True))


def _actual_tokens_from_reports(reports: list[dict]) -> int:
    actual_tokens = 0
    last_by_learner_round: dict[tuple[int, int], int] = {}
    for report in sorted(
        reports,
        key=lambda item: (
            int(item.get("learner_id", -1)),
            int(item.get("control_cycle", -1)),
        ),
    ):
        learner_id = int(report.get("learner_id", -1))
        global_round = int(report.get("global_round", -1))
        tokens_since_sync = int(report.get("tokens_since_sync", 0))
        key = (learner_id, global_round)
        previous = last_by_learner_round.get(key, 0)
        actual_tokens += max(0, tokens_since_sync - previous)
        last_by_learner_round[key] = tokens_since_sync
    return actual_tokens


def _sync_pressure_metrics(cycles: list[dict], coordinator_events: list[dict]) -> dict:
    merge_times = [
        int(event["ts_ms"])
        for event in coordinator_events
        if event.get("event") == "merged" and event.get("ts_ms")
    ]
    sync_cycles = []
    high_pressure_sync_cycles = []
    sync_overlap_ms = 0
    merge_index = 0
    for cycle in cycles:
        ready_reports = int(cycle.get("ready_reports", 0))
        sync_ready = int(cycle.get("sync_ready", 0))
        if ready_reports <= 0 or sync_ready < ready_reports:
            continue
        sync_cycles.append(cycle)
        pressure = cycle.get("pressure", {})
        high_pressure = max(
            float(pressure.get("network", 0.0)),
            float(pressure.get("inference", 0.0)),
            float(pressure.get("checkpoint", 0.0)),
        ) >= 0.6
        if high_pressure:
            high_pressure_sync_cycles.append(cycle)

        cycle_ts = int(cycle.get("ts_ms", 0))
        while merge_index < len(merge_times) and merge_times[merge_index] < cycle_ts:
            merge_index += 1
        if merge_index < len(merge_times):
            duration = max(0, merge_times[merge_index] - cycle_ts)
            if high_pressure:
                sync_overlap_ms += duration
            merge_index += 1

    return {
        "sync_control_cycles": len(sync_cycles),
        "high_pressure_sync_control_cycles": len(high_pressure_sync_cycles),
        "high_pressure_sync_fraction": (
            len(high_pressure_sync_cycles) / len(sync_cycles) if sync_cycles else None
        ),
        "high_pressure_sync_overlap_sec": sync_overlap_ms / 1000.0,
    }


def _serving_metrics(path: Path) -> dict:
    if not path.exists():
        return {}

    latencies = []
    violations = 0
    sync_active_requests = 0
    sync_active_violations = 0
    high_pressure_requests = 0
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            event = json.loads(line)
            latency = float(event["latency_ms"])
            latencies.append(latency)
            violation = bool(event.get("slo_violation"))
            violations += int(violation)
            sync_active = bool(event.get("sync_active"))
            sync_active_requests += int(sync_active)
            sync_active_violations += int(sync_active and violation)
            high_pressure = max(
                float(event.get("network_pressure", 0.0)),
                float(event.get("inference_pressure", 0.0)),
                float(event.get("checkpoint_pressure", 0.0)),
            ) >= 0.6
            high_pressure_requests += int(high_pressure)

    if not latencies:
        return {}

    return {
        "serving_requests": len(latencies),
        "serving_latency_p50_ms": _percentile(latencies, 50),
        "serving_latency_p95_ms": _percentile(latencies, 95),
        "serving_latency_p99_ms": _percentile(latencies, 99),
        "serving_slo_violations": violations,
        "serving_slo_violation_rate": violations / len(latencies),
        "serving_sync_active_requests": sync_active_requests,
        "serving_sync_active_violation_rate": (
            sync_active_violations / sync_active_requests if sync_active_requests else 0.0
        ),
        "serving_high_pressure_request_fraction": high_pressure_requests / len(latencies),
    }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", help="Experiment workspace directory")
    args = parser.parse_args(argv)
    run(args.workspace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

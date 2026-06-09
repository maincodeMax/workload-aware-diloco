#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from wa_diloco.summarize_results import (
    _actual_tokens_from_reports,
    _serving_metrics,
    _sync_pressure_metrics,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspaces", nargs="+", help="Experiment workspace paths")
    parser.add_argument("--mode", choices=["full", "tokens", "wall-clock"], default="full")
    parser.add_argument("--serving-subdir", default="serving")
    args = parser.parse_args()

    runs = [_load_run(Path(path), serving_subdir=args.serving_subdir) for path in args.workspaces]
    if args.mode == "tokens":
        target_tokens = min(run["actual_tokens"] for run in runs)
        rows = [_summarize_prefix(run, target_tokens=target_tokens) for run in runs]
        title = f"Token-matched prefix table, target>={target_tokens:,} tokens"
    elif args.mode == "wall-clock":
        target_duration = min(run["duration_sec"] for run in runs if run["duration_sec"] is not None)
        rows = [_summarize_prefix(run, target_duration_sec=target_duration) for run in runs]
        title = f"Wall-clock-matched prefix table, target={target_duration:.1f}s"
    else:
        rows = [_summarize_prefix(run) for run in runs]
        title = "Full-run table"

    print(f"## {title}\n")
    print(
        "| run | rounds | cycles | tokens | duration_s | mean_loss_end | p99_ms | slo_viol_% | high_pressure_sync_% | pressure_deferrals |"
    )
    print(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for row in rows:
        print(
            "| {run} | {rounds} | {cycles} | {tokens} | {duration:.1f} | {loss:.4f} | {p99:.1f} | {viol:.2f} | {hps:.2f} | {deferrals} |".format(
                run=row["run"],
                rounds=row["global_rounds"],
                cycles=row["control_cycles"],
                tokens=row["actual_tokens_processed_estimate"],
                duration=row["coordinator_duration_sec"] or 0.0,
                loss=row["mean_loss_end"] or 0.0,
                p99=row.get("serving_latency_p99_ms", 0.0),
                viol=100.0 * row.get("serving_slo_violation_rate", 0.0),
                hps=100.0 * (row.get("high_pressure_sync_fraction") or 0.0),
                deferrals=_deferral_count(row.get("decision_reasons", {})),
            )
        )
    return 0


def _load_run(workspace: Path, serving_subdir: str = "serving") -> dict:
    coordinator = _read_jsonl(workspace / "events" / "coordinator.jsonl")
    reports = []
    for path in sorted((workspace / "reports").glob("cycle_*/learner_*.json")):
        with path.open() as f:
            reports.append(json.load(f))
    serving = _read_jsonl(workspace / serving_subdir / "events.jsonl")
    ts_values = [event["ts_ms"] for event in coordinator if event.get("ts_ms")]
    start_ms = min(ts_values) if ts_values else None
    end_ms = max(ts_values) if ts_values else None
    return {
        "workspace": workspace,
        "serving_subdir": serving_subdir,
        "coordinator": coordinator,
        "reports": reports,
        "serving": serving,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "duration_sec": ((end_ms - start_ms) / 1000.0 if start_ms and end_ms else None),
        "actual_tokens": _actual_tokens_from_reports(reports),
    }


def _summarize_prefix(
    run: dict,
    *,
    target_tokens: int | None = None,
    target_duration_sec: float | None = None,
) -> dict:
    start_ms = run["start_ms"]
    cutoff_ms = run["end_ms"]
    cutoff_cycle = None
    if start_ms is not None and target_duration_sec is not None:
        cutoff_ms = start_ms + int(target_duration_sec * 1000)
        cutoff_cycle = _cycle_cutoff_for_time(run["coordinator"], cutoff_ms)
    elif target_tokens is not None:
        cutoff_cycle = _token_cutoff_cycle(run["reports"], target_tokens)
        cutoff_ms = _cycle_cutoff_ms(run["coordinator"], cutoff_cycle)

    included_cycles = _included_cycles(run["coordinator"], cutoff_ms, cutoff_cycle)

    coordinator = [
        event
        for event in run["coordinator"]
        if cutoff_ms is None or int(event.get("ts_ms", 0)) <= cutoff_ms
    ]
    reports = [
        report
        for report in run["reports"]
        if _include_report(report, cutoff_ms=cutoff_ms, included_cycles=included_cycles)
    ]
    serving = [
        event
        for event in run["serving"]
        if cutoff_ms is None or int(float(event.get("ts", 0.0)) * 1000) <= cutoff_ms
    ]

    cycles = [event for event in coordinator if event.get("event") == "control_cycle"]
    merged = [event for event in coordinator if event.get("event") == "merged"]
    reasons = Counter()
    for cycle in cycles:
        for decision in cycle.get("decisions", []):
            reasons[decision.get("reason", "unknown")] += 1

    loss_values = [float(report["loss_end"]) for report in reports if report.get("loss_end")]
    duration_sec = None
    ts_values = [event.get("ts_ms") for event in coordinator if event.get("ts_ms")]
    if ts_values:
        duration_sec = (max(ts_values) - min(ts_values)) / 1000.0

    serving_path = run["workspace"] / run.get("serving_subdir", "serving") / ".prefix-events.jsonl"
    if serving:
        with serving_path.open("w") as f:
            for event in serving:
                f.write(json.dumps(event) + "\n")
        serving_metrics = _serving_metrics(serving_path)
        serving_path.unlink(missing_ok=True)
    else:
        serving_metrics = {}

    return {
        "run": run["workspace"].name,
        "control_cycles": len(cycles),
        "global_rounds": len(merged),
        "decision_reasons": dict(reasons),
        "reports": len(reports),
        "actual_tokens_processed_estimate": _actual_tokens_from_reports(reports),
        "mean_loss_end": sum(loss_values) / len(loss_values) if loss_values else None,
        "coordinator_duration_sec": duration_sec,
        **_sync_pressure_metrics(cycles, coordinator),
        **serving_metrics,
    }


def _include_report(
    report: dict,
    *,
    cutoff_ms: int | None,
    included_cycles: set[int] | None,
) -> bool:
    if cutoff_ms is None:
        return True
    if "ts_ms" in report:
        return int(report.get("ts_ms", 0)) <= cutoff_ms
    if included_cycles is None:
        return True
    return int(report.get("control_cycle", -1)) in included_cycles


def _included_cycles(
    coordinator: list[dict],
    cutoff_ms: int | None,
    cutoff_cycle: int | None,
) -> set[int] | None:
    if cutoff_ms is None and cutoff_cycle is None:
        return None
    cycles = set()
    for event in coordinator:
        if event.get("event") != "control_cycle":
            continue
        cycle = int(event.get("control_cycle", -1))
        if cutoff_cycle is not None and cycle <= cutoff_cycle:
            cycles.add(cycle)
        elif cutoff_ms is not None and int(event.get("ts_ms", 0)) <= cutoff_ms:
            cycles.add(cycle)
    return cycles


def _cycle_cutoff_for_time(coordinator: list[dict], cutoff_ms: int) -> int | None:
    cycles = [
        int(event.get("control_cycle", -1))
        for event in coordinator
        if event.get("event") == "control_cycle" and int(event.get("ts_ms", 0)) <= cutoff_ms
    ]
    return max(cycles) if cycles else None


def _cycle_cutoff_ms(coordinator: list[dict], cutoff_cycle: int | None) -> int | None:
    if cutoff_cycle is None:
        return None
    cutoff_ms = None
    events = list(enumerate(coordinator))
    for index, event in events:
        if (
            event.get("event") == "control_cycle"
            and int(event.get("control_cycle", -1)) == cutoff_cycle
        ):
            cutoff_ms = int(event.get("ts_ms", 0))
            if index + 1 < len(coordinator) and coordinator[index + 1].get("event") == "merged":
                cutoff_ms = int(coordinator[index + 1].get("ts_ms", cutoff_ms))
            break
    return cutoff_ms


def _token_cutoff_cycle(reports: list[dict], target_tokens: int) -> int | None:
    cumulative = 0
    last_by_learner_round: dict[tuple[int, int], int] = {}
    for report in sorted(
        reports,
        key=lambda item: (
            int(item.get("control_cycle", -1)),
            int(item.get("learner_id", -1)),
        ),
    ):
        learner_id = int(report.get("learner_id", -1))
        global_round = int(report.get("global_round", -1))
        tokens_since_sync = int(report.get("tokens_since_sync", 0))
        key = (learner_id, global_round)
        previous = last_by_learner_round.get(key, 0)
        cumulative += max(0, tokens_since_sync - previous)
        last_by_learner_round[key] = tokens_since_sync
        if cumulative >= target_tokens:
            return int(report.get("control_cycle", -1))
    return max((int(report.get("control_cycle", -1)) for report in reports), default=None)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _deferral_count(reasons: dict[str, int]) -> int:
    deferral_reasons = {
        "load_deferral",
        "pressure_exceeds_value",
        "pressure_gate_deferral",
        "random_deferral",
    }
    return sum(int(reasons.get(reason, 0)) for reason in deferral_reasons)


if __name__ == "__main__":
    raise SystemExit(main())

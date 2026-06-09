#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from compare_workspaces import _deferral_count, _load_run, _summarize_prefix


GROUPS = {
    "fixed_h_256": "Fixed-H 256",
    "fixed_h_512": "Fixed-H 512",
    "pressure_gate": "Pressure-gate",
    "random_deferral": "Random deferral",
    "wa_current": "WA-DiLoCo",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspaces", nargs="+")
    parser.add_argument("--output-dir", default="results/stage4")
    parser.add_argument("--serving-subdir", default="serving")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = [
        _load_run(Path(path), serving_subdir=args.serving_subdir)
        for path in args.workspaces
        if Path(path).exists()
    ]
    comparison_runs = [run for run in runs if _group_name(run["workspace"].name) in GROUPS]
    if not comparison_runs:
        raise SystemExit("no recognized workspaces")

    token_target = min(run["actual_tokens"] for run in comparison_runs)
    wall_target = min(
        run["duration_sec"] for run in comparison_runs if run["duration_sec"] is not None
    )
    token_rows = [_summarize_prefix(run, target_tokens=token_target) for run in comparison_runs]
    wall_rows = [
        _summarize_prefix(run, target_duration_sec=wall_target) for run in comparison_runs
    ]
    full_rows = [_summarize_prefix(run) for run in comparison_runs]

    _write_aggregate(
        output_dir / "rescue-token-matched.md",
        title=f"Stage 4 Token-Matched Baselines, target>={token_target:,} tokens",
        rows=token_rows,
    )
    _write_aggregate(
        output_dir / "rescue-wall-clock-matched.md",
        title=f"Stage 4 Wall-Clock-Matched Baselines, target={wall_target:.1f}s",
        rows=wall_rows,
    )
    _write_full_table(output_dir / "rescue-full.md", full_rows)
    _write_eval_table(output_dir / "rescue-eval.md", comparison_runs)
    _write_json_summary(output_dir / "rescue-summary.json", token_rows, wall_rows, full_rows)
    return 0


def _group_name(name: str) -> str:
    if name.startswith("fixed-h-256"):
        return "fixed_h_256"
    if name.startswith("fixed-h-512"):
        return "fixed_h_512"
    if name.startswith("pressure-gate"):
        return "pressure_gate"
    if name.startswith("random-deferral"):
        return "random_deferral"
    if name.startswith("wa-current"):
        return "wa_current"
    return "other"


def _write_aggregate(path: Path, *, title: str, rows: list[dict]) -> None:
    metrics = [
        ("tokens", "actual_tokens_processed_estimate", 1.0),
        ("duration_s", "coordinator_duration_sec", 1.0),
        ("mean_loss_end", "mean_loss_end", 1.0),
        ("p99_ms", "serving_latency_p99_ms", 1.0),
        ("slo_viol_%", "serving_slo_violation_rate", 100.0),
        ("high_pressure_sync_%", "high_pressure_sync_fraction", 100.0),
        ("deferrals", "deferrals", 1.0),
    ]
    grouped = _group_rows(rows)
    ordered_groups = [group for group in GROUPS if grouped.get(group)]

    lines = [f"## {title}", ""]
    lines.append("| metric | " + " | ".join(GROUPS[group] for group in ordered_groups) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in ordered_groups) + " |")
    for label, key, scale in metrics:
        values = []
        for group in ordered_groups:
            group_values = [
                _metric(row, key) * scale
                for row in grouped[group]
                if _metric(row, key) is not None
            ]
            values.append(_fmt_mean_sd(*_mean_sd(group_values)))
        lines.append("| " + label + " | " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n")


def _write_full_table(path: Path, rows: list[dict]) -> None:
    lines = ["## Stage 4 Full Runs", ""]
    lines.append(
        "| run | tokens | duration_s | mean_loss_end | p99_ms | slo_viol_% | "
        "high_pressure_sync_% | deferrals |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in sorted(rows, key=lambda item: item["run"]):
        lines.append(
            f"| {row['run']} | {row['actual_tokens_processed_estimate']} | "
            f"{_fmt(row.get('coordinator_duration_sec'))} | {_fmt(row.get('mean_loss_end'))} | "
            f"{_fmt(row.get('serving_latency_p99_ms'))} | "
            f"{_fmt(100.0 * row.get('serving_slo_violation_rate', 0.0))} | "
            f"{_fmt(100.0 * (row.get('high_pressure_sync_fraction') or 0.0))} | "
            f"{_deferral_count(row.get('decision_reasons', {}))} |"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_eval_table(path: Path, runs: list[dict]) -> None:
    lines = ["## Stage 4 Held-Out Eval", ""]
    lines.append("| run | eval_loss | perplexity | eval_tokens | eval_batches | checkpoint |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for run in sorted(runs, key=lambda item: item["workspace"].name):
        eval_path = run["workspace"] / "eval" / "final.json"
        if not eval_path.exists():
            continue
        payload = json.loads(eval_path.read_text())
        lines.append(
            f"| {run['workspace'].name} | {_fmt(payload.get('eval_loss'))} | "
            f"{_fmt(payload.get('eval_perplexity'))} | {payload.get('eval_tokens', 0)} | "
            f"{payload.get('eval_batches', 0)} | {Path(payload.get('checkpoint', '')).name} |"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_json_summary(
    path: Path,
    token_rows: list[dict],
    wall_rows: list[dict],
    full_rows: list[dict],
) -> None:
    payload = {
        "full": _json_rows(full_rows),
        "token_matched": _json_rows(token_rows),
        "wall_clock_matched": _json_rows(wall_rows),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _json_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            "deferrals": _deferral_count(row.get("decision_reasons", {})),
            "duration_s": row.get("coordinator_duration_sec"),
            "group": _group_name(row["run"]),
            "high_pressure_sync_fraction": row.get("high_pressure_sync_fraction"),
            "mean_loss_end": row.get("mean_loss_end"),
            "p99_ms": row.get("serving_latency_p99_ms"),
            "run": row["run"],
            "slo_violation_rate": row.get("serving_slo_violation_rate"),
            "tokens": row.get("actual_tokens_processed_estimate"),
        }
        for row in rows
    ]


def _group_rows(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(_group_name(row["run"]), []).append(row)
    return grouped


def _metric(row: dict, key: str):
    if key == "deferrals":
        return _deferral_count(row.get("decision_reasons", {}))
    return row.get(key)


def _mean_sd(values: list[float]) -> tuple[float | None, float | None]:
    clean = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    if not clean:
        return None, None
    if len(clean) == 1:
        return clean[0], 0.0
    return statistics.mean(clean), statistics.stdev(clean)


def _fmt_mean_sd(mean: float | None, sd: float | None) -> str:
    if mean is None:
        return ""
    return f"{_fmt(mean)}+-{_fmt(sd or 0.0)}"


def _fmt(value) -> str:
    if value is None:
        return ""
    value = float(value)
    if abs(value) >= 1000:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())

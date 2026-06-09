#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from compare_workspaces import _deferral_count, _load_run, _summarize_prefix


GROUPS = {
    "fixed_h": "Fixed-H 256",
    "wa_current": "WA-current",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspaces", nargs="+")
    parser.add_argument("--output-dir", default="results/stage2")
    parser.add_argument("--serving-subdir", default="serving")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = [
        _load_run(Path(path), serving_subdir=args.serving_subdir)
        for path in args.workspaces
        if Path(path).exists()
    ]
    grouped = _group_runs(runs)

    comparison_runs = [run for run in runs if _group_name(run["workspace"].name) in GROUPS]
    token_target = min(run["actual_tokens"] for run in comparison_runs)
    wall_target = min(run["duration_sec"] for run in comparison_runs if run["duration_sec"] is not None)
    token_rows = [_summarize_prefix(run, target_tokens=token_target) for run in comparison_runs]
    wall_rows = [_summarize_prefix(run, target_duration_sec=wall_target) for run in comparison_runs]

    _write_aggregate(
        output_dir / "aggregate-token-matched.md",
        title=f"Token-Matched Aggregate, target>={token_target:,} tokens",
        rows=token_rows,
    )
    _write_aggregate(
        output_dir / "aggregate-wall-clock-matched.md",
        title=f"Wall-Clock-Matched Aggregate, target={wall_target:.1f}s",
        rows=wall_rows,
    )
    _write_ablation(output_dir / "ablation-full.md", grouped)
    _write_eval_table(output_dir / "eval-final.md", runs)
    _write_json_summary(output_dir / "aggregate-summary.json", token_rows, wall_rows, runs)
    return 0


def _group_runs(runs: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for run in runs:
        grouped.setdefault(_group_name(run["workspace"].name), []).append(run)
    return grouped


def _group_name(name: str) -> str:
    if name.startswith("fixed-h-256"):
        return "fixed_h"
    if name.startswith("wa-current"):
        return "wa_current"
    if name.startswith("wa-no-slo"):
        return "no_slo"
    if name.startswith("wa-no-network"):
        return "no_network"
    if name.startswith("wa-no-staleness"):
        return "no_staleness"
    return "other"


def _write_aggregate(path: Path, *, title: str, rows: list[dict]) -> None:
    by_group: dict[str, list[dict]] = {key: [] for key in GROUPS}
    for row in rows:
        group = _group_name(row["run"])
        if group in by_group:
            by_group[group].append(row)

    metrics = [
        ("tokens", "actual_tokens_processed_estimate", 1.0),
        ("duration_s", "coordinator_duration_sec", 1.0),
        ("mean_loss_end", "mean_loss_end", 1.0),
        ("p99_ms", "serving_latency_p99_ms", 1.0),
        ("slo_viol_%", "serving_slo_violation_rate", 100.0),
        ("high_pressure_sync_%", "high_pressure_sync_fraction", 100.0),
        ("pressure_deferrals", "pressure_deferrals", 1.0),
    ]

    lines = [f"## {title}", ""]
    lines.append("| metric | Fixed-H 256 mean±sd | WA-current mean±sd | WA - Fixed |")
    lines.append("| --- | ---: | ---: | ---: |")
    for label, key, scale in metrics:
        fixed_values = [_metric(row, key) * scale for row in by_group["fixed_h"] if _metric(row, key) is not None]
        wa_values = [_metric(row, key) * scale for row in by_group["wa_current"] if _metric(row, key) is not None]
        fixed_mean, fixed_sd = _mean_sd(fixed_values)
        wa_mean, wa_sd = _mean_sd(wa_values)
        delta = wa_mean - fixed_mean if fixed_mean is not None and wa_mean is not None else None
        lines.append(
            f"| {label} | {_fmt_mean_sd(fixed_mean, fixed_sd)} | "
            f"{_fmt_mean_sd(wa_mean, wa_sd)} | {_fmt(delta)} |"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_ablation(path: Path, grouped: dict[str, list[dict]]) -> None:
    order = [
        ("wa_current", "full_wa"),
        ("no_slo", "no_slo_pressure"),
        ("no_network", "no_network_pressure"),
        ("no_staleness", "no_staleness_penalty"),
    ]
    lines = ["## Full-Run Ablation", ""]
    lines.append(
        "| run | tokens | duration_s | mean_loss_end | p99_ms | slo_viol_% | "
        "high_pressure_sync_% | pressure_deferrals |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for group, label in order:
        run = sorted(grouped.get(group, []), key=lambda item: item["workspace"].name)
        if not run:
            continue
        row = _summarize_prefix(run[0])
        lines.append(
            f"| {label} | {row['actual_tokens_processed_estimate']} | "
            f"{_fmt(row.get('coordinator_duration_sec'))} | {_fmt(row.get('mean_loss_end'))} | "
            f"{_fmt(row.get('serving_latency_p99_ms'))} | "
            f"{_fmt(100.0 * row.get('serving_slo_violation_rate', 0.0))} | "
            f"{_fmt(100.0 * (row.get('high_pressure_sync_fraction') or 0.0))} | "
            f"{_deferral_count(row.get('decision_reasons', {}))} |"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_eval_table(path: Path, runs: list[dict]) -> None:
    lines = ["## Final Held-Out Eval", ""]
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


def _write_json_summary(path: Path, token_rows: list[dict], wall_rows: list[dict], runs: list[dict]) -> None:
    payload = {
        "token_matched": _json_rows(token_rows),
        "wall_clock_matched": _json_rows(wall_rows),
        "workspaces": [str(run["workspace"]) for run in runs],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _json_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            "run": row["run"],
            "tokens": row.get("actual_tokens_processed_estimate"),
            "duration_s": row.get("coordinator_duration_sec"),
            "mean_loss_end": row.get("mean_loss_end"),
            "p99_ms": row.get("serving_latency_p99_ms"),
            "slo_violation_rate": row.get("serving_slo_violation_rate"),
            "high_pressure_sync_fraction": row.get("high_pressure_sync_fraction"),
            "pressure_deferrals": _deferral_count(row.get("decision_reasons", {})),
        }
        for row in rows
    ]


def _metric(row: dict, key: str):
    if key == "pressure_deferrals":
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
    return f"{_fmt(mean)}±{_fmt(sd or 0.0)}"


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

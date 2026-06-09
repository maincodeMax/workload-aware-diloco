#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from compare_workspaces import _deferral_count, _load_run, _summarize_prefix


ORDER = [
    "full-wa",
    "no-slo",
    "no-staleness",
    "no-network",
    "no-loss",
    "no-token",
    "pressure-only",
    "progress-only",
    "no-checkpoint",
    "random-matched",
    "pressure-gate-matched",
    "slo-w025",
    "slo-w050",
    "slo-w100",
    "slo-w150",
    "stale-w050",
    "stale-w100",
    "min64-max256",
    "min64-max512",
    "min256-max512",
    "min256-max1024",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspaces", nargs="+")
    parser.add_argument("--serving-subdir", default="serving")
    parser.add_argument("--output-dir", default="results/stage7")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = [
        _load_run(Path(path), serving_subdir=args.serving_subdir)
        for path in args.workspaces
        if Path(path).exists()
    ]
    runs = [run for run in runs if _variant(run["workspace"].name) in ORDER]
    if not runs:
        raise SystemExit("no recognized ablation workspaces")

    token_target = min(run["actual_tokens"] for run in runs)
    wall_target = min(
        run["duration_sec"] for run in runs if run["duration_sec"] is not None
    )
    token_rows = [_summarize_prefix(run, target_tokens=token_target) for run in runs]
    wall_rows = [_summarize_prefix(run, target_duration_sec=wall_target) for run in runs]
    full_rows = [_summarize_prefix(run) for run in runs]

    _write_aggregate(
        output_dir / "ablation-token-matched.md",
        title=f"Stage 7/8 Token-Matched Ablations, target>={token_target:,} tokens",
        rows=token_rows,
    )
    _write_aggregate(
        output_dir / "ablation-wall-clock-matched.md",
        title=f"Stage 7/8 Wall-Clock-Matched Ablations, target={wall_target:.1f}s",
        rows=wall_rows,
    )
    _write_full(output_dir / "ablation-full.md", full_rows)
    _write_json(output_dir / "ablation-summary.json", token_rows, wall_rows, full_rows)
    return 0


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
    groups = [name for name in ORDER if grouped.get(name)]

    lines = [f"## {title}", ""]
    lines.append("| variant | n | " + " | ".join(label for label, _key, _scale in metrics) + " |")
    lines.append("| --- | ---: | " + " | ".join("---:" for _ in metrics) + " |")
    for group in groups:
        values = []
        for _label, key, scale in metrics:
            group_values = [
                _metric(row, key) * scale
                for row in grouped[group]
                if _metric(row, key) is not None
            ]
            values.append(_fmt_mean_sd(*_mean_sd(group_values)))
        lines.append(f"| {group} | {len(grouped[group])} | " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n")


def _write_full(path: Path, rows: list[dict]) -> None:
    lines = ["## Stage 7/8 Full Ablation Runs", ""]
    lines.append(
        "| run | variant | tokens | duration_s | mean_loss_end | p99_ms | slo_viol_% | high_pressure_sync_% | deferrals |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in sorted(rows, key=lambda item: item["run"]):
        lines.append(
            f"| {row['run']} | {_variant(row['run'])} | "
            f"{row.get('actual_tokens_processed_estimate', 0)} | "
            f"{_fmt(row.get('coordinator_duration_sec'))} | "
            f"{_fmt(row.get('mean_loss_end'))} | "
            f"{_fmt(row.get('serving_latency_p99_ms'))} | "
            f"{_fmt(100.0 * row.get('serving_slo_violation_rate', 0.0))} | "
            f"{_fmt(100.0 * (row.get('high_pressure_sync_fraction') or 0.0))} | "
            f"{_deferral_count(row.get('decision_reasons', {}))} |"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_json(
    path: Path,
    token_rows: list[dict],
    wall_rows: list[dict],
    full_rows: list[dict],
) -> None:
    payload = {
        "full": [_json_row(row) for row in full_rows],
        "token_matched": [_json_row(row) for row in token_rows],
        "wall_clock_matched": [_json_row(row) for row in wall_rows],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _json_row(row: dict) -> dict:
    return {
        "deferrals": _deferral_count(row.get("decision_reasons", {})),
        "duration_s": row.get("coordinator_duration_sec"),
        "high_pressure_sync_fraction": row.get("high_pressure_sync_fraction"),
        "mean_loss_end": row.get("mean_loss_end"),
        "p99_ms": row.get("serving_latency_p99_ms"),
        "run": row["run"],
        "slo_violation_rate": row.get("serving_slo_violation_rate"),
        "tokens": row.get("actual_tokens_processed_estimate"),
        "variant": _variant(row["run"]),
    }


def _group_rows(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(_variant(row["run"]), []).append(row)
    return grouped


def _variant(name: str) -> str:
    if "-s" not in name:
        return name
    return name.rsplit("-s", 1)[0]


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

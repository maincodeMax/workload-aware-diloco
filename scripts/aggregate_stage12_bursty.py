#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
from pathlib import Path


POLICY_ORDER = [
    "no_sync",
    "fixed_h_256",
    "fixed_h_512",
    "pressure_gate",
    "random_matched",
    "wa_diloco",
    "calibrated_wa",
]


def main() -> int:
    output_dir = Path("results/stage12")
    rows = []
    for summary_path in sorted((output_dir / "bursty_sync").glob("*-s*/summary.json")):
        run_name = summary_path.parent.name
        policy, seed_text = run_name.rsplit("-s", 1)
        draw = None
        grouped_policy = policy
        if policy.startswith("random_matched_d"):
            draw = int(policy.removeprefix("random_matched_d"))
            grouped_policy = "random_matched"
        summary = json.loads(summary_path.read_text())
        real = summary.get("real", {})
        rows.append(
            {
                "regime": "bursty_sync",
                "policy": grouped_policy,
                "run_policy": policy,
                "seed": int(seed_text),
                "draw": draw,
                "requests": real.get("requests"),
                "p95_ms": real.get("latency_p95_ms"),
                "p99_ms": real.get("latency_p99_ms"),
                "slo_violation_rate": real.get("slo_violation_rate"),
                "tokens_per_sec": real.get("tokens_per_sec"),
                "high_low_p95_ratio": real.get("high_low_p95_ratio"),
                "sync_ratio": real.get("sync_inactive_p95_ratio"),
                "sync_active_count": real.get("sync_active_count"),
                "sync_events": summary.get("sync_event_count"),
                "load_profile": summary.get("load_profile"),
                "quiet_concurrency": summary.get("quiet_concurrency"),
                "busy_concurrency": summary.get("busy_concurrency"),
            }
        )

    baseline_by_seed = {
        row["seed"]: row["slo_violation_rate"]
        for row in rows
        if row["policy"] == "no_sync" and row.get("slo_violation_rate") is not None
    }
    for row in rows:
        baseline = baseline_by_seed.get(row["seed"])
        row["sync_excess_slo_rate"] = (
            row["slo_violation_rate"] - baseline
            if row.get("slo_violation_rate") is not None and baseline is not None
            else None
        )

    grouped = []
    for policy in POLICY_ORDER:
        values = [row for row in rows if row["policy"] == policy]
        if values:
            grouped.append(_aggregate(policy, values))

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "bursty-full.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    (output_dir / "bursty-summary.json").write_text(json.dumps(grouped, indent=2, sort_keys=True) + "\n")
    _write_markdown(output_dir / "bursty-summary.md", grouped)
    print((output_dir / "bursty-summary.md").read_text())
    return 0


def _aggregate(policy: str, rows: list[dict]) -> dict:
    return {
        "policy": policy,
        "n": len(rows),
        "seeds": sorted({row["seed"] for row in rows}),
        "draws": sorted({row["draw"] for row in rows if row["draw"] is not None}),
        "requests": _mean_sd(rows, "requests"),
        "p95_ms": _mean_sd(rows, "p95_ms"),
        "p99_ms": _mean_sd(rows, "p99_ms"),
        "slo_violation_pct": _mean_sd(rows, "slo_violation_rate", scale=100.0),
        "sync_excess_slo_pct": _mean_sd(rows, "sync_excess_slo_rate", scale=100.0),
        "tokens_per_sec": _mean_sd(rows, "tokens_per_sec"),
        "high_low_p95_ratio": _mean_sd(rows, "high_low_p95_ratio"),
        "sync_ratio": _mean_sd(rows, "sync_ratio"),
        "sync_active_count": _mean_sd(rows, "sync_active_count"),
        "sync_events": _mean_sd(rows, "sync_events"),
    }


def _mean_sd(rows: list[dict], key: str, *, scale: float = 1.0) -> dict | None:
    values = [
        float(row[key]) * scale
        for row in rows
        if row.get(key) is not None
    ]
    if not values:
        return None
    return {
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values) if len(values) >= 2 else 0.0,
    }


def _write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Stage 12 bursty sync-heavy vLLM replay",
        "",
        "| policy | n | raw SLO % | sync-excess SLO % | p95 ms | p99 ms | tok/s | sync p95 ratio | sync active reqs | sync events |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {policy} | {n} | {slo} | {excess} | {p95} | {p99} | {tps} | {sync} | {sync_count} | {sync_events} |".format(
                policy=row["policy"],
                n=row["n"],
                slo=_fmt(row["slo_violation_pct"]),
                excess=_fmt(row["sync_excess_slo_pct"]),
                p95=_fmt(row["p95_ms"]),
                p99=_fmt(row["p99_ms"]),
                tps=_fmt(row["tokens_per_sec"]),
                sync=_fmt(row["sync_ratio"]),
                sync_count=_fmt(row["sync_active_count"]),
                sync_events=_fmt(row["sync_events"]),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def _fmt(value: dict | None) -> str:
    if not value:
        return "--"
    return f"{value['mean']:.3f} ± {value['sd']:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())

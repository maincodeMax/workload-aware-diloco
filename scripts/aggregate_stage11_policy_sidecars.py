#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
from pathlib import Path


POLICY_ORDER = [
    "fixed_h_256",
    "fixed_h_512",
    "pressure_gate",
    "pressure_gate_matched",
    "random_matched",
    "wa_diloco",
]
REGIME_ORDER = ["pressure", "sync"]


def main() -> int:
    output_dir = Path("results/stage11")
    rows = []
    for summary_path in sorted(output_dir.glob("*/*/summary.json")):
        regime = summary_path.parent.parent.name
        run_name = summary_path.parent.name
        if "-s" not in run_name:
            continue
        policy, seed_text = run_name.rsplit("-s", 1)
        try:
            seed = int(seed_text)
        except ValueError:
            continue
        summary = json.loads(summary_path.read_text())
        real = summary.get("real", {})
        rows.append(
            {
                "regime": regime,
                "policy": policy,
                "seed": seed,
                "requests": real.get("requests"),
                "high_low_p95_ratio": real.get("high_low_p95_ratio"),
                "sync_ratio": real.get("sync_inactive_p95_ratio"),
                "p95_ms": real.get("latency_p95_ms"),
                "p99_ms": real.get("latency_p99_ms"),
                "slo_violation_rate": real.get("slo_violation_rate"),
                "tokens_per_sec": real.get("tokens_per_sec"),
                "sync_events": summary.get("sync_event_count"),
                "trace_duration_sec": summary.get("sync_trace_duration_sec"),
            }
        )

    grouped = []
    for regime in REGIME_ORDER:
        for policy in POLICY_ORDER:
            values = [row for row in rows if row["regime"] == regime and row["policy"] == policy]
            if not values:
                continue
            grouped.append(_aggregate(regime, policy, values))

    (output_dir / "policy-sidecar-full.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    (output_dir / "policy-sidecar-summary.json").write_text(json.dumps(grouped, indent=2, sort_keys=True) + "\n")
    _write_markdown(output_dir / "policy-sidecar-summary.md", grouped)
    print((output_dir / "policy-sidecar-summary.md").read_text())
    return 0


def _aggregate(regime: str, policy: str, rows: list[dict]) -> dict:
    return {
        "regime": regime,
        "policy": policy,
        "n": len(rows),
        "requests": _mean_sd(rows, "requests"),
        "high_low_p95_ratio": _mean_sd(rows, "high_low_p95_ratio"),
        "sync_ratio": _mean_sd(rows, "sync_ratio"),
        "p95_ms": _mean_sd(rows, "p95_ms"),
        "p99_ms": _mean_sd(rows, "p99_ms"),
        "slo_violation_pct": _mean_sd(rows, "slo_violation_rate", scale=100.0),
        "tokens_per_sec": _mean_sd(rows, "tokens_per_sec"),
        "sync_events": _mean_sd(rows, "sync_events"),
        "trace_duration_sec": _mean_sd(rows, "trace_duration_sec"),
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
        "# Stage 11 real vLLM policy-trace replay",
        "",
        "| regime | policy | n | p95 ms | p99 ms | SLO viol. % | high/low p95 | sync p95 ratio | tok/s | sync events |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {regime} | {policy} | {n} | {p95} | {p99} | {slo} | {hl} | {sync} | {tps} | {sync_events} |".format(
                regime=row["regime"],
                policy=row["policy"],
                n=row["n"],
                p95=_fmt(row["p95_ms"]),
                p99=_fmt(row["p99_ms"]),
                slo=_fmt(row["slo_violation_pct"]),
                hl=_fmt(row["high_low_p95_ratio"]),
                sync=_fmt(row["sync_ratio"]),
                tps=_fmt(row["tokens_per_sec"]),
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

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/stage10")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    summaries = sorted(output_dir.glob("*/summary.json"))
    rows = []
    for path in summaries:
        payload = json.loads(path.read_text())
        real = payload.get("real", {})
        sim = payload.get("simulator", {})
        rows.append(
            {
                "run": path.parent.name,
                "backend": payload.get("backend"),
                "model": payload.get("model"),
                "duration": payload.get("duration_sec"),
                "concurrency": payload.get("concurrency"),
                "prompt_style": payload.get("prompt_style"),
                "max_new_tokens": payload.get("max_new_tokens"),
                "real_requests": real.get("requests"),
                "real_p95": real.get("latency_p95_ms"),
                "real_p99": real.get("latency_p99_ms"),
                "real_high_low": real.get("high_low_p95_ratio"),
                "real_corr": real.get("correlation_pressure_latency"),
                "real_sync_ratio": real.get("sync_inactive_p95_ratio"),
                "real_tokens_sec": real.get("tokens_per_sec"),
                "sim_high_low": sim.get("high_low_p95_ratio"),
                "sim_corr": sim.get("correlation_pressure_latency"),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sidecar-summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n"
    )
    _write_markdown(output_dir / "sidecar-summary.md", rows)
    return 0


def _write_markdown(path: Path, rows: list[dict]) -> None:
    lines = ["## Stage 10 Real Sidecar Calibration Matrix", ""]
    lines.append(
        "| run | model | load | real high/low p95 | real corr. | sync ratio | p95 ms | p99 ms | tok/s | sim high/low |"
    )
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        load = f"c={row['concurrency']}, {row['prompt_style']}, new={row['max_new_tokens']}"
        lines.append(
            f"| {row['run']} | {row['model']} | {load} | "
            f"{_fmt(row['real_high_low'])} | {_fmt(row['real_corr'])} | "
            f"{_fmt(row['real_sync_ratio'])} | {_fmt(row['real_p95'])} | "
            f"{_fmt(row['real_p99'])} | {_fmt(row['real_tokens_sec'])} | "
            f"{_fmt(row['sim_high_low'])} |"
        )
    path.write_text("\n".join(lines) + "\n")


def _fmt(value) -> str:
    if value is None:
        return ""
    value = float(value)
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())

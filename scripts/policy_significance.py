#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import math
import statistics
from pathlib import Path

from aggregate_stage4_results import GROUPS, _group_name
from compare_workspaces import _load_run, _summarize_prefix


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspaces", nargs="+")
    parser.add_argument("--serving-subdir", default="serving")
    parser.add_argument("--mode", choices=["tokens", "wall-clock"], default="tokens")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    runs = [
        _load_run(Path(path), serving_subdir=args.serving_subdir)
        for path in args.workspaces
        if Path(path).exists()
    ]
    runs = [run for run in runs if _group_name(run["workspace"].name) in GROUPS]
    if not runs:
        raise SystemExit("no recognized workspaces")

    if args.mode == "tokens":
        target_tokens = min(run["actual_tokens"] for run in runs)
        rows = [_summarize_prefix(run, target_tokens=target_tokens) for run in runs]
        title = f"token-matched target>={target_tokens:,} tokens"
    else:
        target_duration = min(run["duration_sec"] for run in runs if run["duration_sec"] is not None)
        rows = [_summarize_prefix(run, target_duration_sec=target_duration) for run in runs]
        title = f"wall-clock-matched target={target_duration:.1f}s"

    grouped: dict[str, list[float]] = {}
    for row in rows:
        group = _group_name(row["run"])
        value = row.get("serving_slo_violation_rate")
        if value is not None and not math.isnan(float(value)):
            grouped.setdefault(group, []).append(100.0 * float(value))

    lines = [f"## Policy Significance ({title}, serving={args.serving_subdir})", ""]
    lines.append("| contrast | n | mean_a_% | mean_b_% | diff_a_minus_b_pp | exact_p_one_sided | exact_p_two_sided |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for baseline in ["pressure_gate", "random_deferral", "fixed_h_512", "fixed_h_256"]:
        if "wa_current" not in grouped or baseline not in grouped:
            continue
        wa = grouped["wa_current"]
        other = grouped[baseline]
        result = _permutation_test(wa, other)
        lines.append(
            "| WA-DiLoCo vs {name} | {n_a}/{n_b} | {mean_a:.4f} | {mean_b:.4f} | "
            "{diff:.4f} | {p_less:.4f} | {p_two:.4f} |".format(
                name=GROUPS[baseline],
                n_a=len(wa),
                n_b=len(other),
                mean_a=statistics.mean(wa),
                mean_b=statistics.mean(other),
                diff=statistics.mean(wa) - statistics.mean(other),
                p_less=result["p_less"],
                p_two=result["p_two"],
            )
        )

    lines.extend(
        [
            "",
            "Lower SLO violation rate is better. The one-sided p-value tests whether WA-DiLoCo's mean is lower than the comparator; the two-sided p-value tests any mean difference.",
        ]
    )
    Path(args.output).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def _permutation_test(a: list[float], b: list[float]) -> dict[str, float]:
    observed = statistics.mean(a) - statistics.mean(b)
    values = a + b
    n_a = len(a)
    diffs = []
    indexes = range(len(values))
    for combo in itertools.combinations(indexes, n_a):
        left_indexes = set(combo)
        left = [values[index] for index in indexes if index in left_indexes]
        right = [values[index] for index in indexes if index not in left_indexes]
        diffs.append(statistics.mean(left) - statistics.mean(right))
    total = len(diffs)
    p_less = sum(1 for diff in diffs if diff <= observed + 1e-12) / total
    p_two = sum(1 for diff in diffs if abs(diff) >= abs(observed) - 1e-12) / total
    return {"p_less": p_less, "p_two": p_two}


if __name__ == "__main__":
    raise SystemExit(main())

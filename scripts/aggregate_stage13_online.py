#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
import statistics
from pathlib import Path


SEEDS = [23, 29, 31, 37, 41]
BASELINE_POLICIES = ["fixed_h_256", "fixed_h_512", "pressure_gate", "random_matched", "wa_diloco"]
POLICY_ORDER = [*BASELINE_POLICIES, "stage12_calibrated_replay", "online_calibrated_wa"]


def main() -> int:
    existing_summary = Path("results/stage13/online-summary.md")
    if not Path("results/stage12/bursty_sync").exists() and existing_summary.exists():
        # The public artifact excludes Stage 12 raw burst-replay summaries. In
        # that case, preserve and display the bundled aggregate rather than
        # overwriting it with the online-only subset.
        print(existing_summary.read_text())
        return 0

    rows = []
    rows.extend(_load_stage12_baselines())
    rows.extend(_load_stage13_online())

    grouped = []
    for policy in POLICY_ORDER:
        values = [row for row in rows if row["policy"] == policy]
        if values:
            grouped.append(_aggregate(policy, values))

    output_dir = Path("results/stage13")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "online-summary.json").write_text(json.dumps(grouped, indent=2, sort_keys=True) + "\n")
    (output_dir / "online-summary.md").write_text(_markdown(grouped, rows) + "\n")
    print((output_dir / "online-summary.md").read_text())
    return 0


def _load_stage12_baselines() -> list[dict]:
    base = Path("results/stage12/bursty_sync")
    no_sync = _no_sync_by_seed(base)
    rows = []
    for summary_path in sorted(base.glob("*-s*/summary.json")):
        run_name = summary_path.parent.name
        policy, seed_text = run_name.rsplit("-s", 1)
        seed = int(seed_text)
        grouped = policy
        draw = None
        if policy.startswith("random_matched_d"):
            draw = int(policy.removeprefix("random_matched_d"))
            grouped = "random_matched"
        elif policy == "calibrated_wa":
            grouped = "stage12_calibrated_replay"
        elif grouped == "no_sync":
            continue
        rows.append(_row(summary_path, policy=grouped, seed=seed, draw=draw, no_sync=no_sync))
    return rows


def _load_stage13_online() -> list[dict]:
    base = Path("results/stage13/bursty_sync")
    stage12_no_sync = _no_sync_by_seed(Path("results/stage12/bursty_sync"))
    rows = []
    for summary_path in sorted(base.glob("online_calibrated_wa-s*/summary.json")):
        seed = int(summary_path.parent.name.rsplit("-s", 1)[1])
        rows.append(_row(summary_path, policy="online_calibrated_wa", seed=seed, draw=None, no_sync=stage12_no_sync))
    return rows


def _no_sync_by_seed(base: Path) -> dict[int, float]:
    values = {}
    for summary_path in sorted(base.glob("no_sync-s*/summary.json")):
        seed = int(summary_path.parent.name.rsplit("-s", 1)[1])
        data = json.loads(summary_path.read_text())
        values[seed] = float(data["real"]["slo_violation_rate"])
    return values


def _row(summary_path: Path, *, policy: str, seed: int, draw: int | None, no_sync: dict[int, float]) -> dict:
    data = json.loads(summary_path.read_text())
    real = data["real"]
    slo = float(real["slo_violation_rate"])
    return {
        "policy": policy,
        "seed": seed,
        "draw": draw,
        "slo_rate": slo,
        "sync_excess_slo_rate": slo - no_sync.get(seed, 0.0),
        "p95_ms": real.get("latency_p95_ms"),
        "p99_ms": real.get("latency_p99_ms"),
        "tokens_per_sec": real.get("tokens_per_sec"),
        "sync_ratio": real.get("sync_inactive_p95_ratio"),
        "sync_active_count": real.get("sync_active_count"),
        "sync_events": data.get("sync_event_count"),
    }


def _aggregate(policy: str, rows: list[dict]) -> dict:
    return {
        "policy": policy,
        "n": len(rows),
        "slo_pct": _mean_sd(rows, "slo_rate", scale=100.0),
        "sync_excess_slo_pct": _mean_sd(rows, "sync_excess_slo_rate", scale=100.0),
        "p95_ms": _mean_sd(rows, "p95_ms"),
        "p99_ms": _mean_sd(rows, "p99_ms"),
        "tokens_per_sec": _mean_sd(rows, "tokens_per_sec"),
        "sync_ratio": _mean_sd(rows, "sync_ratio"),
        "sync_active_count": _mean_sd(rows, "sync_active_count"),
        "sync_events": _mean_sd(rows, "sync_events"),
    }


def _mean_sd(rows: list[dict], key: str, *, scale: float = 1.0) -> dict:
    values = [float(row[key]) * scale for row in rows if row.get(key) is not None]
    if not values:
        return {"mean": None, "sd": None}
    return {
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _markdown(grouped: list[dict], rows: list[dict]) -> str:
    lines = [
        "# Stage 13 online calibrated-WA bursty vLLM replay",
        "",
        "| policy | n | raw SLO % | sync-excess SLO % | p95 ms | p99 ms | tok/s | sync active reqs | sync events |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in grouped:
        lines.append(
            "| {policy} | {n} | {slo} | {excess} | {p95} | {p99} | {tps} | {active} | {events} |".format(
                policy=row["policy"],
                n=row["n"],
                slo=_fmt(row["slo_pct"]),
                excess=_fmt(row["sync_excess_slo_pct"]),
                p95=_fmt(row["p95_ms"]),
                p99=_fmt(row["p99_ms"]),
                tps=_fmt(row["tokens_per_sec"]),
                active=_fmt(row["sync_active_count"]),
                events=_fmt(row["sync_events"]),
            )
        )

    lines.extend(["", "## Paired sign-flip tests", ""])
    online = [row for row in rows if row["policy"] == "online_calibrated_wa"]
    if len(online) == len(SEEDS):
        online_by_seed = {row["seed"]: row for row in online}
        for baseline in ["wa_diloco", "random_matched", "stage12_calibrated_replay"]:
            baseline_by_seed = _baseline_seed_means(rows, baseline)
            if set(baseline_by_seed) >= set(SEEDS):
                diffs = [
                    100.0
                    * (
                        online_by_seed[seed]["sync_excess_slo_rate"]
                        - baseline_by_seed[seed]["sync_excess_slo_rate"]
                    )
                    for seed in SEEDS
                ]
                lines.append(
                    f"- online_calibrated_wa vs {baseline}: "
                    f"mean diff {_num(statistics.mean(diffs))} pp, "
                    f"p_less={_signflip_p_less(diffs):.3f}, "
                    f"diffs={[round(x, 3) for x in diffs]}"
                )
    return "\n".join(lines)


def _baseline_seed_means(rows: list[dict], policy: str) -> dict[int, dict]:
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        if row["policy"] == policy:
            grouped.setdefault(row["seed"], []).append(row)
    means = {}
    for seed, values in grouped.items():
        means[seed] = {
            "sync_excess_slo_rate": statistics.mean(row["sync_excess_slo_rate"] for row in values),
        }
    return means


def _signflip_p_less(diffs: list[float]) -> float:
    obs = statistics.mean(diffs)
    values = []
    for signs in itertools.product([1, -1], repeat=len(diffs)):
        values.append(statistics.mean(sign * diff for sign, diff in zip(signs, diffs)))
    return sum(value <= obs + 1e-12 for value in values) / len(values)


def _fmt(value: dict) -> str:
    if value["mean"] is None:
        return "--"
    return f"{value['mean']:.3f} ± {value['sd']:.3f}"


def _num(value: float) -> str:
    return f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())

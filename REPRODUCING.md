# Reproducing the Paper Summaries

This repository supports two levels of reproduction.

## Laptop/Audit Path

Install the package:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e .
```

Regenerate summary tables from bundled JSON files:

```bash
python3 scripts/aggregate_stage10_sidecars.py
python3 scripts/aggregate_stage11_policy_sidecars.py
python3 scripts/aggregate_stage13_online.py
python3 scripts/aggregate_stage14_fabric.py
```

Expected behavior:

- Stage 10 writes `results/stage10/sidecar-summary.{json,md}`.
- Stage 11 writes `results/stage11/policy-sidecar-{full,summary}.json` and `policy-sidecar-summary.md`.
- Stage 13 prints the bundled combined Stage 12/13 aggregate if raw Stage 12 burst replay summaries are absent.
- Stage 14 prints the bundled fabric summary if raw MI355X probe files are absent.

## Hardware Path

The full experiment path requires cluster access. The launchers are included for transparency and must be adapted to the target site.

| Stage | Purpose | Entry points |
| --- | --- | --- |
| Stage 7 | Guarded ablations and matched-deferral controls | `scripts/generate_stage7_configs.py`, `scripts/submit_b200_stage7_ablation_campaign.sh` |
| Stage 9 | SmolLM2-1.7B scale check | `scripts/generate_stage9_large_model_configs.py`, `scripts/submit_b200_stage9_large_model.sh` |
| Stage 10 | vLLM sidecar calibration | `scripts/submit_b200_stage10_sidecars.sh`, `scripts/aggregate_stage10_sidecars.py` |
| Stage 11 | Pressure-heavy and sync-heavy policy replay | `scripts/submit_b200_stage11_policy_sidecars.sh`, `scripts/aggregate_stage11_policy_sidecars.py` |
| Stage 12 | Bursty calibrated replay | `scripts/generate_stage12_schedules.py`, `scripts/submit_b200_stage12_bursty.sh`, `scripts/aggregate_stage12_bursty.py` |
| Stage 13 | Online calibrated-WA follow-up | `scripts/generate_stage13_online_configs.py`, `scripts/submit_b200_stage13_online.sh`, `scripts/aggregate_stage13_online.py` |
| Stage 14 | MI355X/RCCL fabric smoke test | `scripts/submit_mi355x_stage14_fabric.sh`, `scripts/aggregate_stage14_fabric.py` |

The Slurm scripts intentionally use environment variables and generic user paths where possible. Review all partitions, node lists, container images, and storage paths before launching on another cluster.

## Excluded Data

The release excludes large checkpoints, raw prompt text, raw serving event streams, cluster-local workspaces, and scheduler logs. Released summaries preserve the fields needed to audit the paper tables: seed, policy, regime, sync-window labels, p95/p99 latency, SLO rates, throughput, sync counts, and aggregate significance inputs.

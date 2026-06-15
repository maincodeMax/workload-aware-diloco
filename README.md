Workload-Aware DiLoCo (WA-DiLoCo) treats DiLoCo outer synchronization as a scheduling decision under shared-fleet pressure. This repository contains the controller, stress harness, calibrated vLLM replay scripts, Slurm launchers, and aggregate result summaries for the accompanying EMNLP Industry Track submission.

The repository is organized for auditability. Most commands below run on a laptop from the released summaries; full training and sidecar reproduction requires access to comparable B200/vLLM and MI355X/RCCL hardware.

## What This Repo Supports

- Controlled stress-harness policy comparisons.
- Real-sidecar effect-size calibration for vLLM serving.
- Real vLLM policy replay summaries for pressure-heavy and sync-heavy regimes.
- Bursty calibrated replay and online calibrated-WA summary checks.
- MI355X/RCCL fabric smoke-test summaries.

It does not claim a turnkey production scheduler. See [CLAIMS.md](CLAIMS.md) for the exact claim hierarchy.

## Layout

```text
src/wa_diloco/          WA-DiLoCo controller and training implementation
configs/                Fixed-H, WA-DiLoCo, ablation, large-model, and online configs
scripts/                Simulation, replay, aggregation, significance, and Slurm helpers
slurm/                  Cluster launchers used for the paper stages
results/stage10/        vLLM sidecar calibration summaries
results/stage11/        Pressure-heavy and sync-heavy policy replay summaries
results/stage13/        Combined Stage 12/13 bursty calibrated summaries
results/stage14-fabric/ MI355X/RCCL fabric-smoke summaries
```

## Quickstart

Create an environment and install the package:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e .
```

Regenerate the bundled result summaries:

```bash
python3 scripts/aggregate_stage10_sidecars.py
python3 scripts/aggregate_stage11_policy_sidecars.py
python3 scripts/aggregate_stage13_online.py
python3 scripts/aggregate_stage14_fabric.py
```

These commands should work without cluster access. Stage 13 prints the bundled combined Stage 12/13 bursty aggregate when raw Stage 12 burst summaries are absent. Stage 14 prints the bundled fabric summary unless raw MI355X probe files are supplied through `WA_DILOCO_STAGE14_RESULT_ROOT`.

## Full Reproduction

Full reproduction requires:

- Slurm.
- 8x NVIDIA B200 or comparable GPUs for the B200-style training and vLLM sidecar experiments.
- ROCm/RCCL-capable MI355X nodes for the fabric smoke test.
- Hugging Face access to the model and data used in the paper.
- vLLM for real-serving replay.

The hardware-specific path is documented in [REPRODUCING.md](REPRODUCING.md). The public artifact intentionally excludes large checkpoints, raw prompt text, raw serving event streams, cluster-local workspaces, and scheduler logs.

## Artifact Hygiene

Before making this repository public or sharing it outside the author organization, scan for local home paths, cluster-local scratch paths, login strings, API tokens, scheduler logs, raw prompt text, and raw serving event streams.
Cluster names may be retained if they are intentionally part of the experimental environment description, but tokens, usernames, private paths, prompt text, and raw customer-like logs should not be retained.

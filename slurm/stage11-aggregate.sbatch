#!/usr/bin/env bash
# Aggregate Stage 11 real vLLM policy-trace replay summaries.

#SBATCH --job-name=wa-stage11-agg
#SBATCH --partition=resident
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=results/stage11/aggregate-%j.out
#SBATCH --error=results/stage11/aggregate-%j.err

set -euo pipefail

REPO_ROOT="${WA_DILOCO_REPO:-$PWD}"
PYTHON_BIN="${WA_DILOCO_PYTHON:-${REPO_ROOT}/.venv/bin/python}"

cd "${REPO_ROOT}"
mkdir -p results/stage11
"${PYTHON_BIN}" scripts/aggregate_stage11_policy_sidecars.py

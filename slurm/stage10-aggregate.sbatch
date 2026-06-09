#!/usr/bin/env bash
# Aggregate Stage 10 sidecar calibration summaries.

#SBATCH --job-name=wa-stage10-agg
#SBATCH --partition=resident
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=results/stage10/aggregate-%j.out
#SBATCH --error=results/stage10/aggregate-%j.err

set -euo pipefail

REPO_ROOT="${WA_DILOCO_REPO:-$PWD}"
PYTHON_BIN="${WA_DILOCO_PYTHON:-${REPO_ROOT}/.venv/bin/python}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
mkdir -p results/stage10

"${PYTHON_BIN}" scripts/aggregate_stage10_sidecars.py --output-dir results/stage10
cat results/stage10/sidecar-summary.md

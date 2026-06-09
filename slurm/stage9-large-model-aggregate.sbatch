#!/usr/bin/env bash
# Aggregate stage-9 SmolLM2-1.7B smoke-test tables after the chained runs complete.

#SBATCH --job-name=wa-stage9-agg
#SBATCH --partition=resident
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=results/stage9/aggregate-%j.out
#SBATCH --error=results/stage9/aggregate-%j.err

set -euo pipefail

REPO_ROOT="${WA_DILOCO_REPO:-$PWD}"
PYTHON_BIN="${WA_DILOCO_PYTHON:-${REPO_ROOT}/.venv/bin/python}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

mkdir -p results/stage9

mapfile -t WORKSPACES < <(
  find workspaces/stage9 -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort
)

if [[ "${#WORKSPACES[@]}" == "0" ]]; then
  echo "no stage9 workspaces found" >&2
  exit 1
fi

"${PYTHON_BIN}" scripts/aggregate_stage4_results.py \
  --serving-subdir serving \
  --output-dir results/stage9 \
  "${WORKSPACES[@]}"

ls -lh results/stage9

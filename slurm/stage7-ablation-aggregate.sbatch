#!/usr/bin/env bash
# Aggregate stage-7/8 ablation campaign tables after the chained runs complete.

#SBATCH --job-name=wa-stage7-agg
#SBATCH --partition=resident
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=results/stage7/aggregate-%j.out
#SBATCH --error=results/stage7/aggregate-%j.err

set -euo pipefail

REPO_ROOT="${WA_DILOCO_REPO:-$PWD}"
PYTHON_BIN="${WA_DILOCO_PYTHON:-${REPO_ROOT}/.venv/bin/python}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

mkdir -p results/stage7

mapfile -t WORKSPACES < <(
  find workspaces/stage7 workspaces/stage8 -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort
)

if [[ "${#WORKSPACES[@]}" == "0" ]]; then
  echo "no stage7/stage8 workspaces found" >&2
  exit 1
fi

"${PYTHON_BIN}" scripts/aggregate_stage7_ablations.py \
  --serving-subdir serving \
  --output-dir results/stage7 \
  "${WORKSPACES[@]}"

"${PYTHON_BIN}" scripts/policy_significance.py \
  --serving-subdir serving \
  --mode tokens \
  --output results/stage7/policy-significance-token-matched.md \
  "${WORKSPACES[@]}" || true

ls -lh results/stage7

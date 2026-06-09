#!/usr/bin/env bash
# Real inference sidecar validation for the serving/SLO simulator.

#SBATCH --job-name=wa-sidecar-validation
#SBATCH --partition=resident
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=results/stage2/sidecar-validation-%j.out
#SBATCH --error=results/stage2/sidecar-validation-%j.err

set -euo pipefail
ulimit -c 0

REPO_ROOT="${WA_DILOCO_REPO:-$PWD}"
PYTHON_BIN="${WA_DILOCO_PYTHON:-${REPO_ROOT}/.venv/bin/python}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID

mkdir -p results/stage2
"${PYTHON_BIN}" scripts/real_sidecar_validation.py \
  --output-dir results/stage2/sidecar-validation \
  --duration-sec "${WA_DILOCO_SIDECAR_DURATION_SEC:-240}" \
  --period-sec "${WA_DILOCO_SIDECAR_PERIOD_SEC:-90}" \
  --backend "${WA_DILOCO_SIDECAR_BACKEND:-auto}"

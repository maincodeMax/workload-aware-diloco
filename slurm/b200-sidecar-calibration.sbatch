#!/usr/bin/env bash
# Real inference sidecar calibration plus calibrated replay over stage-2 traces.

#SBATCH --job-name=wa-sidecar-calib
#SBATCH --partition=resident
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=results/stage3-calibration/sidecar-calibration-%j.out
#SBATCH --error=results/stage3-calibration/sidecar-calibration-%j.err

set -euo pipefail
ulimit -c 0

REPO_ROOT="${WA_DILOCO_REPO:-$PWD}"
PYTHON_BIN="${WA_DILOCO_PYTHON:-${REPO_ROOT}/.venv/bin/python}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID

mkdir -p results/stage3-calibration

"${PYTHON_BIN}" scripts/real_sidecar_validation.py \
  --output-dir results/stage3-calibration/sidecar-validation \
  --duration-sec "${WA_DILOCO_SIDECAR_DURATION_SEC:-300}" \
  --period-sec "${WA_DILOCO_SIDECAR_PERIOD_SEC:-90}" \
  --sync-period-sec "${WA_DILOCO_SIDECAR_SYNC_PERIOD_SEC:-37}" \
  --sync-window-sec "${WA_DILOCO_SIDECAR_SYNC_WINDOW_SEC:-8}" \
  --stress-mode "${WA_DILOCO_SIDECAR_STRESS_MODE:-both}" \
  --backend "${WA_DILOCO_SIDECAR_BACKEND:-auto}"

"${PYTHON_BIN}" scripts/calibrate_serving_simulator.py \
  --real-events results/stage3-calibration/sidecar-validation/real_events.jsonl \
  --output-dir results/stage3-calibration/calibrated-replay \
  --replay-workspaces \
    workspaces/stage2/fixed-h-256-s23 \
    workspaces/stage2/fixed-h-256-s29 \
    workspaces/stage2/fixed-h-256-s31 \
    workspaces/stage2/wa-current-s23 \
    workspaces/stage2/wa-current-s29 \
    workspaces/stage2/wa-current-s31 \
  --period-sec "${WA_DILOCO_REPLAY_PERIOD_SEC:-300}" \
  --sync-window-sec "${WA_DILOCO_REPLAY_SYNC_WINDOW_SEC:-8}"

WORKSPACES=(
  workspaces/stage2/fixed-h-256-s23
  workspaces/stage2/fixed-h-256-s29
  workspaces/stage2/fixed-h-256-s31
  workspaces/stage2/wa-current-s23
  workspaces/stage2/wa-current-s29
  workspaces/stage2/wa-current-s31
)

"${PYTHON_BIN}" scripts/compare_workspaces.py --serving-subdir serving_calibrated --mode full "${WORKSPACES[@]}" \
  >results/stage3-calibration/table-full-calibrated.md
"${PYTHON_BIN}" scripts/compare_workspaces.py --serving-subdir serving_calibrated --mode tokens "${WORKSPACES[@]}" \
  >results/stage3-calibration/table-token-matched-calibrated.md
"${PYTHON_BIN}" scripts/compare_workspaces.py --serving-subdir serving_calibrated --mode wall-clock "${WORKSPACES[@]}" \
  >results/stage3-calibration/table-wall-clock-matched-calibrated.md
"${PYTHON_BIN}" scripts/aggregate_stage2_results.py --serving-subdir serving_calibrated \
  --output-dir results/stage3-calibration "${WORKSPACES[@]}"

ls -lh results/stage3-calibration results/stage3-calibration/calibrated-replay

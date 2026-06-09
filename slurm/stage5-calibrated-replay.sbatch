#!/usr/bin/env bash
# Calibrated replay and aggregate statistics for the five-policy stage-5 frontier.

#SBATCH --job-name=wa-stage5-replay
#SBATCH --partition=resident
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=results/stage5-calibrated-replay/replay-%j.out
#SBATCH --error=results/stage5-calibrated-replay/replay-%j.err

set -euo pipefail
ulimit -c 0

REPO_ROOT="${WA_DILOCO_REPO:-$PWD}"
PYTHON_BIN="${WA_DILOCO_PYTHON:-${REPO_ROOT}/.venv/bin/python}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

mkdir -p results/stage5-calibrated-replay/calibrated-replay

WORKSPACES=(
  workspaces/stage2/fixed-h-256-s23
  workspaces/stage2/fixed-h-256-s29
  workspaces/stage2/fixed-h-256-s31
  workspaces/stage4/fixed-h-512-s23
  workspaces/stage4/fixed-h-512-s29
  workspaces/stage4/fixed-h-512-s31
  workspaces/stage4/pressure-gate-s23
  workspaces/stage4/pressure-gate-s29
  workspaces/stage4/pressure-gate-s31
  workspaces/stage4/random-deferral-s23
  workspaces/stage4/random-deferral-s29
  workspaces/stage4/random-deferral-s31
  workspaces/stage2/wa-current-s23
  workspaces/stage2/wa-current-s29
  workspaces/stage2/wa-current-s31
  workspaces/stage5/fixed-h-256-s37
  workspaces/stage5/fixed-h-256-s41
  workspaces/stage5/fixed-h-512-s37
  workspaces/stage5/fixed-h-512-s41
  workspaces/stage5/pressure-gate-s37
  workspaces/stage5/pressure-gate-s41
  workspaces/stage5/random-deferral-s37
  workspaces/stage5/random-deferral-s41
  workspaces/stage5/wa-current-s37
  workspaces/stage5/wa-current-s41
)

"${PYTHON_BIN}" scripts/calibrate_serving_simulator.py \
  --real-events results/stage3-calibration/sidecar-validation/real_events.jsonl \
  --output-dir results/stage5-calibrated-replay/calibrated-replay \
  --replay-workspaces "${WORKSPACES[@]}" \
  --period-sec "${WA_DILOCO_REPLAY_PERIOD_SEC:-300}" \
  --sync-window-sec "${WA_DILOCO_REPLAY_SYNC_WINDOW_SEC:-8}" \
  >results/stage5-calibrated-replay/calibrated-replay/stdout.json

"${PYTHON_BIN}" scripts/aggregate_stage4_results.py \
  --serving-subdir serving \
  --output-dir results/stage5 \
  "${WORKSPACES[@]}"

"${PYTHON_BIN}" scripts/aggregate_stage4_results.py \
  --serving-subdir serving_calibrated \
  --output-dir results/stage5-calibrated-replay \
  "${WORKSPACES[@]}"

"${PYTHON_BIN}" scripts/policy_significance.py \
  --serving-subdir serving \
  --mode tokens \
  --output results/stage5/policy-significance-token-matched.md \
  "${WORKSPACES[@]}"

"${PYTHON_BIN}" scripts/policy_significance.py \
  --serving-subdir serving \
  --mode wall-clock \
  --output results/stage5/policy-significance-wall-clock-matched.md \
  "${WORKSPACES[@]}"

"${PYTHON_BIN}" scripts/policy_significance.py \
  --serving-subdir serving_calibrated \
  --mode tokens \
  --output results/stage5-calibrated-replay/policy-significance-token-matched.md \
  "${WORKSPACES[@]}"

"${PYTHON_BIN}" scripts/policy_significance.py \
  --serving-subdir serving_calibrated \
  --mode wall-clock \
  --output results/stage5-calibrated-replay/policy-significance-wall-clock-matched.md \
  "${WORKSPACES[@]}"

ls -lh results/stage5 results/stage5-calibrated-replay

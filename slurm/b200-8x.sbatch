#!/usr/bin/env bash
# B200 8xB200 single-node WA-DiLoCo launcher.
#
# Usage:
#   sbatch slurm/b200-8x.sbatch configs/b200_smoke.yaml
#
# Confirm B200 partition/account names before first submission.

#SBATCH --job-name=wa-diloco-b200
#SBATCH --partition=resident
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:8
#SBATCH --cpus-per-task=96
#SBATCH --time=08:00:00
#SBATCH --exclusive
#SBATCH --output=results/%x-%j.out
#SBATCH --error=results/%x-%j.err

set -euo pipefail
ulimit -c 0

REPO_ROOT="${WA_DILOCO_REPO:-$PWD}"
CONFIG="${1:-configs/b200_smoke.yaml}"
DEFAULT_PYTHON="${REPO_ROOT}/.venv/bin/python"
PYTHON_BIN="${WA_DILOCO_PYTHON:-${DEFAULT_PYTHON}}"
LEARNERS="${WA_DILOCO_LEARNERS:-8}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID

mkdir -p results

RUN_ID="${SLURM_JOB_ID:-manual-$(date -u +%Y%m%dT%H%M%S)}"
CONFIG_STEM="$(basename "${CONFIG}" .yaml)"
RUN_RESULTS_DIR="results/${RUN_ID}"
mkdir -p "${RUN_RESULTS_DIR}"

WORKSPACE="$("${PYTHON_BIN}" - "${CONFIG}" <<'PY'
from wa_diloco.config import load_config
import sys
print(load_config(sys.argv[1]).runtime.workspace)
PY
)"
POLICY="$("${PYTHON_BIN}" - "${CONFIG}" <<'PY'
from wa_diloco.config import load_config
import sys
print(load_config(sys.argv[1]).policy.name)
PY
)"
PRESSURE_FILE="$("${PYTHON_BIN}" - "${CONFIG}" <<'PY'
from wa_diloco.config import load_config
import sys
pressure_file = load_config(sys.argv[1]).runtime.pressure_file
print(pressure_file or "")
PY
)"

echo "config=${CONFIG}"
echo "repo=${REPO_ROOT}"
echo "python=$(${PYTHON_BIN} --version)"
echo "learners=${LEARNERS}"
echo "workspace=${WORKSPACE}"
echo "policy=${POLICY}"
echo "pressure_file=${PRESSURE_FILE}"
echo "run_results=${RUN_RESULTS_DIR}"
date -u --iso-8601=seconds

if [[ "${WA_DILOCO_FOREIGN_GPU_GUARD:-1}" != "0" ]]; then
  GUARD_POLL_SEC="${WA_DILOCO_FOREIGN_GPU_GUARD_POLL_SEC:-60}"
  GUARD_MAX_WAIT_SEC="${WA_DILOCO_FOREIGN_GPU_GUARD_MAX_WAIT_SEC:-43200}"
  GUARD_START_TS="$(date +%s)"
  while true; do
    FOREIGN_GPU_PIDS="$(
      nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
        | while read -r pid; do
            [[ -z "${pid}" ]] && continue
            owner="$(ps -o user= -p "${pid}" 2>/dev/null | awk '{print $1}')"
            [[ -z "${owner}" ]] && continue
            if [[ "${owner}" != "${USER}" ]]; then
              command="$(ps -o args= -p "${pid}" 2>/dev/null | cut -c1-160)"
              printf '%s %s %s\n' "${pid}" "${owner}" "${command}"
            fi
          done
    )"
    if [[ -z "${FOREIGN_GPU_PIDS}" ]]; then
      echo "foreign_gpu_guard=clear"
      break
    fi

    NOW_TS="$(date +%s)"
    WAITED=$((NOW_TS - GUARD_START_TS))
    if (( WAITED >= GUARD_MAX_WAIT_SEC )); then
      echo "foreign GPU processes still active after ${WAITED}s:" >&2
      echo "${FOREIGN_GPU_PIDS}" >&2
      exit 2
    fi

    echo "foreign_gpu_guard=waiting waited=${WAITED}s poll=${GUARD_POLL_SEC}s"
    echo "${FOREIGN_GPU_PIDS}"
    sleep "${GUARD_POLL_SEC}"
  done
fi

"${PYTHON_BIN}" -m wa_diloco.init_checkpoint "${CONFIG}"

PRESSURE_PID=""
if [[ -n "${PRESSURE_FILE}" && "${WA_DILOCO_PRESSURE_MODE:-mixed}" != "off" ]]; then
  "${PYTHON_BIN}" scripts/serving_slo_sidecar.py \
    "${WORKSPACE}" \
    --mode "${WA_DILOCO_PRESSURE_MODE:-mixed}" \
    --period-sec "${WA_DILOCO_PRESSURE_PERIOD_SEC:-300}" \
    --interval-sec "${WA_DILOCO_PRESSURE_INTERVAL_SEC:-5}" \
    --requests-per-interval "${WA_DILOCO_SERVING_REQUESTS_PER_INTERVAL:-32}" \
    --slo-ms "${WA_DILOCO_SERVING_SLO_MS:-120}" \
    --sync-window-sec "${WA_DILOCO_SERVING_SYNC_WINDOW_SEC:-8}" \
    --sync-penalty-ms "${WA_DILOCO_SERVING_SYNC_PENALTY_MS:-35}" \
    --burst-window-sec "${WA_DILOCO_PRESSURE_BURST_WINDOW_SEC:-30}" \
    --burst-phase-sec "${WA_DILOCO_PRESSURE_BURST_PHASE_SEC:-0}" \
    >"${RUN_RESULTS_DIR}/pressure.out" \
    2>"${RUN_RESULTS_DIR}/pressure.err" &
  PRESSURE_PID=$!
  echo "serving_slo_pid=${PRESSURE_PID}"
fi

cleanup() {
  if [[ -n "${PRESSURE_PID}" ]]; then
    kill "${PRESSURE_PID}" 2>/dev/null || true
    wait "${PRESSURE_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

"${PYTHON_BIN}" -m wa_diloco.coordinator "${CONFIG}" &
COORD_PID=$!

LEARNER_PIDS=()
for learner_id in $(seq 0 $((LEARNERS - 1))); do
  (
    export CUDA_VISIBLE_DEVICES="${learner_id}"
    "${PYTHON_BIN}" -m wa_diloco.train_lm "${CONFIG}" --learner-id "${learner_id}"
  ) >"${RUN_RESULTS_DIR}/learner-${learner_id}.out" 2>"${RUN_RESULTS_DIR}/learner-${learner_id}.err" &
  LEARNER_PIDS+=("$!")
done

STATUS=0
wait "${COORD_PID}" || STATUS=$?
for pid in "${LEARNER_PIDS[@]}"; do
  wait "${pid}" || STATUS=$?
done
cleanup

if [[ "${STATUS}" != "0" ]]; then
  exit "${STATUS}"
fi

"${PYTHON_BIN}" -m wa_diloco.summarize_results "${WORKSPACE}" \
  | tee "${RUN_RESULTS_DIR}/summary-${CONFIG_STEM}-${RUN_ID}.json" \
        "results/summary-${CONFIG_STEM}-${RUN_ID}.json"

if [[ -n "${WA_DILOCO_EVAL_DATASET:-}" ]]; then
  EVAL_DIR="${WORKSPACE}/eval"
  mkdir -p "${EVAL_DIR}"
  EVAL_ARGS=(
    "${CONFIG}"
    --output "${EVAL_DIR}/final.json"
    --dataset "${WA_DILOCO_EVAL_DATASET}"
    --split "${WA_DILOCO_EVAL_SPLIT:-validation}"
    --batch-size "${WA_DILOCO_EVAL_BATCH_SIZE:-4}"
    --max-batches "${WA_DILOCO_EVAL_MAX_BATCHES:-64}"
  )
  if [[ -n "${WA_DILOCO_EVAL_CONFIG:-}" ]]; then
    EVAL_ARGS+=(--dataset-config "${WA_DILOCO_EVAL_CONFIG}")
  fi
  if [[ -n "${WA_DILOCO_EVAL_TEXT_COLUMN:-}" ]]; then
    EVAL_ARGS+=(--text-column "${WA_DILOCO_EVAL_TEXT_COLUMN}")
  fi
  "${PYTHON_BIN}" scripts/eval_checkpoint.py "${EVAL_ARGS[@]}" \
    | tee "${RUN_RESULTS_DIR}/eval-${CONFIG_STEM}-${RUN_ID}.json" \
          "results/eval-${CONFIG_STEM}-${RUN_ID}.json"
fi

if [[ "${WA_DILOCO_KEEP_STATE:-0}" != "1" ]]; then
  rm -rf "${WORKSPACE}/checkpoints" "${WORKSPACE}/learners" "${WORKSPACE}/updates" "${WORKSPACE}/tokenizer"
fi

date -u --iso-8601=seconds

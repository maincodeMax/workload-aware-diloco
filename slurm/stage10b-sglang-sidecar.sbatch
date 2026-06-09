#!/usr/bin/env bash
# Stage 10b SGLang sidecar validation for stack robustness.

#SBATCH --job-name=wa-stage10b-sglang
#SBATCH --partition=resident
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --output=results/stage10b/sglang-%j.out
#SBATCH --error=results/stage10b/sglang-%j.err

set -euo pipefail
ulimit -c 0

REPO_ROOT="${WA_DILOCO_REPO:-$PWD}"
SGLANG_VENV="${WA_DILOCO_SGLANG_VENV:-${REPO_ROOT}/.venv-sglang}"
SGLANG_PACKAGE="${WA_DILOCO_SGLANG_PACKAGE:-sglang[all]}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID

mkdir -p results/stage10b

RUN_NAME="${WA_DILOCO_SIDECAR_RUN_NAME:-stage10b-${SLURM_JOB_ID:-manual}}"
SIDECAR_OUTPUT_DIR="${WA_DILOCO_SIDECAR_OUTPUT_DIR:-results/stage10b/${RUN_NAME}}"
mkdir -p "${SIDECAR_OUTPUT_DIR}"

echo "run_name=${RUN_NAME}"
echo "output_dir=${SIDECAR_OUTPUT_DIR}"
echo "model=${WA_DILOCO_SIDECAR_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
echo "duration=${WA_DILOCO_SIDECAR_DURATION_SEC:-1200}"
echo "concurrency=${WA_DILOCO_SIDECAR_CONCURRENCY:-4}"
echo "prompt_style=${WA_DILOCO_SIDECAR_PROMPT_STYLE:-mixed}"
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

if [[ ! -x "${SGLANG_VENV}/bin/python" ]]; then
  python3 -m venv "${SGLANG_VENV}"
  "${SGLANG_VENV}/bin/python" -m pip install -U pip
fi

if ! "${SGLANG_VENV}/bin/python" - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("sglang") else 1)
PY
then
  "${SGLANG_VENV}/bin/python" -m pip install "${SGLANG_PACKAGE}"
fi

"${SGLANG_VENV}/bin/python" scripts/real_sidecar_validation.py \
  --output-dir "${SIDECAR_OUTPUT_DIR}" \
  --model "${WA_DILOCO_SIDECAR_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}" \
  --duration-sec "${WA_DILOCO_SIDECAR_DURATION_SEC:-1200}" \
  --period-sec "${WA_DILOCO_SIDECAR_PERIOD_SEC:-90}" \
  --sync-period-sec "${WA_DILOCO_SIDECAR_SYNC_PERIOD_SEC:-37}" \
  --sync-window-sec "${WA_DILOCO_SIDECAR_SYNC_WINDOW_SEC:-8}" \
  --max-new-tokens "${WA_DILOCO_SIDECAR_MAX_NEW_TOKENS:-96}" \
  --stress-mode "${WA_DILOCO_SIDECAR_STRESS_MODE:-both}" \
  --backend sglang \
  --concurrency "${WA_DILOCO_SIDECAR_CONCURRENCY:-4}" \
  --prompt-style "${WA_DILOCO_SIDECAR_PROMPT_STYLE:-mixed}" \
  --gpu-memory-utilization "${WA_DILOCO_SIDECAR_GPU_MEMORY_UTILIZATION:-0.75}" \
  --max-model-len "${WA_DILOCO_SIDECAR_MAX_MODEL_LEN:-4096}" \
  --stress-matmul-size "${WA_DILOCO_SIDECAR_STRESS_MATMUL_SIZE:-4096}" \
  --stress-steps "${WA_DILOCO_SIDECAR_STRESS_STEPS:-12}"

cat "${SIDECAR_OUTPUT_DIR}/summary.json"
date -u --iso-8601=seconds

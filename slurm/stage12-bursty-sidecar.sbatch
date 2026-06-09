#!/usr/bin/env bash
# Stage 12 bursty sync-heavy vLLM replay.

#SBATCH --job-name=wa-stage12-bursty
#SBATCH --partition=resident
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=03:00:00
#SBATCH --output=results/stage12/sidecar-%j.out
#SBATCH --error=results/stage12/sidecar-%j.err

set -euo pipefail
ulimit -c 0

REPO_ROOT="${WA_DILOCO_REPO:-$PWD}"
VLLM_VENV="${WA_DILOCO_VLLM_VENV:-${REPO_ROOT}/.venv-vllm-0102}"
VLLM_VERSION="${WA_DILOCO_VLLM_VERSION:-0.10.2}"
TRANSFORMERS_VERSION="${WA_DILOCO_TRANSFORMERS_VERSION:-4.55.4}"
TOKENIZERS_VERSION="${WA_DILOCO_TOKENIZERS_VERSION:-0.21.4}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID

mkdir -p results/stage12

RUN_NAME="${WA_DILOCO_SIDECAR_RUN_NAME:-stage12-${SLURM_JOB_ID:-manual}}"
SIDECAR_OUTPUT_DIR="${WA_DILOCO_SIDECAR_OUTPUT_DIR:-results/stage12/${RUN_NAME}}"
SYNC_EVENTS_JSONL="${WA_DILOCO_SYNC_EVENTS_JSONL:-}"
mkdir -p "${SIDECAR_OUTPUT_DIR}"

echo "run_name=${RUN_NAME}"
echo "output_dir=${SIDECAR_OUTPUT_DIR}"
echo "sync_events=${SYNC_EVENTS_JSONL}"
echo "model=${WA_DILOCO_SIDECAR_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
echo "stress_mode=${WA_DILOCO_SIDECAR_STRESS_MODE:-sync}"
echo "load_profile=${WA_DILOCO_SIDECAR_LOAD_PROFILE:-bursty}"
echo "slo_ms=${WA_DILOCO_SIDECAR_SLO_MS:-1000}"
date -u --iso-8601=seconds

if [[ "${WA_DILOCO_FOREIGN_GPU_GUARD:-1}" != "0" ]]; then
  ALLOCATED_GPU_UUIDS="$(
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      IFS=',' read -ra visible_devices <<<"${CUDA_VISIBLE_DEVICES}"
      for device_id in "${visible_devices[@]}"; do
        if [[ "${device_id}" == GPU-* ]]; then
          printf '%s\n' "${device_id}"
        else
          nvidia-smi -i "${device_id}" --query-gpu=uuid --format=csv,noheader,nounits 2>/dev/null || true
        fi
      done
    fi
  )"
  GUARD_POLL_SEC="${WA_DILOCO_FOREIGN_GPU_GUARD_POLL_SEC:-60}"
  GUARD_MAX_WAIT_SEC="${WA_DILOCO_FOREIGN_GPU_GUARD_MAX_WAIT_SEC:-43200}"
  GUARD_START_TS="$(date +%s)"
  while true; do
    FOREIGN_GPU_PIDS="$(
      nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null \
        | while IFS=',' read -r gpu_uuid pid; do
            gpu_uuid="$(echo "${gpu_uuid}" | awk '{print $1}')"
            pid="$(echo "${pid}" | awk '{print $1}')"
            if [[ -n "${ALLOCATED_GPU_UUIDS}" ]] && ! grep -qx "${gpu_uuid}" <<<"${ALLOCATED_GPU_UUIDS}"; then
              continue
            fi
            [[ -z "${pid}" ]] && continue
            owner="$(ps -o user= -p "${pid}" 2>/dev/null | awk '{print $1}')"
            [[ -z "${owner}" ]] && continue
            if [[ "${owner}" != "${USER}" ]]; then
              command="$(ps -o args= -p "${pid}" 2>/dev/null | cut -c1-160)"
              printf '%s %s %s %s\n' "${gpu_uuid}" "${pid}" "${owner}" "${command}"
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

if [[ ! -x "${VLLM_VENV}/bin/python" ]]; then
  python3 -m venv "${VLLM_VENV}"
  "${VLLM_VENV}/bin/python" -m pip install -U pip
fi

if ! "${VLLM_VENV}/bin/python" - "${VLLM_VERSION}" "${TRANSFORMERS_VERSION}" "${TOKENIZERS_VERSION}" <<'PY'
import importlib.metadata as metadata
import sys

expected = {
    "vllm": sys.argv[1],
    "transformers": sys.argv[2],
    "tokenizers": sys.argv[3],
}
for package, version in expected.items():
    try:
        installed = metadata.version(package)
    except metadata.PackageNotFoundError:
        raise SystemExit(1)
    if installed != version:
        raise SystemExit(1)
raise SystemExit(0)
PY
then
  "${VLLM_VENV}/bin/python" -m pip install \
    "vllm==${VLLM_VERSION}" \
    "transformers==${TRANSFORMERS_VERSION}" \
    "tokenizers==${TOKENIZERS_VERSION}"
fi

sync_args=()
if [[ "${WA_DILOCO_SIDECAR_SYNC_WINDOWS:-1}" == "0" ]]; then
  sync_args+=(--no-sync-windows)
elif [[ -n "${SYNC_EVENTS_JSONL}" ]]; then
  sync_args+=(--sync-events-jsonl "${SYNC_EVENTS_JSONL}" --sync-events-repeat)
fi

"${VLLM_VENV}/bin/python" scripts/real_sidecar_validation.py \
  --output-dir "${SIDECAR_OUTPUT_DIR}" \
  --model "${WA_DILOCO_SIDECAR_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}" \
  --duration-sec "${WA_DILOCO_SIDECAR_DURATION_SEC:-600}" \
  --period-sec "${WA_DILOCO_SIDECAR_PERIOD_SEC:-120}" \
  --sync-period-sec "${WA_DILOCO_SIDECAR_SYNC_PERIOD_SEC:-37}" \
  --sync-window-sec "${WA_DILOCO_SIDECAR_SYNC_WINDOW_SEC:-8}" \
  "${sync_args[@]}" \
  --max-new-tokens "${WA_DILOCO_SIDECAR_MAX_NEW_TOKENS:-128}" \
  --slo-ms "${WA_DILOCO_SIDECAR_SLO_MS:-1000}" \
  --stress-mode "${WA_DILOCO_SIDECAR_STRESS_MODE:-sync}" \
  --backend vllm \
  --concurrency "${WA_DILOCO_SIDECAR_CONCURRENCY:-8}" \
  --load-profile "${WA_DILOCO_SIDECAR_LOAD_PROFILE:-bursty}" \
  --quiet-concurrency "${WA_DILOCO_SIDECAR_QUIET_CONCURRENCY:-4}" \
  --busy-concurrency "${WA_DILOCO_SIDECAR_BUSY_CONCURRENCY:-12}" \
  --burst-period-sec "${WA_DILOCO_SIDECAR_BURST_PERIOD_SEC:-120}" \
  --burst-window-sec "${WA_DILOCO_SIDECAR_BURST_WINDOW_SEC:-30}" \
  --burst-phase-sec "${WA_DILOCO_SIDECAR_BURST_PHASE_SEC:-0}" \
  --prompt-style "${WA_DILOCO_SIDECAR_PROMPT_STYLE:-mixed}" \
  --gpu-memory-utilization "${WA_DILOCO_SIDECAR_GPU_MEMORY_UTILIZATION:-0.80}" \
  --max-model-len "${WA_DILOCO_SIDECAR_MAX_MODEL_LEN:-4096}" \
  --stress-matmul-size "${WA_DILOCO_SIDECAR_STRESS_MATMUL_SIZE:-6144}" \
  --stress-steps "${WA_DILOCO_SIDECAR_STRESS_STEPS:-20}"

cat "${SIDECAR_OUTPUT_DIR}/summary.json"
date -u --iso-8601=seconds

#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "usage: $0 <sbatch-file> [sbatch-args...]" >&2
  exit 2
fi

POLL_SEC="${WA_DILOCO_GPU_POLL_SEC:-60}"
MAX_WAIT_SEC="${WA_DILOCO_GPU_MAX_WAIT_SEC:-21600}"
MAX_MEMORY_USED_MIB="${WA_DILOCO_GPU_MAX_MEMORY_USED_MIB:-5000}"
START_TS="$(date +%s)"

while true; do
  if nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
      | awk -v limit="${MAX_MEMORY_USED_MIB}" '$1 + 0 > limit { found = 1 } END { exit found ? 0 : 1 }'; then
    BUSY_REASON="GPU memory above ${MAX_MEMORY_USED_MIB} MiB"
  elif nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -Eq '[0-9]'; then
    BUSY_REASON="active GPU compute process"
  else
    break
  fi

  NOW_TS="$(date +%s)"
  WAITED=$((NOW_TS - START_TS))
  if (( WAITED >= MAX_WAIT_SEC )); then
    echo "timed out after ${WAITED}s waiting for GPUs to become free" >&2
    exit 1
  fi

  echo "$(date -Iseconds) GPUs still busy (${BUSY_REASON}); waiting ${POLL_SEC}s" >&2
  sleep "${POLL_SEC}"
done

echo "$(date -Iseconds) GPUs free; submitting: sbatch $*" >&2
sbatch "$@"

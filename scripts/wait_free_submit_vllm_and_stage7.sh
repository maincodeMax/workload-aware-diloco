#!/usr/bin/env bash
set -euo pipefail

POLL_SEC="${WA_DILOCO_GPU_POLL_SEC:-60}"
MAX_WAIT_SEC="${WA_DILOCO_GPU_MAX_WAIT_SEC:-43200}"
MAX_MEMORY_USED_MIB="${WA_DILOCO_GPU_MAX_MEMORY_USED_MIB:-5000}"
REPO_ROOT="${WA_DILOCO_REPO:-$PWD}"

cd "${REPO_ROOT}"
mkdir -p results/stage6-real-sidecar results/stage7

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

echo "$(date -Iseconds) GPUs free; submitting real vLLM validation" >&2
vllm_job_id="$(sbatch --parsable slurm/b200-real-sidecar-validation.sbatch)"
echo "vllm_job_id=${vllm_job_id}" | tee results/stage6-real-sidecar/stage6-stage7-chain.tsv

echo "$(date -Iseconds) queueing stage7/stage8 ablation campaign after vLLM job ${vllm_job_id}" >&2
WA_DILOCO_DEPENDENCY_JOB="${vllm_job_id}" \
WA_DILOCO_DEPENDENCY_TYPE="${WA_DILOCO_STAGE7_DEPENDENCY_TYPE:-afterany}" \
  ./scripts/submit_b200_stage7_ablation_campaign.sh \
  | tee -a results/stage6-real-sidecar/stage6-stage7-chain.tsv

#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  configs/stage4/fixed_h_512_s23.yaml
  configs/stage4/fixed_h_512_s29.yaml
  configs/stage4/fixed_h_512_s31.yaml
  configs/stage4/pressure_gate_s23.yaml
  configs/stage4/pressure_gate_s29.yaml
  configs/stage4/pressure_gate_s31.yaml
  configs/stage4/random_deferral_s23.yaml
  configs/stage4/random_deferral_s29.yaml
  configs/stage4/random_deferral_s31.yaml
)

REPO_ROOT="${WA_DILOCO_REPO:-$PWD}"
PYTHON_BIN="${WA_DILOCO_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
RESET="${WA_DILOCO_STAGE4_RESET:-1}"
SUBMIT_VLLM="${WA_DILOCO_STAGE4_VLLM:-1}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
mkdir -p results/stage4 results/stage4-vllm-sidecar

previous_job=""
manifest="results/stage4/submitted-$(date -u +%Y%m%dT%H%M%S).tsv"
printf "job_id\tkind\tconfig\tworkspace\n" >"${manifest}"

eval_export="ALL,WA_DILOCO_EVAL_DATASET=${WA_DILOCO_EVAL_DATASET:-Salesforce/wikitext},WA_DILOCO_EVAL_CONFIG=${WA_DILOCO_EVAL_CONFIG:-wikitext-2-raw-v1},WA_DILOCO_EVAL_SPLIT=${WA_DILOCO_EVAL_SPLIT:-validation},WA_DILOCO_EVAL_BATCH_SIZE=${WA_DILOCO_EVAL_BATCH_SIZE:-4},WA_DILOCO_EVAL_MAX_BATCHES=${WA_DILOCO_EVAL_MAX_BATCHES:-64}"

for config in "${CONFIGS[@]}"; do
  workspace="$("${PYTHON_BIN}" - "${config}" <<'PY'
from wa_diloco.config import load_config
import sys
print(load_config(sys.argv[1]).runtime.workspace)
PY
)"
  if [[ "${RESET}" == "1" ]]; then
    rm -rf "${workspace}"
  fi

  if [[ -n "${previous_job}" ]]; then
    job_id="$(sbatch --parsable --export="${eval_export}" --dependency=afterok:${previous_job} slurm/b200-8x.sbatch "${config}")"
  else
    job_id="$(sbatch --parsable --export="${eval_export}" slurm/b200-8x.sbatch "${config}")"
  fi

  printf "%s\ttraining\t%s\t%s\n" "${job_id}" "${config}" "${workspace}" | tee -a "${manifest}"
  previous_job="${job_id}"
done

if [[ "${SUBMIT_VLLM}" == "1" ]]; then
  vllm_export="ALL,WA_DILOCO_VLLM_VENV=${WA_DILOCO_VLLM_VENV:-${REPO_ROOT}/.venv-vllm-0102},WA_DILOCO_VLLM_VERSION=${WA_DILOCO_VLLM_VERSION:-0.10.2},WA_DILOCO_SIDECAR_MODEL=${WA_DILOCO_SIDECAR_MODEL:-Qwen/Qwen2.5-0.5B-Instruct},WA_DILOCO_SIDECAR_OUTPUT_DIR=${WA_DILOCO_SIDECAR_OUTPUT_DIR:-results/stage4-vllm-sidecar/sidecar-validation},WA_DILOCO_SIDECAR_DURATION_SEC=${WA_DILOCO_SIDECAR_DURATION_SEC:-300},WA_DILOCO_SIDECAR_PERIOD_SEC=${WA_DILOCO_SIDECAR_PERIOD_SEC:-90},WA_DILOCO_SIDECAR_SYNC_PERIOD_SEC=${WA_DILOCO_SIDECAR_SYNC_PERIOD_SEC:-37},WA_DILOCO_SIDECAR_SYNC_WINDOW_SEC=${WA_DILOCO_SIDECAR_SYNC_WINDOW_SEC:-8},WA_DILOCO_SIDECAR_STRESS_MODE=${WA_DILOCO_SIDECAR_STRESS_MODE:-both}"
  if [[ -n "${previous_job}" ]]; then
    vllm_job_id="$(sbatch --parsable --export="${vllm_export}" --dependency=afterok:${previous_job} slurm/b200-vllm-sidecar.sbatch)"
  else
    vllm_job_id="$(sbatch --parsable --export="${vllm_export}" slurm/b200-vllm-sidecar.sbatch)"
  fi
  printf "%s\tvllm_sidecar\t%s\t%s\n" "${vllm_job_id}" "slurm/b200-vllm-sidecar.sbatch" "results/stage4-vllm-sidecar/sidecar-validation" | tee -a "${manifest}"
fi

echo "manifest=${manifest}"

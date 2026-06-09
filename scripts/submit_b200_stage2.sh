#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  configs/stage2/fixed_h_256_s23.yaml
  configs/stage2/fixed_h_256_s29.yaml
  configs/stage2/fixed_h_256_s31.yaml
  configs/stage2/wa_current_s23.yaml
  configs/stage2/wa_current_s29.yaml
  configs/stage2/wa_current_s31.yaml
  configs/stage2/wa_no_slo_s23.yaml
  configs/stage2/wa_no_network_s23.yaml
  configs/stage2/wa_no_staleness_s23.yaml
)

REPO_ROOT="${WA_DILOCO_REPO:-$PWD}"
PYTHON_BIN="${WA_DILOCO_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
RESET="${WA_DILOCO_STAGE2_RESET:-1}"
SUBMIT_SIDECAR="${WA_DILOCO_STAGE2_SIDECAR:-1}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
mkdir -p results/stage2

previous_job=""
manifest="results/stage2/submitted-$(date -u +%Y%m%dT%H%M%S).tsv"
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

if [[ "${SUBMIT_SIDECAR}" == "1" ]]; then
  if [[ -n "${previous_job}" ]]; then
    sidecar_job_id="$(sbatch --parsable --dependency=afterok:${previous_job} slurm/b200-sidecar-validation.sbatch)"
  else
    sidecar_job_id="$(sbatch --parsable slurm/b200-sidecar-validation.sbatch)"
  fi
  printf "%s\tsidecar_validation\t%s\t%s\n" "${sidecar_job_id}" "slurm/b200-sidecar-validation.sbatch" "results/stage2/sidecar-validation" | tee -a "${manifest}"
fi

echo "manifest=${manifest}"

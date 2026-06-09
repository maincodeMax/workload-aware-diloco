#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  configs/stage5/fixed_h_256_s37.yaml
  configs/stage5/fixed_h_256_s41.yaml
  configs/stage5/fixed_h_512_s37.yaml
  configs/stage5/fixed_h_512_s41.yaml
  configs/stage5/pressure_gate_s37.yaml
  configs/stage5/pressure_gate_s41.yaml
  configs/stage5/random_deferral_s37.yaml
  configs/stage5/random_deferral_s41.yaml
  configs/stage5/wa_current_s37.yaml
  configs/stage5/wa_current_s41.yaml
)

REPO_ROOT="${WA_DILOCO_REPO:-$PWD}"
PYTHON_BIN="${WA_DILOCO_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
RESET="${WA_DILOCO_STAGE5_RESET:-1}"
SUBMIT_REPLAY="${WA_DILOCO_STAGE5_REPLAY:-1}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
mkdir -p results/stage5

previous_job=""
manifest="results/stage5/submitted-$(date -u +%Y%m%dT%H%M%S).tsv"
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

if [[ "${SUBMIT_REPLAY}" == "1" ]]; then
  replay_job_id="$(sbatch --parsable --dependency=afterok:${previous_job} slurm/stage5-calibrated-replay.sbatch)"
  printf "%s\tcalibrated_replay\t%s\t%s\n" "${replay_job_id}" "slurm/stage5-calibrated-replay.sbatch" "results/stage5-calibrated-replay" | tee -a "${manifest}"
fi

echo "manifest=${manifest}"

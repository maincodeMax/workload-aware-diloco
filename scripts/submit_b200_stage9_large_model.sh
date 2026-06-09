#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${WA_DILOCO_REPO:-$PWD}"
PYTHON_BIN="${WA_DILOCO_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
RESET="${WA_DILOCO_STAGE9_RESET:-1}"
SUBMIT_AGGREGATE="${WA_DILOCO_STAGE9_AGGREGATE:-1}"
DEPENDENCY_JOB="${WA_DILOCO_DEPENDENCY_JOB:-}"
DEPENDENCY_TYPE="${WA_DILOCO_DEPENDENCY_TYPE:-afterany}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

"${PYTHON_BIN}" scripts/generate_stage9_large_model_configs.py

mapfile -t CONFIGS < configs/stage9/configs.txt
mkdir -p results/stage9

manifest="results/stage9/submitted-$(date -u +%Y%m%dT%H%M%S).tsv"
printf "job_id\tkind\tconfig\tworkspace\n" >"${manifest}"

eval_export="ALL,WA_DILOCO_EVAL_DATASET=${WA_DILOCO_EVAL_DATASET:-Salesforce/wikitext},WA_DILOCO_EVAL_CONFIG=${WA_DILOCO_EVAL_CONFIG:-wikitext-2-raw-v1},WA_DILOCO_EVAL_SPLIT=${WA_DILOCO_EVAL_SPLIT:-validation},WA_DILOCO_EVAL_BATCH_SIZE=${WA_DILOCO_EVAL_BATCH_SIZE:-2},WA_DILOCO_EVAL_MAX_BATCHES=${WA_DILOCO_EVAL_MAX_BATCHES:-32}"

previous_job="${DEPENDENCY_JOB}"
for config in "${CONFIGS[@]}"; do
  [[ -z "${config}" ]] && continue
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
    job_id="$(sbatch --parsable --export="${eval_export}" --dependency="${DEPENDENCY_TYPE}:${previous_job}" slurm/b200-8x.sbatch "${config}")"
  else
    job_id="$(sbatch --parsable --export="${eval_export}" slurm/b200-8x.sbatch "${config}")"
  fi

  printf "%s\ttraining\t%s\t%s\n" "${job_id}" "${config}" "${workspace}" | tee -a "${manifest}"
  previous_job="${job_id}"
done

if [[ "${SUBMIT_AGGREGATE}" == "1" && -n "${previous_job}" ]]; then
  aggregate_job_id="$(sbatch --parsable --dependency="afterany:${previous_job}" slurm/stage9-large-model-aggregate.sbatch)"
  printf "%s\taggregate\t%s\t%s\n" "${aggregate_job_id}" "slurm/stage9-large-model-aggregate.sbatch" "results/stage9" | tee -a "${manifest}"
fi

echo "manifest=${manifest}"

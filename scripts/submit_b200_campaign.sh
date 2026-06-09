#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  configs/campaign/fixed_h_256_s23_token_match.yaml
  configs/campaign/fixed_h_256_s29_token_match.yaml
  configs/campaign/fixed_h_256_s31_token_match.yaml
  configs/campaign/wa_balanced_s23.yaml
  configs/campaign/wa_balanced_s29.yaml
  configs/campaign/wa_balanced_s31.yaml
  configs/campaign/wa_current_s23.yaml
  configs/campaign/wa_fast_sync_s23.yaml
)

REPO_ROOT="${WA_DILOCO_REPO:-$PWD}"
PYTHON_BIN="${WA_DILOCO_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
RESET="${WA_DILOCO_CAMPAIGN_RESET:-1}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
mkdir -p results/campaign

previous_job=""
manifest="results/campaign/submitted-$(date -u +%Y%m%dT%H%M%S).tsv"
printf "job_id\tconfig\tworkspace\n" >"${manifest}"

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
    job_id="$(sbatch --parsable --dependency=afterok:${previous_job} slurm/b200-8x.sbatch "${config}")"
  else
    job_id="$(sbatch --parsable slurm/b200-8x.sbatch "${config}")"
  fi

  printf "%s\t%s\t%s\n" "${job_id}" "${config}" "${workspace}" | tee -a "${manifest}"
  previous_job="${job_id}"
done

echo "manifest=${manifest}"

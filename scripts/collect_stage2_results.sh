#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${WA_DILOCO_REPO:-$PWD}"
PYTHON_BIN="${WA_DILOCO_PYTHON:-${REPO_ROOT}/.venv/bin/python}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
mkdir -p results/stage2

WORKSPACES=(
  workspaces/stage2/fixed-h-256-s23
  workspaces/stage2/fixed-h-256-s29
  workspaces/stage2/fixed-h-256-s31
  workspaces/stage2/wa-current-s23
  workspaces/stage2/wa-current-s29
  workspaces/stage2/wa-current-s31
  workspaces/stage2/wa-no-slo-s23
  workspaces/stage2/wa-no-network-s23
  workspaces/stage2/wa-no-staleness-s23
)

for workspace in "${WORKSPACES[@]}"; do
  if [[ -d "${workspace}" ]]; then
    "${PYTHON_BIN}" -m wa_diloco.summarize_results "${workspace}" \
      >"results/stage2/summary-$(basename "${workspace}").json"
  fi
done

"${PYTHON_BIN}" scripts/compare_workspaces.py --mode full "${WORKSPACES[@]}" \
  >results/stage2/table-full.md
"${PYTHON_BIN}" scripts/compare_workspaces.py --mode tokens "${WORKSPACES[@]}" \
  >results/stage2/table-token-matched.md
"${PYTHON_BIN}" scripts/compare_workspaces.py --mode wall-clock "${WORKSPACES[@]}" \
  >results/stage2/table-wall-clock-matched.md
"${PYTHON_BIN}" scripts/aggregate_stage2_results.py --output-dir results/stage2 "${WORKSPACES[@]}"

ls -lh results/stage2/*.md results/stage2/summary-*.json results/stage2/sidecar-validation/summary.json 2>/dev/null || true

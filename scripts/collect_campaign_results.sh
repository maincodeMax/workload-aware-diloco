#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${WA_DILOCO_REPO:-$PWD}"
PYTHON_BIN="${WA_DILOCO_PYTHON:-${REPO_ROOT}/.venv/bin/python}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
mkdir -p results/campaign

WORKSPACES=(
  workspaces/campaign/fixed-h-256-s23-token-match
  workspaces/campaign/fixed-h-256-s29-token-match
  workspaces/campaign/fixed-h-256-s31-token-match
  workspaces/campaign/wa-balanced-s23
  workspaces/campaign/wa-balanced-s29
  workspaces/campaign/wa-balanced-s31
  workspaces/campaign/wa-current-s23
  workspaces/campaign/wa-fast-sync-s23
)

for workspace in "${WORKSPACES[@]}"; do
  if [[ -d "${workspace}" ]]; then
    "${PYTHON_BIN}" -m wa_diloco.summarize_results "${workspace}" \
      >"results/campaign/summary-$(basename "${workspace}").json"
  fi
done

"${PYTHON_BIN}" scripts/compare_workspaces.py --mode full "${WORKSPACES[@]}" \
  >results/campaign/table-full.md
"${PYTHON_BIN}" scripts/compare_workspaces.py --mode tokens "${WORKSPACES[@]}" \
  >results/campaign/table-token-matched.md
"${PYTHON_BIN}" scripts/compare_workspaces.py --mode wall-clock "${WORKSPACES[@]}" \
  >results/campaign/table-wall-clock-matched.md

ls -lh results/campaign/table-*.md results/campaign/summary-*.json 2>/dev/null || true

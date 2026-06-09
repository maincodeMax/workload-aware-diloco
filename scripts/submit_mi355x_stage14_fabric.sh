#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${WA_DILOCO_REPO:-/shared/training/${USER}/wa-diloco-fabric/repo}"
RESULT_ROOT="${WA_DILOCO_FABRIC_RESULT_ROOT:-/shared/training/${USER}/wa-diloco-fabric/results}"
SBATCH_SCRIPT="${REPO_ROOT}/slurm/mi355x-fabric-probe.sbatch"

# Optional comma-separated node list for sites that require explicit placement.
# If unset, Slurm chooses nodes from the requested partition.
FABRIC_NODELIST="${FABRIC_NODELIST:-}"

mkdir -p "${RESULT_ROOT}/logs" "${RESULT_ROOT}/stage14-fabric"

submit_one() {
  local nodes="$1"
  local dependency="${2:-}"
  local list="scheduler-selected"
  local args=(
    --parsable
    --nodes="${nodes}"
    --ntasks="${nodes}"
    --job-name="wa14-fabric-${nodes}n"
    --export="ALL,WA_DILOCO_REPO=${REPO_ROOT},WA_DILOCO_FABRIC_RESULT_ROOT=${RESULT_ROOT}"
  )
  if [[ -n "${FABRIC_NODELIST}" ]]; then
    IFS=',' read -r -a all_nodes <<< "${FABRIC_NODELIST}"
    if (( ${#all_nodes[@]} < nodes )); then
      echo "FABRIC_NODELIST has ${#all_nodes[@]} nodes but ${nodes} requested" >&2
      exit 2
    fi
    list="$(IFS=,; echo "${all_nodes[*]:0:${nodes}}")"
    args+=(--nodelist="${list}")
  fi
  if [[ -n "${dependency}" ]]; then
    args+=(--dependency="afterok:${dependency}")
  fi
  local job_id
  job_id="$(sbatch "${args[@]}" "${SBATCH_SCRIPT}")"
  printf "%s\t%s\n" "${job_id}" "${list}"
}

read -r job2 list2 < <(submit_one 2)
read -r job4 list4 < <(submit_one 4 "${job2}")
read -r job6 list6 < <(submit_one 6 "${job4}")

manifest="${RESULT_ROOT}/stage14-fabric/manifest-$(date -u +%Y%m%dT%H%M%SZ).tsv"
{
  printf "nodes\tjob_id\tnodelist\n"
  printf "2\t%s\t%s\n" "${job2}" "${list2}"
  printf "4\t%s\t%s\n" "${job4}" "${list4}"
  printf "6\t%s\t%s\n" "${job6}" "${list6}"
} | tee "${manifest}"

echo "submitted stage14 fabric probe chain: ${job2} -> ${job4} -> ${job6}"

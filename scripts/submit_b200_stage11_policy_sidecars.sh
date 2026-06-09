#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${WA_DILOCO_REPO:-$PWD}"
SUBMIT_AGGREGATE="${WA_DILOCO_STAGE11_AGGREGATE:-1}"

cd "${REPO_ROOT}"
mkdir -p results/stage11

manifest="results/stage11/submitted-$(date -u +%Y%m%dT%H%M%S).tsv"
printf "job_id\tregime\tpolicy\tseed\tsync_events\toutput_dir\n" >"${manifest}"

policy_workspace() {
  local policy="$1"
  local seed="$2"
  case "${policy}:${seed}" in
    fixed_h_256:23|fixed_h_256:29|fixed_h_256:31)
      printf "workspaces/stage2/fixed-h-256-s%s/events/coordinator.jsonl" "${seed}" ;;
    fixed_h_256:37|fixed_h_256:41)
      printf "workspaces/stage5/fixed-h-256-s%s/events/coordinator.jsonl" "${seed}" ;;
    wa_diloco:23|wa_diloco:29|wa_diloco:31)
      printf "workspaces/stage2/wa-current-s%s/events/coordinator.jsonl" "${seed}" ;;
    wa_diloco:37|wa_diloco:41)
      printf "workspaces/stage5/wa-current-s%s/events/coordinator.jsonl" "${seed}" ;;
    fixed_h_512:23|fixed_h_512:29|fixed_h_512:31)
      printf "workspaces/stage4/fixed-h-512-s%s/events/coordinator.jsonl" "${seed}" ;;
    fixed_h_512:37|fixed_h_512:41)
      printf "workspaces/stage5/fixed-h-512-s%s/events/coordinator.jsonl" "${seed}" ;;
    pressure_gate:23|pressure_gate:29|pressure_gate:31)
      printf "workspaces/stage4/pressure-gate-s%s/events/coordinator.jsonl" "${seed}" ;;
    pressure_gate:37|pressure_gate:41)
      printf "workspaces/stage5/pressure-gate-s%s/events/coordinator.jsonl" "${seed}" ;;
    pressure_gate_matched:23|pressure_gate_matched:29|pressure_gate_matched:31|pressure_gate_matched:37|pressure_gate_matched:41)
      printf "workspaces/stage7/pressure-gate-matched-s%s/events/coordinator.jsonl" "${seed}" ;;
    random_matched:23|random_matched:29|random_matched:31|random_matched:37|random_matched:41)
      printf "workspaces/stage7/random-matched-s%s/events/coordinator.jsonl" "${seed}" ;;
    *)
      return 1 ;;
  esac
}

policies=(fixed_h_256 fixed_h_512 pressure_gate pressure_gate_matched random_matched wa_diloco)
seeds=(23 29 31 37 41)
regimes=(pressure sync)
job_ids=()

for regime in "${regimes[@]}"; do
  for seed in "${seeds[@]}"; do
    for policy in "${policies[@]}"; do
      sync_events="$(policy_workspace "${policy}" "${seed}")"
      if [[ ! -f "${sync_events}" ]]; then
        echo "missing sync trace: policy=${policy} seed=${seed} path=${sync_events}" >&2
        exit 2
      fi

      run_name="stage11-${regime}-${policy}-s${seed}"
      output_dir="results/stage11/${regime}/${policy}-s${seed}"
      mkdir -p "$(dirname "${output_dir}")"
      export_arg="ALL,WA_DILOCO_SIDECAR_RUN_NAME=${run_name},WA_DILOCO_SIDECAR_OUTPUT_DIR=${output_dir},WA_DILOCO_SYNC_EVENTS_JSONL=${sync_events},WA_DILOCO_SIDECAR_STRESS_MODE=${regime},WA_DILOCO_SIDECAR_MODEL=Qwen/Qwen2.5-1.5B-Instruct,WA_DILOCO_SIDECAR_DURATION_SEC=${WA_DILOCO_STAGE11_DURATION_SEC:-600},WA_DILOCO_SIDECAR_CONCURRENCY=${WA_DILOCO_STAGE11_CONCURRENCY:-8},WA_DILOCO_SIDECAR_PROMPT_STYLE=mixed,WA_DILOCO_SIDECAR_MAX_NEW_TOKENS=128,WA_DILOCO_SIDECAR_SLO_MS=${WA_DILOCO_STAGE11_SLO_MS:-1000},WA_DILOCO_SIDECAR_STRESS_MATMUL_SIZE=6144,WA_DILOCO_SIDECAR_STRESS_STEPS=20,WA_DILOCO_SIDECAR_GPU_MEMORY_UTILIZATION=0.80,WA_DILOCO_SIDECAR_MAX_MODEL_LEN=4096"

      job_id="$(sbatch --parsable --export="${export_arg}" slurm/stage11-real-policy-sidecar.sbatch)"
      job_ids+=("${job_id}")
      printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${job_id}" "${regime}" "${policy}" "${seed}" "${sync_events}" "${output_dir}" \
        | tee -a "${manifest}"
    done
  done
done

if [[ "${SUBMIT_AGGREGATE}" == "1" && "${#job_ids[@]}" -gt 0 ]]; then
  dependency="$(IFS=:; echo "${job_ids[*]}")"
  aggregate_job_id="$(sbatch --parsable --dependency="afterany:${dependency}" slurm/stage11-aggregate.sbatch)"
  printf "%s\taggregate\tstage11\tall\t\tresults/stage11\n" "${aggregate_job_id}" | tee -a "${manifest}"
fi

echo "manifest=${manifest}"

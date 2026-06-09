#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${WA_DILOCO_REPO:-$PWD}"
SUBMIT_AGGREGATE="${WA_DILOCO_STAGE12_AGGREGATE:-1}"
RANDOM_DRAWS="${WA_DILOCO_STAGE12_RANDOM_DRAWS:-5}"

cd "${REPO_ROOT}"
mkdir -p results/stage12

python3 scripts/generate_stage12_schedules.py \
  --output-dir results/stage12/schedules \
  --duration-sec "${WA_DILOCO_STAGE12_DURATION_SEC:-600}" \
  --sync-window-sec "${WA_DILOCO_STAGE12_SYNC_WINDOW_SEC:-8}" \
  --burst-period-sec "${WA_DILOCO_STAGE12_BURST_PERIOD_SEC:-120}" \
  --burst-window-sec "${WA_DILOCO_STAGE12_BURST_WINDOW_SEC:-30}" \
  --burst-phase-sec "${WA_DILOCO_STAGE12_BURST_PHASE_SEC:-0}" \
  --random-draws "${RANDOM_DRAWS}"

manifest="results/stage12/submitted-$(date -u +%Y%m%dT%H%M%S).tsv"
printf "job_id\tregime\tpolicy\tseed\tdraw\tsync_events\toutput_dir\n" >"${manifest}"

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
    calibrated_wa:23|calibrated_wa:29|calibrated_wa:31|calibrated_wa:37|calibrated_wa:41)
      printf "results/stage12/schedules/calibrated_wa-s%s.jsonl" "${seed}" ;;
    no_sync:23|no_sync:29|no_sync:31|no_sync:37|no_sync:41)
      printf "results/stage12/schedules/no_sync-s%s.jsonl" "${seed}" ;;
    random_matched_d*:*)
      local draw="${policy#random_matched_d}"
      printf "results/stage12/schedules/random_matched_d%s-s%s.jsonl" "${draw}" "${seed}" ;;
    *)
      return 1 ;;
  esac
}

seeds=(23 29 31 37 41)
base_policies=(fixed_h_256 fixed_h_512 pressure_gate wa_diloco calibrated_wa no_sync)
policies=("${base_policies[@]}")
for ((draw=0; draw<RANDOM_DRAWS; draw++)); do
  policies+=("random_matched_d${draw}")
done

job_ids=()
for seed in "${seeds[@]}"; do
  for policy in "${policies[@]}"; do
    sync_events="$(policy_workspace "${policy}" "${seed}")"
    if [[ ! -f "${sync_events}" ]]; then
      echo "missing sync trace: policy=${policy} seed=${seed} path=${sync_events}" >&2
      exit 2
    fi

    draw=""
    grouped_policy="${policy}"
    if [[ "${policy}" == random_matched_d* ]]; then
      draw="${policy#random_matched_d}"
      grouped_policy="random_matched"
    fi

    run_name="stage12-bursty-sync-${policy}-s${seed}"
    output_dir="results/stage12/bursty_sync/${policy}-s${seed}"
    sync_windows=1
    stress_mode=sync
    if [[ "${policy}" == "no_sync" ]]; then
      sync_windows=0
      stress_mode=off
    fi

    export_arg="ALL,WA_DILOCO_SIDECAR_RUN_NAME=${run_name},WA_DILOCO_STAGE12_POLICY=${grouped_policy},WA_DILOCO_STAGE12_DRAW=${draw},WA_DILOCO_SIDECAR_OUTPUT_DIR=${output_dir},WA_DILOCO_SYNC_EVENTS_JSONL=${sync_events},WA_DILOCO_SIDECAR_SYNC_WINDOWS=${sync_windows},WA_DILOCO_SIDECAR_STRESS_MODE=${stress_mode},WA_DILOCO_SIDECAR_MODEL=Qwen/Qwen2.5-1.5B-Instruct,WA_DILOCO_SIDECAR_DURATION_SEC=${WA_DILOCO_STAGE12_DURATION_SEC:-600},WA_DILOCO_SIDECAR_LOAD_PROFILE=bursty,WA_DILOCO_SIDECAR_QUIET_CONCURRENCY=${WA_DILOCO_STAGE12_QUIET_CONCURRENCY:-4},WA_DILOCO_SIDECAR_BUSY_CONCURRENCY=${WA_DILOCO_STAGE12_BUSY_CONCURRENCY:-12},WA_DILOCO_SIDECAR_BURST_PERIOD_SEC=${WA_DILOCO_STAGE12_BURST_PERIOD_SEC:-120},WA_DILOCO_SIDECAR_BURST_WINDOW_SEC=${WA_DILOCO_STAGE12_BURST_WINDOW_SEC:-30},WA_DILOCO_SIDECAR_BURST_PHASE_SEC=${WA_DILOCO_STAGE12_BURST_PHASE_SEC:-0},WA_DILOCO_SIDECAR_PROMPT_STYLE=mixed,WA_DILOCO_SIDECAR_MAX_NEW_TOKENS=128,WA_DILOCO_SIDECAR_SLO_MS=${WA_DILOCO_STAGE12_SLO_MS:-1000},WA_DILOCO_SIDECAR_STRESS_MATMUL_SIZE=6144,WA_DILOCO_SIDECAR_STRESS_STEPS=20,WA_DILOCO_SIDECAR_GPU_MEMORY_UTILIZATION=0.80,WA_DILOCO_SIDECAR_MAX_MODEL_LEN=4096"

    job_id="$(sbatch --parsable --export="${export_arg}" slurm/stage12-bursty-sidecar.sbatch)"
    job_ids+=("${job_id}")
    printf "%s\tbursty_sync\t%s\t%s\t%s\t%s\t%s\n" \
      "${job_id}" "${grouped_policy}" "${seed}" "${draw}" "${sync_events}" "${output_dir}" \
      | tee -a "${manifest}"
  done
done

if [[ "${SUBMIT_AGGREGATE}" == "1" && "${#job_ids[@]}" -gt 0 ]]; then
  dependency="$(IFS=:; echo "${job_ids[*]}")"
  aggregate_job_id="$(sbatch --parsable --dependency="afterany:${dependency}" slurm/stage12-aggregate.sbatch)"
  printf "%s\taggregate\tstage12\tall\t\t\tresults/stage12\n" "${aggregate_job_id}" | tee -a "${manifest}"
fi

echo "manifest=${manifest}"

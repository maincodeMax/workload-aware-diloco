#!/usr/bin/env bash
set -euo pipefail

python3 scripts/generate_stage13_online_configs.py >/tmp/stage13-configs.txt

seeds=(23 29 31 37 41)
train_ids=()
for seed in "${seeds[@]}"; do
  config="configs/stage13/wa_calibrated_s${seed}.yaml"
  job_id="$(
    sbatch --parsable \
      --job-name="wa13-train-s${seed}" \
      --export="ALL,WA_DILOCO_PRESSURE_MODE=bursty,WA_DILOCO_PRESSURE_PERIOD_SEC=${WA_DILOCO_STAGE13_BURST_PERIOD_SEC:-120},WA_DILOCO_PRESSURE_BURST_WINDOW_SEC=${WA_DILOCO_STAGE13_BURST_WINDOW_SEC:-30},WA_DILOCO_PRESSURE_BURST_PHASE_SEC=${WA_DILOCO_STAGE13_BURST_PHASE_SEC:-0},WA_DILOCO_SERVING_SYNC_WINDOW_SEC=${WA_DILOCO_STAGE13_SYNC_WINDOW_SEC:-8},WA_DILOCO_SERVING_REQUESTS_PER_INTERVAL=${WA_DILOCO_STAGE13_REQUESTS_PER_INTERVAL:-32},WA_DILOCO_EVAL_DATASET=${WA_DILOCO_EVAL_DATASET:-Salesforce/wikitext},WA_DILOCO_EVAL_CONFIG=${WA_DILOCO_EVAL_CONFIG:-wikitext-2-raw-v1},WA_DILOCO_EVAL_SPLIT=${WA_DILOCO_EVAL_SPLIT:-validation},WA_DILOCO_EVAL_BATCH_SIZE=${WA_DILOCO_EVAL_BATCH_SIZE:-4},WA_DILOCO_EVAL_MAX_BATCHES=${WA_DILOCO_EVAL_MAX_BATCHES:-64}" \
      slurm/b200-8x.sbatch "${config}"
  )"
  train_ids+=("${job_id}")
  printf "train\t%s\t%s\t%s\n" "${job_id}" "${seed}" "${config}"
done

train_dep="$(IFS=:; echo "${train_ids[*]}")"
sidecar_ids=()
manifest="results/stage13/manifest.tsv"
mkdir -p results/stage13
printf "job_id\tpolicy\tseed\tsync_events\toutput_dir\n" >"${manifest}"
for seed in "${seeds[@]}"; do
  sync_events="workspaces/stage13/wa-calibrated-s${seed}/events/coordinator.jsonl"
  output_dir="results/stage13/bursty_sync/online_calibrated_wa-s${seed}"
  run_name="stage13-online-calibrated-wa-s${seed}"
  job_id="$(
    sbatch --parsable \
      --dependency="afterok:${train_dep}" \
      --export="ALL,WA_DILOCO_SIDECAR_RUN_NAME=${run_name},WA_DILOCO_STAGE12_POLICY=online_calibrated_wa,WA_DILOCO_SIDECAR_OUTPUT_DIR=${output_dir},WA_DILOCO_SYNC_EVENTS_JSONL=${sync_events},WA_DILOCO_SIDECAR_SYNC_WINDOWS=1,WA_DILOCO_SIDECAR_STRESS_MODE=sync,WA_DILOCO_SIDECAR_MODEL=Qwen/Qwen2.5-1.5B-Instruct,WA_DILOCO_SIDECAR_DURATION_SEC=${WA_DILOCO_STAGE13_DURATION_SEC:-600},WA_DILOCO_SIDECAR_LOAD_PROFILE=bursty,WA_DILOCO_SIDECAR_QUIET_CONCURRENCY=${WA_DILOCO_STAGE13_QUIET_CONCURRENCY:-4},WA_DILOCO_SIDECAR_BUSY_CONCURRENCY=${WA_DILOCO_STAGE13_BUSY_CONCURRENCY:-12},WA_DILOCO_SIDECAR_BURST_PERIOD_SEC=${WA_DILOCO_STAGE13_BURST_PERIOD_SEC:-120},WA_DILOCO_SIDECAR_BURST_WINDOW_SEC=${WA_DILOCO_STAGE13_BURST_WINDOW_SEC:-30},WA_DILOCO_SIDECAR_BURST_PHASE_SEC=${WA_DILOCO_STAGE13_BURST_PHASE_SEC:-0},WA_DILOCO_SIDECAR_PROMPT_STYLE=mixed,WA_DILOCO_SIDECAR_MAX_NEW_TOKENS=128,WA_DILOCO_SIDECAR_SLO_MS=${WA_DILOCO_STAGE13_SLO_MS:-1000},WA_DILOCO_SIDECAR_STRESS_MATMUL_SIZE=6144,WA_DILOCO_SIDECAR_STRESS_STEPS=20,WA_DILOCO_SIDECAR_GPU_MEMORY_UTILIZATION=0.80,WA_DILOCO_SIDECAR_MAX_MODEL_LEN=4096" \
      slurm/stage12-bursty-sidecar.sbatch
  )"
  sidecar_ids+=("${job_id}")
  printf "%s\tonline_calibrated_wa\t%s\t%s\t%s\n" "${job_id}" "${seed}" "${sync_events}" "${output_dir}" | tee -a "${manifest}"
done

sidecar_dep="$(IFS=:; echo "${sidecar_ids[*]}")"
agg_id="$(sbatch --parsable --dependency="afterok:${sidecar_dep}" slurm/stage13-aggregate.sbatch)"
printf "aggregate\t%s\n" "${agg_id}"

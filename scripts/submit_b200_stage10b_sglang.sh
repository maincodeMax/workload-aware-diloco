#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${WA_DILOCO_REPO:-$PWD}"
DEPENDENCY_JOB="${WA_DILOCO_DEPENDENCY_JOB:-}"
DEPENDENCY_TYPE="${WA_DILOCO_DEPENDENCY_TYPE:-afterany}"

cd "${REPO_ROOT}"
mkdir -p results/stage10b

manifest="results/stage10b/submitted-$(date -u +%Y%m%dT%H%M%S).tsv"
printf "job_id\tkind\trun_name\tmodel\tload\toutput_dir\n" >"${manifest}"

RUNS=(
  "sglang-1p5b-baseline|Qwen/Qwen2.5-1.5B-Instruct|1200|4|mixed|96|both|4096|12|0.75|4096"
  "sglang-1p5b-highcon|Qwen/Qwen2.5-1.5B-Instruct|1200|8|long|128|both|6144|16|0.80|4096"
)

previous_job="${DEPENDENCY_JOB}"
for spec in "${RUNS[@]}"; do
  IFS='|' read -r run_name model duration concurrency prompt_style max_new stress_mode stress_size stress_steps gpu_mem max_model_len <<<"${spec}"
  output_dir="results/stage10b/${run_name}"
  export_arg="ALL,WA_DILOCO_SIDECAR_RUN_NAME=${run_name},WA_DILOCO_SIDECAR_OUTPUT_DIR=${output_dir},WA_DILOCO_SIDECAR_MODEL=${model},WA_DILOCO_SIDECAR_DURATION_SEC=${duration},WA_DILOCO_SIDECAR_CONCURRENCY=${concurrency},WA_DILOCO_SIDECAR_PROMPT_STYLE=${prompt_style},WA_DILOCO_SIDECAR_MAX_NEW_TOKENS=${max_new},WA_DILOCO_SIDECAR_STRESS_MODE=${stress_mode},WA_DILOCO_SIDECAR_STRESS_MATMUL_SIZE=${stress_size},WA_DILOCO_SIDECAR_STRESS_STEPS=${stress_steps},WA_DILOCO_SIDECAR_GPU_MEMORY_UTILIZATION=${gpu_mem},WA_DILOCO_SIDECAR_MAX_MODEL_LEN=${max_model_len}"

  if [[ -n "${previous_job}" ]]; then
    job_id="$(sbatch --parsable --dependency="${DEPENDENCY_TYPE}:${previous_job}" --export="${export_arg}" slurm/stage10b-sglang-sidecar.sbatch)"
  else
    job_id="$(sbatch --parsable --export="${export_arg}" slurm/stage10b-sglang-sidecar.sbatch)"
  fi

  printf "%s\tsglang\t%s\t%s\tc=%s,%s,new=%s,stress=%s\t%s\n" \
    "${job_id}" "${run_name}" "${model}" "${concurrency}" "${prompt_style}" "${max_new}" "${stress_mode}" "${output_dir}" \
    | tee -a "${manifest}"
  previous_job="${job_id}"
done

aggregate_job_id="$(sbatch --parsable --dependency="afterany:${previous_job}" slurm/stage10b-aggregate.sbatch)"
printf "%s\taggregate\t%s\t%s\t%s\t%s\n" \
  "${aggregate_job_id}" "stage10b-aggregate" "" "" "results/stage10b" | tee -a "${manifest}"

echo "manifest=${manifest}"

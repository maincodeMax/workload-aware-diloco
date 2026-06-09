#!/usr/bin/env bash
# MI355X multi-node fabric probe for DiLoCo outer-sync payloads.

#SBATCH --job-name=wa14-fabric
#SBATCH --partition=mi355x
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:mi355x:8
#SBATCH --exclusive
#SBATCH --time=00:45:00
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${WA_DILOCO_REPO:-$PWD}"
RESULT_ROOT="${WA_DILOCO_FABRIC_RESULT_ROOT:-$PWD/results}"
RUN_ID="${SLURM_JOB_ID:-manual-$(date -u +%Y%m%dT%H%M%S)}-${SLURM_JOB_NUM_NODES:-nodes}n"
RESULT_DIR="${RESULT_ROOT}/stage14-fabric/${RUN_ID}"
IMAGE="${FABRIC_ROCM_IMAGE:-}"
IMAGE_TAR="${FABRIC_ROCM_IMAGE_TAR:-}"
EXPECTED_IMAGE_ID="${FABRIC_ROCM_IMAGE_ID:-}"
GPUS_PER_NODE="${FABRIC_GPUS_PER_NODE:-8}"
NNODES="${SLURM_JOB_NUM_NODES}"
MASTER_NODE="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
MASTER_ADDR="$(
  scontrol show node -o "${MASTER_NODE}" |
    sed -n 's/.*NodeAddr=\([^ ]*\).*/\1/p'
)"
MASTER_PORT="${FABRIC_MASTER_PORT:-$((29500 + SLURM_JOB_ID % 1000))}"
NCCL_IB_HCA="${NCCL_IB_HCA:-rocep3s0,rocep55s0,rocep169s0,rocep214s0,roceP1p3s0,roceP1p53s0,roceP1p166s0,roceP1p211s0}"
SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0.310}"

mkdir -p "${RESULT_DIR}" "${RESULT_ROOT}/logs"
export REPO_ROOT RESULT_DIR IMAGE IMAGE_TAR EXPECTED_IMAGE_ID
export GPUS_PER_NODE NNODES MASTER_ADDR MASTER_PORT NCCL_IB_HCA SOCKET_IFNAME

if [[ -z "${IMAGE}" ]]; then
  echo "Set FABRIC_ROCM_IMAGE to a ROCm/PyTorch container available on your cluster." >&2
  exit 2
fi

{
  echo "run_id=${RUN_ID}"
  echo "job_id=${SLURM_JOB_ID}"
  echo "started_utc=$(date -u --iso-8601=seconds)"
  echo "nodes=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | paste -sd, -)"
  echo "nnodes=${NNODES}"
  echo "gpus_per_node=${GPUS_PER_NODE}"
  echo "world_size=$((NNODES * GPUS_PER_NODE))"
  echo "master=${MASTER_ADDR}:${MASTER_PORT}"
  echo "repo=${REPO_ROOT}"
  echo "image=${IMAGE}"
  echo "sizes_mib=${FABRIC_BENCH_SIZES_MIB:-64,256,1024}"
  echo "warmup=${FABRIC_BENCH_WARMUP_ITERS:-3}"
  echo "iters=${FABRIC_BENCH_MEASURE_ITERS:-8}"
  echo "slurm_job_nodelist=${SLURM_JOB_NODELIST}"
} | tee "${RESULT_DIR}/run-metadata.txt"

srun --ntasks="${NNODES}" --ntasks-per-node=1 --kill-on-bad-exit=1 \
  bash -lc 'set -euo pipefail
    mkdir -p /scratch/containers/${USER}/run /scratch/containers/${USER}/tmp "$1/nodes/$(hostname -s)"
    if [[ -n "$2" ]]; then
      test -f "$2"
    fi
  ' bash "${RESULT_DIR}" "${IMAGE_TAR}"

srun --ntasks="${NNODES}" --ntasks-per-node=1 --kill-on-bad-exit=1 \
  bash -lc '
    set -euo pipefail
    export HOME="${HOME:-/home/${USER}}"
    export XDG_RUNTIME_DIR=/scratch/containers/${USER}/run
    export PODMAN_TMPDIR=/scratch/containers/${USER}/tmp
    export TMPDIR=/scratch/containers/${USER}/tmp/${SLURM_JOB_ID}/${SLURM_NODEID}
    mkdir -p "${XDG_RUNTIME_DIR}" "${PODMAN_TMPDIR}" "${TMPDIR}"
    chmod 700 "${XDG_RUNTIME_DIR}" "${PODMAN_TMPDIR}" "${TMPDIR}"

    node_dir="${RESULT_DIR}/nodes/$(hostname -s)"
    mkdir -p "${node_dir}/logs"
    image_id="$(podman image inspect --format="{{.Id}}" "${IMAGE}" 2>/dev/null || true)"
    if [[ -z "${image_id}" || ( -n "${EXPECTED_IMAGE_ID}" && "${image_id}" != "${EXPECTED_IMAGE_ID}" ) ]]; then
      if [[ -z "${IMAGE_TAR}" ]]; then
        echo "image ${IMAGE} not found locally and FABRIC_ROCM_IMAGE_TAR is unset" >&2
        exit 30
      fi
      echo "loading_image=${IMAGE_TAR}" | tee "${node_dir}/logs/podman-load.log"
      podman load -i "${IMAGE_TAR}" >> "${node_dir}/logs/podman-load.log" 2>&1
      image_id="$(podman image inspect --format="{{.Id}}" "${IMAGE}")"
    fi
    if [[ -n "${EXPECTED_IMAGE_ID}" && "${image_id}" != "${EXPECTED_IMAGE_ID}" ]]; then
      echo "image_id_mismatch got=${image_id} expected=${EXPECTED_IMAGE_ID}" >&2
      exit 31
    fi

    podman run --rm --init \
      --name="wa14_fabric_${SLURM_JOB_ID}_${SLURM_NODEID}" \
      --group-add keep-groups \
      --ulimit nofile=131072:131072 \
      --ulimit memlock=-1:-1 \
      --ulimit stack=67108864 \
      --cap-add IPC_LOCK \
      --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
      -v /sys/class/infiniband:/sys/class/infiniband:ro \
      -v /sys/class/infiniband_verbs:/sys/class/infiniband_verbs:ro \
      --security-opt label=disable \
      --security-opt seccomp=unconfined \
      --network=host --ipc=host \
      -v "${REPO_ROOT}:/workspace/repo:ro" \
      -v "${RESULT_DIR}:/results:rw" \
      -e NCCL_IB_DISABLE=0 \
      -e NCCL_IB_HCA="${NCCL_IB_HCA}" \
      -e NCCL_IB_GID_INDEX=3 \
      -e NCCL_SOCKET_IFNAME="${SOCKET_IFNAME}" \
      -e GLOO_SOCKET_IFNAME="${SOCKET_IFNAME}" \
      -e NCCL_DEBUG="${NCCL_DEBUG:-WARN}" \
      -e RCCL_LOG_LEVEL="${RCCL_LOG_LEVEL:-WARN}" \
      -e NNODES="${NNODES}" \
      -e GPUS_PER_NODE="${GPUS_PER_NODE}" \
      -e NODE_RANK="${SLURM_NODEID}" \
      -e MASTER_ADDR="${MASTER_ADDR}" \
      -e MASTER_PORT="${MASTER_PORT}" \
      -e WORLD_SIZE="$(( NNODES * GPUS_PER_NODE ))" \
      -e FABRIC_BENCH_SIZES_MIB="${FABRIC_BENCH_SIZES_MIB:-64,256,1024}" \
      -e FABRIC_BENCH_WARMUP_ITERS="${FABRIC_BENCH_WARMUP_ITERS:-3}" \
      -e FABRIC_BENCH_MEASURE_ITERS="${FABRIC_BENCH_MEASURE_ITERS:-8}" \
      -e FABRIC_RESULT_DIR=/results \
      "${IMAGE}" \
      bash -lc "python3 -m torch.distributed.run \
        --nnodes=\"${NNODES}\" \
        --nproc_per_node=\"${GPUS_PER_NODE}\" \
        --node_rank=\"${SLURM_NODEID}\" \
        --master_addr=\"${MASTER_ADDR}\" \
        --master_port=\"${MASTER_PORT}\" \
        /workspace/repo/scripts/diloco_fabric_probe.py" \
      2>&1 | tee "${node_dir}/logs/torchrun.log"
  '

echo "finished_utc=$(date -u --iso-8601=seconds)" | tee -a "${RESULT_DIR}/run-metadata.txt"

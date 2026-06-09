#!/usr/bin/env bash
# Isolated vLLM sidecar validation. This intentionally does not use the training venv.

#SBATCH --job-name=wa-vllm-sidecar
#SBATCH --partition=resident
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=results/stage3-vllm-sidecar/vllm-sidecar-%j.out
#SBATCH --error=results/stage3-vllm-sidecar/vllm-sidecar-%j.err

set -euo pipefail
ulimit -c 0

REPO_ROOT="${WA_DILOCO_REPO:-$PWD}"
VLLM_VENV="${WA_DILOCO_VLLM_VENV:-${REPO_ROOT}/.venv-vllm-0102}"
VLLM_VERSION="${WA_DILOCO_VLLM_VERSION:-0.10.2}"
TRANSFORMERS_VERSION="${WA_DILOCO_TRANSFORMERS_VERSION:-4.55.4}"
TOKENIZERS_VERSION="${WA_DILOCO_TOKENIZERS_VERSION:-0.21.4}"

cd "${REPO_ROOT}"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID

mkdir -p results/stage3-vllm-sidecar
SIDECAR_OUTPUT_DIR="${WA_DILOCO_SIDECAR_OUTPUT_DIR:-results/stage3-vllm-sidecar/sidecar-validation}"
mkdir -p "$(dirname "${SIDECAR_OUTPUT_DIR}")"

if [[ ! -x "${VLLM_VENV}/bin/python" ]]; then
  python3 -m venv "${VLLM_VENV}"
  "${VLLM_VENV}/bin/python" -m pip install -U pip
fi

if ! "${VLLM_VENV}/bin/python" - "${VLLM_VERSION}" "${TRANSFORMERS_VERSION}" "${TOKENIZERS_VERSION}" <<'PY'
import importlib.metadata as metadata
import sys

expected = {
    "vllm": sys.argv[1],
    "transformers": sys.argv[2],
    "tokenizers": sys.argv[3],
}
for package, version in expected.items():
    try:
        installed = metadata.version(package)
    except metadata.PackageNotFoundError:
        raise SystemExit(1)
    if installed != version:
        raise SystemExit(1)
raise SystemExit(0)
PY
then
  "${VLLM_VENV}/bin/python" -m pip install \
    "vllm==${VLLM_VERSION}" \
    "transformers==${TRANSFORMERS_VERSION}" \
    "tokenizers==${TOKENIZERS_VERSION}"
fi

"${VLLM_VENV}/bin/python" scripts/real_sidecar_validation.py \
  --output-dir "${SIDECAR_OUTPUT_DIR}" \
  --model "${WA_DILOCO_SIDECAR_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}" \
  --duration-sec "${WA_DILOCO_SIDECAR_DURATION_SEC:-240}" \
  --period-sec "${WA_DILOCO_SIDECAR_PERIOD_SEC:-90}" \
  --sync-period-sec "${WA_DILOCO_SIDECAR_SYNC_PERIOD_SEC:-37}" \
  --sync-window-sec "${WA_DILOCO_SIDECAR_SYNC_WINDOW_SEC:-8}" \
  --max-new-tokens "${WA_DILOCO_SIDECAR_MAX_NEW_TOKENS:-24}" \
  --stress-mode "${WA_DILOCO_SIDECAR_STRESS_MODE:-both}" \
  --backend vllm

cat "${SIDECAR_OUTPUT_DIR}/summary.json"

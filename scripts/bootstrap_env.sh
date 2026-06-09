#!/usr/bin/env bash
# Prepare a lightweight local Python environment for this repo.
#
# Defaults intentionally avoid downloading PyTorch/Transformers/Datasets.
# Set INSTALL_DEPS=1 only on a node/environment where large ML wheels are acceptable.

set -euo pipefail

REPO_ROOT="${WA_DILOCO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${WA_DILOCO_VENV:-${REPO_ROOT}/.venv}"
PYTHON_BIN="${WA_DILOCO_PYTHON:-python3}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"
TORCH_VERSION="${WA_DILOCO_TORCH_VERSION:-2.11.0+cu128}"
TORCH_INDEX_URL="${WA_DILOCO_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

cd "${REPO_ROOT}"

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

if [[ "${INSTALL_DEPS}" == "1" ]]; then
  python -m pip install --index-url "${TORCH_INDEX_URL}" "torch==${TORCH_VERSION}"
  python -m pip install -e '.[train]'
else
  python -m pip install -e .
fi

python -m compileall src
python -m wa_diloco.coordinator --help >/dev/null

cat <<EOF
WA-DiLoCo environment ready.

Repo:   ${REPO_ROOT}
Venv:   ${VENV_DIR}
Deps:   $([[ "${INSTALL_DEPS}" == "1" ]] && echo "installed" || echo "repo only; ML deps not installed")
Torch:  $([[ "${INSTALL_DEPS}" == "1" ]] && echo "${TORCH_VERSION} from ${TORCH_INDEX_URL}" || echo "not installed by bootstrap")

Activate:
  source ${VENV_DIR}/bin/activate
EOF

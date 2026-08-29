#!/usr/bin/env bash
# Install the LIBERO sim venv on Colab/Linux.
# Always pass --python to uv pip. `source activate` is not reliable here.
# Skip the official gym.make smoke test; Colab extracts LIBERO assets later.
set -euo pipefail

ISAAC_ROOT="${1:-}"
if [[ -z "${ISAAC_ROOT}" ]]; then
  echo "usage: setup_libero_colab.sh /path/to/Isaac-GR00T" >&2
  exit 2
fi

export PATH="${UV_INSTALL_DIR:-/content/uv-bin}:${PATH}"
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv not on PATH. Install Native Runtime Tools first." >&2
  exit 1
fi

LIBERO_REPO="${ISAAC_ROOT}/external_dependencies/LIBERO"
SETUP_DIR="${ISAAC_ROOT}/gr00t/eval/sim/LIBERO"
LIBERO_UV_ENV="${SETUP_DIR}/libero_uv"
PY="${LIBERO_UV_ENV}/.venv/bin/python"
PATCHED_REQUIREMENTS="${LIBERO_UV_ENV}/requirements-py312.txt"

if [[ ! -d "${LIBERO_REPO}" ]]; then
  echo "ERROR: LIBERO sources missing at ${LIBERO_REPO}" >&2
  exit 2
fi

mkdir -p "${LIBERO_UV_ENV}"
if [[ ! -x "${PY}" ]]; then
  uv venv "${LIBERO_UV_ENV}/.venv" --python 3.12
fi

python3 - <<PY
from pathlib import Path

replacements = {
    "hydra-core": "hydra-core==1.3.2",
    "numpy": "numpy==1.26.4",
    "transformers": "transformers==4.57.3",
    "opencv-python": "opencv-python==4.10.0.84",
    "matplotlib": "matplotlib==3.9.4",
    "wandb": "wandb==0.18.7",
}

src = Path("${LIBERO_REPO}/requirements.txt")
dst = Path("${PATCHED_REQUIREMENTS}")
lines = []
for raw in src.read_text().splitlines():
    stripped = raw.strip()
    if not stripped or stripped.startswith("#"):
        lines.append(raw)
        continue
    name = stripped.split("==", 1)[0].strip().lower()
    lines.append(replacements.get(name, raw))
dst.write_text("\n".join(lines) + "\n")
PY

uv pip install --python "${PY}" --requirements "${PATCHED_REQUIREMENTS}"
uv pip install --python "${PY}" -e "${LIBERO_REPO}" --config-settings editable_mode=compat
uv pip install --python "${PY}" \
  torch==2.9.0 torchvision==0.24.0 pydantic av tianshou==0.5.1 \
  numba==0.65.1 llvmlite==0.47.0 tyro pandas dm_tree einops==0.8.1 \
  albumentations==1.4.18 zmq
uv pip install --python "${PY}" \
  transformers==4.57.3 msgpack==1.1.0 msgpack-numpy==0.4.8 gymnasium==0.29.1
uv pip install --python "${PY}" numpy==1.26.4 mujoco==3.3.1

"${PY}" -c "import sysconfig, pathlib; pathlib.Path(sysconfig.get_path('purelib'), 'gr00t.pth').write_text(pathlib.Path('${ISAAC_ROOT}').resolve().as_posix() + chr(10))"

LIBERO_PKG="${ISAAC_ROOT}/external_dependencies/LIBERO/libero/libero"
LIBERO_CFG_DIR="${HOME}/.libero"
mkdir -p "${LIBERO_CFG_DIR}"
cat > "${LIBERO_CFG_DIR}/config.yaml" <<EOF
assets: ${LIBERO_PKG}/assets
bddl_files: ${LIBERO_PKG}/bddl_files
benchmark_root: ${LIBERO_PKG}
datasets: ${ISAAC_ROOT}/external_dependencies/LIBERO/libero/datasets
init_states: ${LIBERO_PKG}/init_files
EOF
"${PY}" -c "import numpy, gymnasium, libero; print('libero ok', numpy.__version__, libero.__file__)"
echo "LIBERO venv ready: ${PY}"

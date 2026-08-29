#!/usr/bin/env bash
# Colab/Linux helper: run official setup_libero.sh without git submodule.
# MI_VLA vendors Isaac-GR00T without its own .git, so submodule init would fail.
# The official script's gym.make smoke test needs LIBERO assets and a working
# EGL context; Colab extracts those later, so skip that tail here.
set -euo pipefail

ISAAC_ROOT="${1:-}"
if [[ -z "${ISAAC_ROOT}" ]]; then
  echo "usage: setup_libero_colab.sh /path/to/Isaac-GR00T" >&2
  exit 2
fi

export PATH="${UV_INSTALL_DIR:-/content/uv-bin}:${PATH}"
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv not on PATH. Install Native Runtime Tools first." >&2
  echo "PATH=${PATH}" >&2
  exit 1
fi

SETUP="${ISAAC_ROOT}/gr00t/eval/sim/LIBERO/setup_libero.sh"
LIBERO_REPO="${ISAAC_ROOT}/external_dependencies/LIBERO"
if [[ ! -f "${SETUP}" ]]; then
  echo "ERROR: missing ${SETUP}" >&2
  exit 2
fi
if [[ ! -d "${LIBERO_REPO}" ]]; then
  echo "ERROR: LIBERO sources missing at ${LIBERO_REPO}" >&2
  exit 2
fi

# Write a sibling script so BASH_SOURCE / SCRIPT_DIR still point at the LIBERO
# setup directory. Piping sed into bash would break those paths.
patched="${SETUP%.sh}.colab.sh"
# Drop submodule init and the env smoke test that wipes ~/.libero then gym.make.
sed \
  -e '/git submodule update/d' \
  -e '/^rm -rf \$HOME\/\.libero$/,$d' \
  "${SETUP}" > "${patched}"
chmod +x "${patched}"
echo "Running patched LIBERO setup (install only, no gym.make smoke test)"
bash "${patched}"
echo "LIBERO venv ready: ${ISAAC_ROOT}/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python"

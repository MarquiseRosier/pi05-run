#!/usr/bin/env bash
# Colab/Linux helper: run official setup_libero.sh without git submodule.
# MI_VLA vendors Isaac-GR00T without its own .git, so submodule init would fail.
set -euo pipefail

ISAAC_ROOT="${1:-}"
if [[ -z "${ISAAC_ROOT}" ]]; then
  echo "usage: setup_libero_colab.sh /path/to/Isaac-GR00T" >&2
  exit 2
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
sed '/git submodule update/d' "${SETUP}" > "${patched}"
chmod +x "${patched}"
bash "${patched}"

#!/usr/bin/env bash
set -euo pipefail

TASKS="${1:-libero_spatial}"
EPISODES="${2:-1}"
VIEW_ACTION="${VIEW_ACTION:-open-dir}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"${ROOT}/scripts/gcp_create_l4_vm.sh"
"${ROOT}/scripts/gcp_run_pi05_libero_uv.sh" "${TASKS}" "${EPISODES}"
"${ROOT}/scripts/gcp_fetch_pi05_results.sh" latest
"${ROOT}/scripts/show_pi05_results.sh" latest "${VIEW_ACTION}"

#!/usr/bin/env bash
set -euo pipefail

TASKS="${1:-libero_spatial}"
EPISODES="${2:-1}"
TASK_IDS="${TASK_IDS:-[0]}"
TASK_ID_FOR_VIDEO="${TASK_ID_FOR_VIDEO:-0}"
STOP_AFTER="${STOP_AFTER:-0}"
CAPTURE_MAX_CHUNKS="${CAPTURE_MAX_CHUNKS:-80}"
CAPTURE_LAYER_STRIDE="${CAPTURE_LAYER_STRIDE:-1}"
CAPTURE_MAX_BINS="${CAPTURE_MAX_BINS:-64}"
CAPTURE_FAMILIES="${CAPTURE_FAMILIES:-vision,prefix,expert,projection}"
CAPTURE_PARAM_STATS="${CAPTURE_PARAM_STATS:-0}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"${ROOT}/scripts/gcp_create_l4_vm.sh"

CAPTURE_ACTIVATIONS=1 \
CAPTURE_MAX_CHUNKS="${CAPTURE_MAX_CHUNKS}" \
CAPTURE_LAYER_STRIDE="${CAPTURE_LAYER_STRIDE}" \
CAPTURE_MAX_BINS="${CAPTURE_MAX_BINS}" \
CAPTURE_FAMILIES="${CAPTURE_FAMILIES}" \
CAPTURE_PARAM_STATS="${CAPTURE_PARAM_STATS}" \
TASK_IDS="${TASK_IDS}" \
"${ROOT}/scripts/gcp_run_pi05_libero_uv.sh" "${TASKS}" "${EPISODES}"

"${ROOT}/scripts/gcp_fetch_pi05_results.sh" latest
"${ROOT}/scripts/make_pi05_analysis_video.py" --run latest --task-id "${TASK_ID_FOR_VIDEO}" --open

if [[ "${STOP_AFTER}" == "1" ]]; then
  gcloud compute instances stop "${VM_NAME:-lerobot-libero-l4}" \
    --project="${PROJECT:-breadwinner-415122}" \
    --zone="${ZONE:-us-central1-a}" \
    --quiet
fi

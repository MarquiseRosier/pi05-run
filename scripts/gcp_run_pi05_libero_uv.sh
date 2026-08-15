#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-breadwinner-415122}"
ZONE="${ZONE:-us-central1-a}"
VM_NAME="${VM_NAME:-lerobot-libero-l4}"
TASKS="${1:-libero_spatial}"
EPISODES="${2:-1}"

REMOTE_ROOT="~/groot-run"

gcloud compute ssh "${VM_NAME}" \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --command="mkdir -p ${REMOTE_ROOT}/cloud ${REMOTE_ROOT}/outputs ~/.cache/huggingface"

gcloud compute scp \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --recurse cloud/libero "${VM_NAME}:${REMOTE_ROOT}/cloud/"

gcloud compute ssh "${VM_NAME}" \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --command="cd ${REMOTE_ROOT} && \
    export PATH=\"\$HOME/.local/bin:\$PATH\" && \
    export HF_HOME=\"\$HOME/.cache/huggingface\" && \
    export HF_HUB_ENABLE_HF_TRANSFER=1 && \
    export HF_XET_HIGH_PERFORMANCE=1 && \
    export MUJOCO_GL=egl && \
    export PYOPENGL_PLATFORM=egl && \
    export MUJOCO_EGL_DEVICE_ID=0 && \
    export MIN_GPU_MEM_GB='${MIN_GPU_MEM_GB:-0}' && \
    export MIN_HOST_RAM_GB='${MIN_HOST_RAM_GB:-0}' && \
    export CAPTURE_ACTIVATIONS='${CAPTURE_ACTIVATIONS:-0}' && \
    export CAPTURE_MAX_CHUNKS='${CAPTURE_MAX_CHUNKS:-80}' && \
    export CAPTURE_LAYER_STRIDE='${CAPTURE_LAYER_STRIDE:-1}' && \
    export CAPTURE_MAX_BINS='${CAPTURE_MAX_BINS:-64}' && \
    export CAPTURE_FAMILIES='${CAPTURE_FAMILIES:-vision,prefix,expert,projection}' && \
    export CAPTURE_PARAM_STATS='${CAPTURE_PARAM_STATS:-0}' && \
    export TASK_IDS='${TASK_IDS:-}' && \
    . .venv/bin/activate && \
    ./cloud/libero/run_pi05_libero.sh '${TASKS}' '${EPISODES}'"

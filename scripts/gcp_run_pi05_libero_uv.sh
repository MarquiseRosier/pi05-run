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
    export RUN_ID='${RUN_ID:-}' && \
    export OUTPUT_ROOT='${OUTPUT_ROOT:-outputs/eval/pi05_libero}' && \
    export SEED='${SEED:-1000}' && \
    export MIN_GPU_MEM_GB='${MIN_GPU_MEM_GB:-0}' && \
    export MIN_HOST_RAM_GB='${MIN_HOST_RAM_GB:-0}' && \
    export CAPTURE_ACTIVATIONS='${CAPTURE_ACTIVATIONS:-0}' && \
    export CAPTURE_MAX_CHUNKS='${CAPTURE_MAX_CHUNKS:-80}' && \
    export CAPTURE_LAYER_STRIDE='${CAPTURE_LAYER_STRIDE:-1}' && \
    export CAPTURE_MAX_BINS='${CAPTURE_MAX_BINS:-64}' && \
    export CAPTURE_FAMILIES='${CAPTURE_FAMILIES:-vision,prefix,expert,projection}' && \
    export CAPTURE_PARAM_STATS='${CAPTURE_PARAM_STATS:-0}' && \
    export CAPTURE_FEEDBACK_TRACE='${CAPTURE_FEEDBACK_TRACE:-1}' && \
    export CAPTURE_ENV_STEPS='${CAPTURE_ENV_STEPS:-1}' && \
    export CAPTURE_ENV_STEP_IMAGES='${CAPTURE_ENV_STEP_IMAGES:-0}' && \
    export CAPTURE_ENV_STEP_IMAGE_EVERY_N='${CAPTURE_ENV_STEP_IMAGE_EVERY_N:-10}' && \
    export CAPTURE_MAX_ENV_STEP_IMAGES='${CAPTURE_MAX_ENV_STEP_IMAGES:-80}' && \
    export CAPTURE_TOKEN_IDS='${CAPTURE_TOKEN_IDS:-1}' && \
    export CAPTURE_DECODE_LANGUAGE='${CAPTURE_DECODE_LANGUAGE:-0}' && \
    export CAPTURE_BATCH_TENSOR_SUMMARY='${CAPTURE_BATCH_TENSOR_SUMMARY:-1}' && \
    export CAPTURE_DENOISE_TRACE='${CAPTURE_DENOISE_TRACE:-1}' && \
    export CAPTURE_MAX_TENSOR_VALUES='${CAPTURE_MAX_TENSOR_VALUES:-64}' && \
    export PI05_PROMPT_FEEDBACK_MODE='${PI05_PROMPT_FEEDBACK_MODE:-off}' && \
    export TASK_IDS='${TASK_IDS:-}' && \
    . .venv/bin/activate && \
    ./cloud/libero/run_pi05_libero.sh '${TASKS}' '${EPISODES}'"

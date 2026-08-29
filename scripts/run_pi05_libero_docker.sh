#!/usr/bin/env bash
set -euo pipefail

TASKS="${1:-libero_spatial}"
EPISODES="${2:-1}"
IMAGE="${IMAGE:-lerobot-libero:latest}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p \
  "${ROOT}/outputs" \
  "${ROOT}/data" \
  "${ROOT}/.libero" \
  "${HOME}/.cache/huggingface"

docker build -t "${IMAGE}" "${ROOT}/cloud/libero"

docker run --rm --gpus all --ipc=host --network=host \
  -e HF_HOME=/workspace/.cache/huggingface \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  -e HF_XET_HIGH_PERFORMANCE=1 \
  -e MUJOCO_GL=egl \
  -e PYOPENGL_PLATFORM=egl \
  -e MUJOCO_EGL_DEVICE_ID=0 \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,video \
  -e RUN_ID="${RUN_ID:-}" \
  -e OUTPUT_ROOT=/workspace/outputs/eval/pi05_libero \
  -e DEVICE="${DEVICE:-cuda}" \
  -e SEED="${SEED:-1000}" \
  -e DTYPE="${DTYPE:-bfloat16}" \
  -e MIN_GPU_MEM_GB="${MIN_GPU_MEM_GB:-0}" \
  -e MIN_HOST_RAM_GB="${MIN_HOST_RAM_GB:-0}" \
  -e LIBERO_CONFIG_PATH=/workspace/.libero \
  -e LIBERO_DATASET_DIR=/workspace/data/libero/datasets \
  -e CAPTURE_ACTIVATIONS="${CAPTURE_ACTIVATIONS:-0}" \
  -e CAPTURE_MAX_CHUNKS="${CAPTURE_MAX_CHUNKS:-80}" \
  -e CAPTURE_LAYER_STRIDE="${CAPTURE_LAYER_STRIDE:-1}" \
  -e CAPTURE_MAX_BINS="${CAPTURE_MAX_BINS:-64}" \
  -e CAPTURE_FAMILIES="${CAPTURE_FAMILIES:-vision,prefix,expert,projection}" \
  -e CAPTURE_PARAM_STATS="${CAPTURE_PARAM_STATS:-0}" \
  -e CAPTURE_FEEDBACK_TRACE="${CAPTURE_FEEDBACK_TRACE:-1}" \
  -e CAPTURE_ENV_STEPS="${CAPTURE_ENV_STEPS:-1}" \
  -e CAPTURE_ENV_STEP_IMAGES="${CAPTURE_ENV_STEP_IMAGES:-0}" \
  -e CAPTURE_ENV_STEP_IMAGE_EVERY_N="${CAPTURE_ENV_STEP_IMAGE_EVERY_N:-10}" \
  -e CAPTURE_MAX_ENV_STEP_IMAGES="${CAPTURE_MAX_ENV_STEP_IMAGES:-80}" \
  -e CAPTURE_TOKEN_IDS="${CAPTURE_TOKEN_IDS:-1}" \
  -e CAPTURE_DECODE_LANGUAGE="${CAPTURE_DECODE_LANGUAGE:-0}" \
  -e CAPTURE_BATCH_TENSOR_SUMMARY="${CAPTURE_BATCH_TENSOR_SUMMARY:-1}" \
  -e CAPTURE_DENOISE_TRACE="${CAPTURE_DENOISE_TRACE:-1}" \
  -e CAPTURE_MAX_TENSOR_VALUES="${CAPTURE_MAX_TENSOR_VALUES:-64}" \
  -e PI05_PROMPT_FEEDBACK_MODE="${PI05_PROMPT_FEEDBACK_MODE:-off}" \
  -e TASK_IDS="${TASK_IDS:-}" \
  -v "${HOME}/.cache/huggingface:/workspace/.cache/huggingface" \
  -v "${ROOT}/outputs:/workspace/outputs" \
  -v "${ROOT}/data:/workspace/data" \
  -v "${ROOT}/.libero:/workspace/.libero" \
  "${IMAGE}" /workspace/run_pi05_libero.sh "${TASKS}" "${EPISODES}"

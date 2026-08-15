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
  -e DEVICE="${DEVICE:-cuda}" \
  -e DTYPE="${DTYPE:-bfloat16}" \
  -e LIBERO_CONFIG_PATH=/workspace/.libero \
  -e LIBERO_DATASET_DIR=/workspace/data/libero/datasets \
  -v "${HOME}/.cache/huggingface:/workspace/.cache/huggingface" \
  -v "${ROOT}/outputs:/workspace/outputs" \
  -v "${ROOT}/data:/workspace/data" \
  -v "${ROOT}/.libero:/workspace/.libero" \
  "${IMAGE}" /workspace/run_pi05_libero.sh "${TASKS}" "${EPISODES}"

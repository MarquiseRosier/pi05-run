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
  --command="mkdir -p ${REMOTE_ROOT}/cloud ${REMOTE_ROOT}/outputs ${REMOTE_ROOT}/data ${REMOTE_ROOT}/.libero ~/.cache/huggingface"

gcloud compute scp \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --recurse cloud/libero "${VM_NAME}:${REMOTE_ROOT}/cloud/"

gcloud compute ssh "${VM_NAME}" \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --command="if ! sudo docker image inspect lerobot-libero:latest >/dev/null 2>&1; then cd ${REMOTE_ROOT}/cloud/libero && sudo docker build -t lerobot-libero:latest .; fi"

gcloud compute ssh "${VM_NAME}" \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --command="cd ${REMOTE_ROOT} && sudo docker run --rm --gpus all --ipc=host --network=host \
    -e HF_HOME=/workspace/.cache/huggingface \
    -e HF_HUB_ENABLE_HF_TRANSFER=1 \
    -e HF_XET_HIGH_PERFORMANCE=1 \
    -e MUJOCO_GL=egl \
    -e PYOPENGL_PLATFORM=egl \
    -e MUJOCO_EGL_DEVICE_ID=0 \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,video \
    -e DEVICE=cuda \
    -e DTYPE=bfloat16 \
    -e MIN_GPU_MEM_GB='${MIN_GPU_MEM_GB:-0}' \
    -e MIN_HOST_RAM_GB='${MIN_HOST_RAM_GB:-0}' \
    -e LIBERO_CONFIG_PATH=/workspace/.libero \
    -e LIBERO_DATASET_DIR=/workspace/data/libero/datasets \
    -e CAPTURE_ACTIVATIONS='${CAPTURE_ACTIVATIONS:-0}' \
    -e CAPTURE_MAX_CHUNKS='${CAPTURE_MAX_CHUNKS:-80}' \
    -e CAPTURE_LAYER_STRIDE='${CAPTURE_LAYER_STRIDE:-1}' \
    -e CAPTURE_MAX_BINS='${CAPTURE_MAX_BINS:-64}' \
    -e CAPTURE_FAMILIES='${CAPTURE_FAMILIES:-vision,prefix,expert,projection}' \
    -e CAPTURE_PARAM_STATS='${CAPTURE_PARAM_STATS:-0}' \
    -e TASK_IDS='${TASK_IDS:-}' \
    -v /home/\$USER/.cache/huggingface:/workspace/.cache/huggingface \
    -v /home/\$USER/groot-run/outputs:/workspace/outputs \
    -v /home/\$USER/groot-run/data:/workspace/data \
    -v /home/\$USER/groot-run/.libero:/workspace/.libero \
    lerobot-libero:latest /workspace/run_pi05_libero.sh '${TASKS}' '${EPISODES}'"

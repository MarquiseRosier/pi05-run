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
  --command="if ! sudo docker image inspect lerobot-libero:latest >/dev/null 2>&1; then cd ${REMOTE_ROOT}/cloud/libero && sudo docker build -t lerobot-libero:latest .; fi"

gcloud compute ssh "${VM_NAME}" \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --command="cd ${REMOTE_ROOT} && sudo docker run --rm --gpus all --ipc=host --network=host \
    -e HF_HOME=/workspace/.cache/huggingface \
    -e HF_HUB_ENABLE_HF_TRANSFER=1 \
    -e MUJOCO_GL=egl \
    -v /home/\$USER/.cache/huggingface:/workspace/.cache/huggingface \
    -v /home/\$USER/groot-run/outputs:/workspace/outputs \
    lerobot-libero:latest /workspace/run_pi05_libero.sh '${TASKS}' '${EPISODES}'"

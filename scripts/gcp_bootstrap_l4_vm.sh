#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-breadwinner-415122}"
ZONE="${ZONE:-us-central1-a}"
VM_NAME="${VM_NAME:-lerobot-libero-l4}"

gcloud compute ssh "${VM_NAME}" \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --command='set -euxo pipefail
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io libnvidia-gl-580-server
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl enable --now docker
sudo systemctl restart docker
sudo usermod -aG docker "$USER"
test -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json
ldconfig -p | grep libEGL_nvidia
sudo docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,video nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi'

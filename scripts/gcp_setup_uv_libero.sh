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
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  build-essential \
  ca-certificates \
  cmake \
  curl \
  ffmpeg \
  git \
  libegl1 \
  libgl1 \
  libglib2.0-0 \
  libglvnd0 \
  libglx0 \
  libopengl0 \
  libosmesa6-dev \
  libsm6 \
  libxext6 \
  libxrender1 \
  pkg-config
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$HOME/groot-run/outputs" "$HOME/.cache/huggingface"
uv venv "$HOME/groot-run/.venv" --python 3.12
uv pip install --python "$HOME/groot-run/.venv/bin/python" --torch-backend cu128 \
  "lerobot[evaluation,libero,pi]" \
  hf-transfer
"$HOME/groot-run/.venv/bin/python" - <<PY
import torch
import lerobot
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("lerobot", getattr(lerobot, "__version__", "unknown"))
PY'

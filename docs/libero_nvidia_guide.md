# Linux NVIDIA Docker Guide

This is the supported teammate workflow for running Pi0.5 on LIBERO.

```text
Linux host + NVIDIA GPU + Docker + NVIDIA Container Toolkit
```

If a collaborator does not have a Linux NVIDIA machine, use the Colab L4 notebook instead: `docs/colab_l4_guide.md`.

The runner uses Docker for the LeRobot/LIBERO runtime and persists results on the host.

## Prerequisites

Check the host GPU:

```bash
nvidia-smi
```

Check Docker GPU passthrough:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

If that command fails, install/fix NVIDIA Container Toolkit before continuing.

Ubuntu setup commands:

```bash
sudo apt-get update
sudo apt-get install -y git python3-pip docker.io ca-certificates curl gnupg2
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
newgrp docker
```

NVIDIA Container Toolkit:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Recommended host:

- Ubuntu 22.04/24.04.
- NVIDIA driver `570+`.
- 50 GB free disk.
- Docker Engine.
- NVIDIA Container Toolkit.

## Hugging Face Setup

The Pi0.5-LIBERO checkpoint also loads gated PaliGemma assets.

1. Accept/request access:

```text
https://huggingface.co/google/paligemma-3b-pt-224
```

2. Log in on the Linux host:

```bash
python3 -m pip install --user -U "huggingface_hub[cli]"
~/.local/bin/hf auth login
```

The token stays in:

```text
~/.cache/huggingface/token
```

The Docker runner mounts `~/.cache/huggingface` at runtime. It does not copy the token into the image.

## Smoke Runs

Build the container:

```bash
docker build -t lerobot-libero:latest cloud/libero
```

Container CUDA sanity check:

```bash
docker run --rm --gpus all --ipc=host --network=host \
  -v "$HOME/.cache/huggingface:/workspace/.cache/huggingface" \
  -v "$PWD/outputs:/workspace/outputs" \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/.libero:/workspace/.libero" \
  lerobot-libero:latest python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

One episode for all 10 `libero_spatial` tasks:

```bash
./scripts/run_pi05_libero_docker.sh libero_spatial 1
```

Only task 0:

```bash
TASK_IDS='[0]' ./scripts/run_pi05_libero_docker.sh libero_spatial 1
```

The run streams progress in the terminal and writes:

```text
outputs/eval/pi05_libero/<timestamp>/
```

## Full Benchmark

```bash
./scripts/run_pi05_libero_docker.sh libero_spatial,libero_object,libero_goal,libero_10 10
```

This runs:

```text
40 tasks x 10 episodes = 400 episodes
```

Expect roughly 2 hours on an L4-class GPU after caches are warm. First run also builds the Docker image and downloads model/assets.

## Activation Capture

Capture forward-pass activation summaries for one task:

```bash
CAPTURE_ACTIVATIONS=1 CAPTURE_MAX_CHUNKS=40 TASK_IDS='[0]' \
  ./scripts/run_pi05_libero_docker.sh libero_spatial 1
```

Also capture static parameter summaries for hooked modules:

```bash
CAPTURE_ACTIVATIONS=1 CAPTURE_PARAM_STATS=1 CAPTURE_MAX_CHUNKS=40 TASK_IDS='[0]' \
  ./scripts/run_pi05_libero_docker.sh libero_spatial 1
```

Capture output:

```text
outputs/eval/pi05_libero/<timestamp>/activation_capture/events.jsonl
outputs/eval/pi05_libero/<timestamp>/activation_capture/images/
```

The capture stores reduced summaries and camera thumbnails, not full hidden-state tensors.

## Inspect Results

Summary:

```bash
./scripts/show_pi05_results.sh
```

Open the result folder:

```bash
./scripts/show_pi05_results.sh latest open-dir
```

Open first rollout video:

```bash
./scripts/show_pi05_results.sh latest open-video
```

Generate activation analysis video:

```bash
python3 -m pip install --user -U opencv-python numpy
./scripts/make_pi05_analysis_video.py --run latest --task-id 0 --preview-frame 30 --open-preview --open
```

Headless server:

```bash
./scripts/make_pi05_analysis_video.py --run latest --task-id 0 --preview-frame 30
```

Then copy/open:

```text
outputs/eval/pi05_libero/<timestamp>/analysis/
```

## Runtime Shape

LeRobot creates a local Pi0.5 policy inside the container:

```text
LIBERO simulator -> camera images + robot state + task text -> Pi0.5 policy -> robot actions -> LIBERO simulator
```

Pi0.5 receives two meaningful camera views on each policy call:

```text
observation.images.image   # agent view
observation.images.image2  # wrist / eye-in-hand view
```

It predicts a `[50, 7]` action chunk. The env executes the first 10 actions, then asks the policy to replan from fresh camera/state observations.

## Mounted Host Paths

```text
~/.cache/huggingface/  Hugging Face token and model cache
./outputs/            logs, metrics, videos, activation traces
./data/               LIBERO data cache
./.libero/            generated LIBERO config
```

These local artifact directories are gitignored.

## Security

- Do not commit Hugging Face tokens.
- Do not put tokens in Docker build args, env vars, scripts, or docs.
- Use `hf auth login` so the token lives in `~/.cache/huggingface/token`.
- If a token was pasted into chat or logs, rotate it in Hugging Face.

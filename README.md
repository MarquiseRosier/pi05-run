# Pi0.5 LIBERO on Linux NVIDIA Docker

This repo runs `lerobot/pi05_libero_finetuned` against LIBERO simulations on a Linux machine with an NVIDIA GPU.

Colab L4/A100 workflow:

- Notebook: `notebooks/pi05_libero_colab_l4.ipynb`
- Guide: `docs/colab_l4_guide.md`
- Open in Colab:

```text
https://colab.research.google.com/github/MarquiseRosier/groot-run/blob/main/notebooks/pi05_libero_colab_l4.ipynb
```

The Colab path runs the same Pi0.5/LIBERO eval natively because Colab does not reliably support NVIDIA Docker. It uses a restricted Google Drive folder for shared model caches, LIBERO assets, outputs, activation traces, and videos. Hugging Face tokens stay in each user's private Colab Secrets and are not committed or copied into Drive.

Use this path when the machine has:

- Linux, preferably Ubuntu 22.04/24.04.
- An NVIDIA GPU with a driver new enough for CUDA 12.8, driver `570+` recommended.
- Docker Engine.
- NVIDIA Container Toolkit.
- At least 50 GB free disk for the Docker image, model cache, LIBERO assets, and outputs.

macOS Docker cannot pass Apple MPS into Linux containers. The supported teammate path is Linux + NVIDIA + Docker.

## 1. Clone

```bash
git clone https://github.com/MarquiseRosier/groot-run.git
cd groot-run
```

## Copy-Paste Ubuntu Setup

If Docker and NVIDIA Container Toolkit are already installed, skip to step 2.

```bash
sudo apt-get update
sudo apt-get install -y git python3-pip docker.io ca-certificates curl gnupg2
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
newgrp docker
```

Install/configure NVIDIA Container Toolkit:

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

## 2. Verify NVIDIA Docker

The host must see the GPU:

```bash
nvidia-smi
```

Docker must also see the GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

If this fails, fix NVIDIA Container Toolkit before running the simulation.

## 2.5. Build And Sanity Check The Container

```bash
docker build -t lerobot-libero:latest cloud/libero
```

Check PyTorch sees CUDA inside the container:

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

## 3. Hugging Face Access

Pi0.5-LIBERO needs:

```text
lerobot/pi05_libero_finetuned
google/paligemma-3b-pt-224
```

Accept/request access for PaliGemma with the same Hugging Face account:

```text
https://huggingface.co/google/paligemma-3b-pt-224
```

Log in on the Linux host:

```bash
python3 -m pip install --user -U "huggingface_hub[cli]"
~/.local/bin/hf auth login
```

The token is stored outside the repo at:

```text
~/.cache/huggingface/token
```

The Docker runner mounts this cache at runtime. It does not bake the token into the image.

## 4. Run A Smoke Test

Run one episode for all 10 `libero_spatial` tasks:

```bash
./scripts/run_pi05_libero_docker.sh libero_spatial 1
```

Run only one task, useful for quick debug:

```bash
TASK_IDS='[0]' ./scripts/run_pi05_libero_docker.sh libero_spatial 1
```

That is the fastest real test. It starts the container, loads Pi0.5, runs LIBERO task 0, writes a video, and exits.

Results are persisted under:

```text
outputs/eval/pi05_libero/<timestamp>/
```

Each run writes `run.log`, `eval_info.json`, and rollout videos.

## 5. Run The Benchmark

Run the four common LIBERO suites:

```bash
./scripts/run_pi05_libero_docker.sh libero_spatial,libero_object,libero_goal,libero_10 10
```

This is:

```text
40 tasks x 10 episodes = 400 episodes
```

On an L4-class GPU, budget roughly 2 hours once model/assets are cached. First run is slower because it builds the image and downloads model/assets.

## 6. Activation Diagnostics

Run one targeted task with PyTorch activation capture:

```bash
CAPTURE_ACTIVATIONS=1 CAPTURE_MAX_CHUNKS=40 TASK_IDS='[0]' \
  ./scripts/run_pi05_libero_docker.sh libero_spatial 1
```

Optional static parameter summaries for hooked modules:

```bash
CAPTURE_ACTIVATIONS=1 CAPTURE_PARAM_STATS=1 CAPTURE_MAX_CHUNKS=40 TASK_IDS='[0]' \
  ./scripts/run_pi05_libero_docker.sh libero_spatial 1
```

Activation files are written next to the run:

```text
outputs/eval/pi05_libero/<timestamp>/activation_capture/events.jsonl
outputs/eval/pi05_libero/<timestamp>/activation_capture/images/
```

## 7. View Results

Summarize latest run:

```bash
./scripts/show_pi05_results.sh
```

Open the first rollout video:

```bash
./scripts/show_pi05_results.sh latest open-video
```

For activation runs, generate the diagnostic video:

```bash
python3 -m pip install --user -U opencv-python numpy
./scripts/make_pi05_analysis_video.py --run latest --task-id 0 --preview-frame 30 --open-preview --open
```

The diagnostic video shows:

- rollout behavior;
- the two camera tensors the model saw;
- signed 50-step action chunks;
- the current executed action inside the first 10 executed actions;
- action-expert activation magnitude by layer and denoise step;
- final-denoise layer x action-token activations.

On a headless server, omit `--open-preview --open` and copy the generated files from:

```text
outputs/eval/pi05_libero/<timestamp>/analysis/
```

## Security

- Do not commit Hugging Face tokens.
- Do not add tokens to Docker `ARG`, Docker `ENV`, scripts, docs, or logs.
- Keep authentication in `~/.cache/huggingface/token`.
- `outputs/`, `data/`, `.libero/`, `.env*`, `.cache/`, and `next_steps.md` are gitignored.

## Common Failures

If Docker cannot see the GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

Fix NVIDIA Container Toolkit before continuing.

If model loading fails on PaliGemma access, accept/request access here and rerun:

```text
https://huggingface.co/google/paligemma-3b-pt-224
```

If the first run is slow, that is expected. It builds the Docker image, downloads Pi0.5/PaliGemma weights, and downloads LIBERO assets.

# Pi0.5 LIBERO NVIDIA Run Guide

This guide is for running `lerobot/pi05_libero_finetuned` against LIBERO simulations with results persisted automatically.

## What Works

- Validated: GCP L4 Spot VM, Ubuntu 24.04, CUDA 12.8 PyTorch, NVIDIA driver 580.
- Expected to work: Linux machine with an NVIDIA GPU, current NVIDIA driver, Docker Engine, and NVIDIA Container Toolkit.
- Useful only for smoke tests: CPU-only Linux. Pi0.5 is large, so full benchmark runs will be very slow.
- Not supported: passing Apple MPS from macOS into a Linux Docker container. Docker Desktop runs Linux containers in a VM; MPS is a macOS Metal backend.

## Runtime Shape

LeRobot does not call a text completion endpoint during LIBERO eval. The evaluator creates a local policy object:

```text
LIBERO simulator -> images/state/task text -> Pi0.5 PyTorch policy -> robot actions -> LIBERO simulator
```

A Mac MPS model server plus a Linux LIBERO client is possible, but it requires a custom remote-policy wrapper. That is not implemented in this repo yet.

## Hugging Face Access

Pi0.5-LIBERO loads `lerobot/pi05_libero_finetuned` and also needs gated Google PaliGemma files. Use the same Hugging Face account for both:

1. Accept/request access at `https://huggingface.co/google/paligemma-3b-pt-224`.
2. Log in on the machine that will run the model:

```bash
hf auth login
```

Tokens are stored in `~/.cache/huggingface/token`, outside this repo.

## Security Checklist

- Do not paste Hugging Face tokens into scripts, Dockerfiles, commit messages, or README files.
- Do not use Docker `ARG` or `ENV` for long-lived tokens.
- Authenticate with `hf auth login` on the runtime machine, or mount an existing `~/.cache/huggingface` directory.
- The Docker runner mounts the Hugging Face cache at runtime; it does not copy the token into the image layer.
- `outputs/`, `data/`, `.libero/`, `.env*`, `.cache/`, and `next_steps.md` are gitignored.
- If a token was pasted into chat or a terminal transcript, rotate it in Hugging Face after the environment is working.

## GCP L4 Path

Create or start the validated L4 Spot VM:

```bash
./scripts/gcp_create_l4_vm.sh
```

Install Docker, NVIDIA Container Toolkit wiring, and host NVIDIA EGL libraries:

```bash
./scripts/gcp_bootstrap_l4_vm.sh
```

Install the direct uv runtime on the VM:

```bash
./scripts/gcp_setup_uv_libero.sh
```

Log in to Hugging Face on the VM:

```bash
gcloud compute ssh lerobot-libero-l4 --zone us-central1-a
hf auth login
exit
```

Run the validated smoke test:

```bash
./scripts/gcp_run_pi05_libero_uv.sh libero_spatial 1
```

This runs one episode for each `libero_spatial` task, so the result has 10 episodes total.

Run the four common LIBERO suites:

```bash
./scripts/gcp_run_pi05_libero_uv.sh libero_spatial,libero_object,libero_goal,libero_10 10
```

That is 40 tasks times 10 episodes per task, or 400 total episodes.

## Streaming Progress

The run command streams progress in your terminal automatically. From another terminal, stream the latest persisted remote log:

```bash
./scripts/gcp_stream_pi05_libero.sh
```

Stream a specific run:

```bash
./scripts/gcp_stream_pi05_libero.sh 20260815-082451
```

Fetch the latest result directory back to your local machine:

```bash
./scripts/gcp_fetch_pi05_results.sh
```

Fetch a specific run:

```bash
./scripts/gcp_fetch_pi05_results.sh 20260815-082451
```

Summarize the latest fetched run locally:

```bash
./scripts/show_pi05_results.sh
```

Open the local result folder:

```bash
./scripts/show_pi05_results.sh latest open-dir
```

Open the first local rollout video:

```bash
./scripts/show_pi05_results.sh latest open-video
```

Run on GCP, fetch the results, and open the local folder when it finishes:

```bash
./scripts/gcp_run_fetch_view_pi05.sh libero_spatial 1
```

This starts the VM if needed, streams the run in your terminal, copies the finished result into local `outputs/eval/pi05_libero/<timestamp>/`, then opens that folder.

Stop the VM automatically after fetching:

```bash
STOP_AFTER=1 ./scripts/gcp_run_fetch_view_pi05.sh libero_spatial 1
```

## Linux NVIDIA Docker Path

On a Linux host with an NVIDIA GPU:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

If the Docker GPU test fails, install or fix NVIDIA Container Toolkit before continuing.

Build and run the LIBERO container:

```bash
./scripts/run_pi05_libero_docker.sh libero_spatial 1
```

For benchmark runs:

```bash
./scripts/run_pi05_libero_docker.sh libero_spatial,libero_object,libero_goal,libero_10 10
```

The Docker runner mounts these host directories:

```text
~/.cache/huggingface/        Hugging Face token and model cache
./outputs/                  logs, metrics, and videos
./data/                     LIBERO data cache
./.libero/                  generated LIBERO config
```

## CPU Notes

CPU-only execution is possible in principle by using:

```bash
DEVICE=cpu DTYPE=float32 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa ./cloud/libero/run_pi05_libero.sh libero_spatial 1
```

Use this only after installing the same Python dependencies on a Linux host. It is not a good path for a full benchmark.

On macOS, Docker plus CPU may run Linux simulation code, but it cannot use MPS. A native macOS Pi0.5 server using MPS would need a custom remote-policy integration.

## Timing

On the validated L4 VM with model/assets already cached:

- `libero_spatial 1`: about 6 minutes wall-clock, including model setup, for 10 total episodes.
- Evaluation loop after setup: about 16 seconds per episode in the successful smoke run.
- Four suites with `10` episodes per task: roughly 1.8 to 2.5 hours after caches are warm.

First run adds dependency install time and model downloads. The Pi0.5 checkpoint cache is about 7 GB.

## Result Persistence

Every run writes under:

```text
outputs/eval/pi05_libero/<timestamp>/
```

The successful smoke test produced:

```text
run.log
videos/libero_spatial_*/eval_episode_0.mp4
```

The `outputs/`, `data/`, `.libero/`, `.env*`, and `next_steps.md` paths are gitignored.

## Stop Costs

Stop the VM when you are done:

```bash
gcloud compute instances stop lerobot-libero-l4 --zone us-central1-a
```

Delete it completely:

```bash
gcloud compute instances delete lerobot-libero-l4 --zone us-central1-a
```

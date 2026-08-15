# LeRobot Simulation Setup

This repo is set up for two simulation paths:

- Apple Silicon local smoke tests with MPS on PushT.
- Pi0.5 LIBERO evaluation on Linux/NVIDIA, validated on a GCP L4 VM.

For the full NVIDIA/LIBERO workflow, use [docs/libero_nvidia_guide.md](docs/libero_nvidia_guide.md).

## One-time setup

Install `uv` and `ffmpeg` if needed:

```bash
brew install uv ffmpeg
```

Create the local environment and install dependencies:

```bash
uv sync
```

Verify MPS:

```bash
uv run python - <<'PY'
import torch
print(torch.__version__)
assert torch.backends.mps.is_available(), "MPS is not available"
print("MPS OK")
PY
```

## Run PushT

Run 5 episodes:

```bash
./scripts/run_pusht_mps.sh
```

Run a different episode count:

```bash
./scripts/run_pusht_mps.sh 10
```

Results are automatically persisted under:

```text
outputs/eval/pusht/<timestamp>/
```

Each run writes `run.log`, `eval_info.json`, and rendered videos when `lerobot-eval` emits them.

## Direct command

The helper script wraps this command:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run lerobot-eval \
  --policy.path=pbelevich/diffusion_pusht \
  --env.type=pusht \
  --eval.batch_size=1 \
  --eval.n_episodes=5 \
  --policy.use_amp=false \
  --policy.device=mps \
  --output_dir=outputs/eval/pusht/manual
```

## GCP LIBERO Pi0.5

Use this for the real `lerobot/pi05_libero_finetuned` benchmark path. The VM is separate from any other running GPU machines.

Create or start the L4 Spot VM:

```bash
./scripts/gcp_create_l4_vm.sh
```

Install Docker and verify GPU containers:

```bash
./scripts/gcp_bootstrap_l4_vm.sh
```

Install the LeRobot LIBERO runtime directly on the VM. This also installs the NVIDIA EGL userspace package needed for headless LIBERO rendering on the validated GCP image:

```bash
./scripts/gcp_setup_uv_libero.sh
```

The Pi0.5 checkpoint uses gated Hugging Face assets. SSH once and log in:

```bash
gcloud compute ssh lerobot-libero-l4 --zone us-central1-a
hf auth login
exit
```

Also accept/request access for the same Hugging Face account here:

```text
https://huggingface.co/google/paligemma-3b-pt-224
```

Without that Google PaliGemma access, Pi0.5-LIBERO loads the fine-tuned weights but fails when it instantiates the tokenizer.

Run one smoke-test episode on LIBERO Spatial:

```bash
./scripts/gcp_run_pi05_libero_uv.sh libero_spatial 1
```

The validated smoke run completed `libero_spatial` with 10 total episodes, `pc_success: 100.0`, and persisted videos under `outputs/eval/pi05_libero/<timestamp>/videos/`.

Run the four benchmark suites:

```bash
./scripts/gcp_run_pi05_libero_uv.sh libero_spatial,libero_object,libero_goal,libero_10 10
```

This is 40 tasks times 10 episodes per task. Budget roughly 2 hours on an L4 after the model and assets are cached.

The foreground run command streams progress directly. From another terminal, stream the latest remote log:

```bash
./scripts/gcp_stream_pi05_libero.sh
```

Fetch the latest remote results locally:

```bash
./scripts/gcp_fetch_pi05_results.sh
```

Summarize the latest fetched run locally:

```bash
./scripts/show_pi05_results.sh
```

Open the local result folder or first rollout video:

```bash
./scripts/show_pi05_results.sh latest open-dir
./scripts/show_pi05_results.sh latest open-video
```

Run a smoke test on GCP, fetch it, and open the local result folder when it finishes:

```bash
./scripts/gcp_run_fetch_view_pi05.sh libero_spatial 1
```

This starts the VM if needed, streams the run, fetches results to local `outputs/eval/pi05_libero/<timestamp>/`, then opens the local folder.

To stop the VM automatically after the results are fetched:

```bash
STOP_AFTER=1 ./scripts/gcp_run_fetch_view_pi05.sh libero_spatial 1
```

Remote results persist on the VM under:

```text
~/groot-run/outputs/eval/pi05_libero/<timestamp>/
```

LIBERO datasets/config are created automatically under:

```text
~/.libero/config.yaml
~/groot-run/data/libero/datasets/
```

The VM user must be in the `render` group for headless EGL rendering. The setup script applies this; reconnect if you just ran it for the first time.

## Linux NVIDIA Docker

On a Linux host with NVIDIA drivers and NVIDIA Container Toolkit:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
hf auth login
./scripts/run_pi05_libero_docker.sh libero_spatial 1
```

The Docker runner mounts Hugging Face cache, `outputs/`, `data/`, and `.libero/` from the host. Tokens are not copied into the image.

macOS Docker cannot pass Apple MPS into Linux containers. For MPS-backed Pi0.5 with Linux LIBERO, the next step is a custom remote-policy server running natively on macOS and serving action predictions to the simulator.

Stop the VM when done:

```bash
gcloud compute instances stop lerobot-libero-l4 --zone us-central1-a
```

Delete it completely:

```bash
gcloud compute instances delete lerobot-libero-l4 --zone us-central1-a
```

## Notes

- The simulator itself may run on CPU/OpenGL; `--policy.device=mps` puts the PyTorch policy on Apple GPU.
- Keep `PYTORCH_ENABLE_MPS_FALLBACK=1` enabled because some PyTorch ops may still fall back to CPU.
- If Hugging Face downloads fail for gated/private models, run `uv run hf auth login`.
- `next_steps.md` is intentionally gitignored for local planning notes.

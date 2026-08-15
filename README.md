# LeRobot Simulation Setup

This repo is set up so everyone can run LeRobot simulations locally on Apple Silicon using MPS.

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

Install the LeRobot LIBERO runtime directly on the VM:

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

Run the four benchmark suites:

```bash
./scripts/gcp_run_pi05_libero_uv.sh libero_spatial,libero_object,libero_goal,libero_10 10
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

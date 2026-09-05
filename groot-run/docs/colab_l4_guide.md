# GR00T LIBERO Colab L4 Guide

This notebook mirrors the Pi0.5 Colab workflow so teammates can compare `nvidia/GR00T-N1.7-LIBERO` and `lerobot/pi05_libero_finetuned` on the same LIBERO tasks.

Use this notebook from the `groot-run` branch:

```text
groot-run/notebooks/groot_libero_colab_l4.ipynb
```

Open from GitHub:

```text
https://colab.research.google.com/github/MarquiseRosier/pi05-run/blob/groot-run/groot-run/notebooks/groot_libero_colab_l4.ipynb
```

Pi0.5 counterpart on `main`:

```text
https://colab.research.google.com/github/MarquiseRosier/pi05-run/blob/main/notebooks/pi05_libero_colab_l4.ipynb
```

Do not put this GR00T flow into the Pi0.5 notebook. The two notebooks stay side by side.

## Hardware

Same bar as Pi0.5 so the comparison is fair:

- GPU: L4 or A100. T4 is rejected.
- About 20 GiB VRAM and 24 GiB system RAM.
- Colab Pro / Pro+ compute units. Free Colab cannot allocate L4.

## Shared Drive Layout

Default Drive root is the same shared folder the Pi0.5 notebook uses. GR00T files are separate so the two caches do not overwrite each other:

```text
DRIVE_ROOT/
  archives/groot_hf_home.tar
  archives/groot_ckpt.tar
  archives/hf_home.tar          # Pi0.5 only; do not reuse for GR00T
  groot_hf_home/
  outputs/eval/groot_libero/<timestamp>/
  outputs/eval/pi05_libero/<timestamp>/
  secrets/HF_TOKEN.txt          # optional, restricted
```

Gated Hugging Face repos:

- [nvidia/GR00T-N1.7-LIBERO](https://huggingface.co/nvidia/GR00T-N1.7-LIBERO)
- [nvidia/Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B)

Preferred path: populate the two GR00T archives once, then run with `HF_OFFLINE=True`.

## Smoke-Test Controls

These names match the Pi0.5 notebook:

```text
SUITE = "libero_spatial"
TASK_IDS = "[0]"
EPISODES = 1
CAPTURE_ACTIVATIONS = True
GENERATE_DIAGNOSTIC_VIDEO = False
HF_OFFLINE = True
N_ACTION_STEPS = 8
```

`SUITE` + `TASK_IDS` are translated through `groot-run/task_map.json`. Spatial task 0 is:

```text
libero_sim/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate
```

That is the same official LIBERO spatial task the Pi0.5 notebook runs as `libero_spatial` / `[0]`.

## What Run All Does

1. Reject T4 / low RAM.
2. Mount Drive and extract `groot_hf_home.tar` / `groot_ckpt.tar`.
3. Install two `uv` environments: GR00T policy server and LIBERO sim client.
4. Start `run_gr00t_server.py`, wait for port 5555, then run `rollout_policy.py`.
5. Write videos and `eval_info.json` under `outputs/eval/groot_libero/<timestamp>/`.
6. If capture is on, write `activation_capture/events.jsonl` and build the HTML report.
7. Optional language intervention probe: same cameras/state, original vs `PROBE_LANGUAGE`, compare each 8-step replan.

Heartbeat lines print elapsed time, videos vs expected episodes, output size, captured chunks, and the latest log line.

## Activation Report

The report UX matches Pi0.5: chunk matrix, family heatmaps, interactive HTML with family / metric / layer / action overlay.

The hooks are not PaliGemma layers. GR00T families are:

- `vision`: backbone vision / visual encoder
- `prefix`: backbone language
- `expert`: DiT action head
- `projection`: action/state encoders and decoders

Each chunk row is one policy call. The sim executes the first 8 actions (`N_ACTION_STEPS`), then replans. Pi0.5 executes 10.

## Language Intervention Probe

The last cell is a counterfactual language probe, not a second eval. At each replan it queries GR00T twice on the same cameras and state: the original LIBERO instruction, then `PROBE_LANGUAGE`. The robot executes the original-language chunk so the scene follows the real task. The table is the 8-step delta.

`PROBE_LANGUAGE` must differ from the official instruction. Artifacts:

```text
outputs/probes/<suite>/task_<id>/compare.md
outputs/probes/<suite>/task_<id>/language_probe.json
outputs/probes/<suite>/task_<id>/render.png
outputs/probes/<suite>/task_<id>/observation_images_image.png
outputs/probes/<suite>/task_<id>/observation_images_image2.png
```

This is not an official success metric.

## First Cache Fill

Checkpoints are not in git. The notebook downloads `nvidia/GR00T-N1.7-LIBERO/libero_10` from Hugging Face when that folder is missing. Pi0.5's `archives/hf_home.tar` is not used.

```text
HF_OFFLINE = False
ALLOW_AUTH_REFRESH = True
```

Add a private Colab Secret `HF_TOKEN`, or `DRIVE_ROOT/secrets/HF_TOKEN.txt`. Accept:

- https://huggingface.co/nvidia/GR00T-N1.7-LIBERO
- https://huggingface.co/nvidia/Cosmos-Reason2-2B

After the first successful run, the notebook can persist `archives/groot_ckpt.tar`. Later runs can use `HF_OFFLINE=True`.

## Local / Linux Runner

The same eval path can run outside Colab:

```bash
export GROOT_PYTHON=/path/to/groot/python
export LIBERO_PYTHON=/path/to/libero_uv/.venv/bin/python
export MODEL_PATH=/path/to/checkpoints/GR00T-N1.7-LIBERO/libero_10
TASK_IDS='[0]' ./groot-run/scripts/run_groot_libero.sh libero_spatial 1
```

Windows teammates should keep using `MI_VLA/Isaac-GR00T/examples/LIBERO/TEAM_RUN_SIM.md`. This Colab path is Linux / Colab only (`MUJOCO_GL=egl`).

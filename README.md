# Pi0.5 LIBERO Colab Notebook

This repo is centered around one shared Google Colab notebook for running and inspecting `lerobot/pi05_libero_finetuned` on LIBERO simulation tasks.

Most collaborators should start here:

- Notebook: `notebooks/pi05_libero_colab_l4.ipynb`
- Transcoder notebook: `notebooks/pi05_libero_transcoder_colab.ipynb`
- Jayden transcoder notebook: `notebooks/pi05_libero_transcoder_colab_jayden.ipynb`
- Matched steering notebook: [`notebooks/pi05_matched_steering.ipynb`](notebooks/pi05_matched_steering.ipynb)
- Transcoder training audit: [`notebooks/pi05_transcoder_training_audit.ipynb`](notebooks/pi05_transcoder_training_audit.ipynb)

The [training-audit Colab](https://colab.research.google.com/github/MarquiseRosier/pi05-run/blob/feature/marquise-transcoder-training-audit/notebooks/pi05_transcoder_training_audit.ipynb)
inspects the existing checkpoint's optimizer evidence, then compares reconstruction on native
denoising trajectories, fresh noise at scheduled times, and fresh noise at sampled training times.
`PHASE="metadata"` needs no GPU. `PHASE="all"` uses a high-RAM L4/A100 for 12 initial scenes
(tasks 0/7/8/9, starts 0/1/2), with same-noise action comparisons for layers 0/8/17 and all layers.
All diagnostic definitions execute in the notebook kernel. Per-case evidence resumes from Drive
under `outputs/transcoder-audit/<RUN_NAME>`; a small evidence ZIP can be exported in the last cell.
The results measure fidelity and sampled occupancy, not semantic attribution or collision prevention.
Regenerate it with `python scripts/embed_pi05_transcoder_audit_notebook.py`.
- Guide: `docs/colab_l4_guide.md`
- Open in Colab:

```text
https://colab.research.google.com/github/MarquiseRosier/pi05-run/blob/main/notebooks/pi05_libero_colab_l4.ipynb
```

Transcoder feature-inspection branch notebook:

```text
https://colab.research.google.com/github/MarquiseRosier/pi05-run/blob/feature/marquise-transcoder-feature-inspection/notebooks/pi05_libero_transcoder_colab_jayden.ipynb
```

The notebook runs Pi0.5/LIBERO eval natively on a Colab L4 or A100 runtime because Colab does not reliably support NVIDIA Docker. It uses a restricted Google Drive folder for shared model caches, LIBERO assets, outputs, activation traces, videos, and prompt probes.

On `feature/marquise-matched-steering`, the [matched steering experiment](https://colab.research.google.com/github/MarquiseRosier/pi05-run/blob/feature/marquise-matched-steering/notebooks/pi05_matched_steering.ipynb)
compares a dense linear-probe direction with selected sparse **transcoder** features using opposite
bowl instructions on identical starting images. It freezes a shared layer on validation data,
then measures actual gripper movement on held-out tasks with matched intervention budgets,
zero/random controls and ablations. The notebook contains the protocol and executable definitions;
experiment work runs in the kernel. `STUDY_SIZE="smoke"` checks infrastructure; the full default
has 71 starting states and approximately 574 single-chunk trials, with per-trial resume.
Each trial now uses a fresh simulator, seeded before construction, while keeping policy weights
loaded. Exact initial states and camera pixels are checked before inference; differences are saved
in `reset_diagnostics.json`. Use the new default `RUN_NAME="matched-steering-reset-002"` for
this reset protocol; older run folders remain intact.
The default held-out case list is committed in [`configs/pi05_matched_steering_test_set.json`](configs/pi05_matched_steering_test_set.json).
Outputs live under `outputs/matched-steering/<RUN_NAME>/`; `report.json` includes the paired
sparse-minus-dense score and task-bootstrap interval. This is a new experiment, not a measured
claim that either representation is more steerable, and the existing checkpoint is not an SAE.

The intended workflow is:

1. Open the Colab notebook.
2. Select an L4 or A100 GPU runtime, preferably high-RAM.
3. Mount the restricted shared Drive folder.
4. Run the setup/cache cells.
5. Run a smoke eval or full benchmark.
6. Inspect rollout videos, chunk matrices, action/layer activation reports, and prompt probe outputs inline.
7. Let the notebook persist logs, videos, activation traces, reports, and cache archives back to Drive.

## Preliminary monitoring study: one collection, four notebooks

Use these latest notebooks on `feature/marquise-monitor-abstract-pilot` for the two-hour
monitoring handoff. They reuse the repaired occlusion pipeline and contain literal experiment
definitions that run in the notebook kernel, with live output and native exceptions.

| Notebook | Run order and purpose |
| --- | --- |
| [A — Collection and primary comparison](https://colab.research.google.com/github/MarquiseRosier/pi05-run/blob/feature/marquise-monitor-abstract-pilot/notebooks/pi05_abstract_a_collection.ipynb) | Start here. One A100/high-RAM L4; collects once, fits the primary comparison, exports all available summaries. |
| [B — Matched dense/transcoder probes](https://colab.research.google.com/github/MarquiseRosier/pi05-run/blob/feature/marquise-monitor-abstract-pilot/notebooks/pi05_abstract_b_matched_probes.ipynb) | CPU; reopen A's saved features and fit/resume the prespecified comparison within its original deadline. |
| [C — Appearance controls](https://colab.research.google.com/github/MarquiseRosier/pi05-run/blob/feature/marquise-monitor-abstract-pilot/notebooks/pi05_abstract_c_appearance_controls.ipynb) | CPU; inspect frozen within-condition and unseen-screen results, exclusions and denominators. |
| [D — Delay and threshold sensitivity](https://colab.research.google.com/github/MarquiseRosier/pi05-run/blob/feature/marquise-monitor-abstract-pilot/notebooks/pi05_abstract_d_delay_sensitivity.ipynb) | CPU; inspect seconds-based delay curves, fixed-monitor 5/10/20 N endpoint sensitivity and collection-path timing. |

Only A needs GPU simulation. A also runs the analyses used by B–D; opening those separately is
optional. Set the same `RUN_PATH` in B–D. The default new run is
`outputs/occlusion-monitor/abstract-layer8-reset-v2-001` under the shared Drive root. Do not reuse
an older pilot's run name. After first-time installation, use **Runtime → Restart session** once;
rerunning a cell does not replace loaded native packages, and deleting the runtime discards the
installation. The saved clock survives a session restart.

The minimal core targets tasks 0/1/2 with four starts per split per task: **156 episodes, 36
starting groups, 12 test groups**. It records only the prespecified action-expert layer 8 by
default: state, matched expert input/output, and frozen transcoder code. All use the same rollouts
and 256-dimensional fixed projections. The 39-episode `feasibility` alternative cannot meet the
reporting safeguards. Neither cohort guarantees enough harmful outcomes. Complete valid groups
from a partial run are explicitly a time-budget-selected exploratory sample.

The protocol includes 50-action chunks, exact seeded paired resets, terminal physics measurements
before any auto-reset, BDDL instance resolution, NumPy-2 segmentation decoding, invalid-control
group exclusion, lossless camera captures every eight steps, bounded image writes and full weight
load checks. Time limits include setup: minute 20 setup gate, minute 90 collection stop, minute 110
analysis cutoff, minute 120 report deadline. Native calls are cooperative, so actual overruns are
recorded. Do not run concurrent collectors on the same GPU.

`evidence/<signature>/evidence_packet.zip` contains counts, source hashes, uncertainty, acceptance
gates, all-arm CSV, timing, plots and an abstract candidate. The notebook downloads this small
packet privately; raw rollouts and fitted models remain in Drive. A strong comparative summary
requires at least 20 good calibration episodes, 20 harmful and 20 good test episodes, and 10
independent test groups; otherwise report descriptive/inconclusive results. These are safeguards,
not a power calculation. The image-encoder baseline is unvalidated and omitted; RGB-grid features
do not support a CV superiority claim. Offline warnings never establish collision prevention.

Regenerate current notebook definitions after source edits:

```bash
python scripts/embed_pi05_monitor_notebook.py
python scripts/embed_pi05_steering_notebook.py
python scripts/embed_pi05_abstract_notebooks.py
```

Hugging Face tokens must not be committed. The normal path is token-free after caches are populated. If a cache refresh is needed, use each user's private Colab Secret named `HF_TOKEN`, or the intentionally restricted shared Drive token file described in `docs/colab_l4_guide.md`.

## Primary Workflow: Colab

Use the notebook for team experiments, benchmark runs, mechanistic interpretability views, and plaintext prompt probes.

Recommended smoke-test controls:

```text
SUITE = "libero_spatial"
TASK_IDS = "[0]"
EPISODES = 1
CAPTURE_ACTIVATIONS = True
CAPTURE_PARAM_STATS = False
CAPTURE_MAX_CHUNKS = 40
GENERATE_DIAGNOSTIC_VIDEO = False
```

Generated Colab outputs include:

- rollout videos from LIBERO;
- chunk matrix PNGs with camera inputs, executed 10-step action chunks, and expert activation deltas;
- action-to-layer correlation plots;
- interactive HTML reports with activation family/layer/metric controls and action overlays;
- prompt probe images and predicted action chunks;
- transcoder layer-flow reports under `outputs/features/pi05_libero/transcoder-probe`;
- persisted logs and summaries under the shared Drive `outputs/` folder.

For the GCP L4 path, `scripts/gcp_run_transcoder_probe.sh` creates
`~/groot-run/outputs/features/pi05_libero/transcoder-probe/`, renders the feature
browser and flowchart, and archives it as `transcoder-probe.tar.gz`.

Read the full notebook workflow here:

```text
docs/colab_l4_guide.md
```

## Secondary Workflow: Linux NVIDIA Docker

Use this only when you have a local or cloud Linux machine with an NVIDIA GPU and want to run outside Colab.

Use this path when the machine has:

- Linux, preferably Ubuntu 22.04/24.04.
- An NVIDIA GPU with a driver new enough for CUDA 12.8, driver `570+` recommended.
- Docker Engine.
- NVIDIA Container Toolkit.
- At least about 20 GiB GPU VRAM for Pi0.5 LIBERO. Use enough host RAM for model loading; 24 GiB+ is recommended, and Colab should use a high-RAM L4/A100 runtime.
- At least 50 GB free disk for the Docker image, model cache, LIBERO assets, and outputs.

macOS Docker cannot pass Apple MPS into Linux containers. The supported teammate path is Linux + NVIDIA + Docker.

### 1. Clone

```bash
git clone https://github.com/MarquiseRosier/pi05-run.git
cd pi05-run
```

### Copy-Paste Ubuntu Setup

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

### 2. Verify NVIDIA Docker

The host must see the GPU:

```bash
nvidia-smi
```

Docker must also see the GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

If this fails, fix NVIDIA Container Toolkit before running the simulation.

### 2.5. Build And Sanity Check The Container

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

### 3. Hugging Face Access

Pi0.5-LIBERO needs:

```text
lerobot/pi05_libero_finetuned
google/paligemma-3b-pt-224
lerobot/libero-assets
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

### 4. Run A Smoke Test

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

### 5. Run The Benchmark

Run the four common LIBERO suites:

```bash
./scripts/run_pi05_libero_docker.sh libero_spatial,libero_object,libero_goal,libero_10 10
```

This is:

```text
40 tasks x 10 episodes = 400 episodes
```

On an L4-class GPU, budget roughly 2 hours once model/assets are cached. First run is slower because it builds the image and downloads model/assets.

If a run exits with code `137` near `Making policy`, the OS killed the process for memory pressure while loading Pi0.5. Use an L4/A100-class GPU and enough host RAM; a T4/low-RAM Colab runtime is not sufficient for this path.

To fail early on a Linux/Docker host instead of waiting for model load:

```bash
MIN_GPU_MEM_GB=20 MIN_HOST_RAM_GB=16 TASK_IDS='[0]' \
  ./scripts/run_pi05_libero_docker.sh libero_spatial 1
```

### 6. Activation Diagnostics

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

### 7. View Results

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

If eval fails with a missing file like `libero_tabletop_base_style.xml`, the LIBERO assets cache is missing. The runner now installs `lerobot/libero-assets` into the local LIBERO package before model load; rerun with network enabled once, then offline runs can reuse the cache.

# Pi0.5 LIBERO Colab L4 Guide

This workflow lets invited collaborators run Pi0.5 LIBERO experiments in Google Colab without configuring a separate Linux machine.

Use the notebook:

```text
notebooks/pi05_libero_colab_l4.ipynb
```

Open from GitHub:

```text
https://colab.research.google.com/github/MarquiseRosier/pi05-run/blob/main/notebooks/pi05_libero_colab_l4.ipynb
```

## Access Model

Keep access restricted in Google Drive/Colab:

- Save one shared notebook copy in Google Drive.
- Share the notebook only with approved Google accounts.
- Current requested account: `programmer908@gmail.com`.
- Share the Drive cache/output folder only with the same approved accounts.
- Do not put Hugging Face tokens in the notebook, repo, scripts, Drive docs, or logs.
- If a cache refresh is needed, each user stores `HF_TOKEN` only in their private Colab Secrets.

This repo cannot enforce Google Drive or Colab sharing permissions. Apply those ACLs in the Google Drive/Colab UI.

## Shared Drive Folder Layout

Default notebook path:

```text
/content/drive/MyDrive/groot-run-shared-programmer908
```

Expected contents:

```text
archives/        Fast Colab cache tar files: hf_home.tar, libero_cache.tar, libero_datasets.tar
hf_home/          Hugging Face home; model snapshots live under hf_home/hub/
libero_cache/     LIBERO runtime cache
libero_datasets/  LIBERO downloaded datasets/init-state artifacts
outputs/          persisted logs, eval summaries, videos, activation traces, probes
notebook_meta/    reserved metadata folder
```

The notebook prefers `archives/*.tar` because Google Drive is very slow when copying a Hugging Face cache as thousands of individual files. It extracts archives to Colab local disk at startup, runs from local disk for speed, then syncs results and refreshed cache archives back to Drive after each experiment.

## One-Time Cache Fill

Use this only if `archives/hf_home.tar` does not already exist and `hf_home/` does not already contain:

```text
hf_home/hub/models--lerobot--pi05_libero_finetuned
hf_home/hub/models--google--paligemma-3b-pt-224
```

Steps:

1. Accept/request Hugging Face access for `google/paligemma-3b-pt-224`.
2. In Colab, open Secrets and add `HF_TOKEN` for your own account.
3. Set notebook controls:

```text
ALLOW_AUTH_REFRESH = True
FORCE_AUTH_REFRESH = False
HF_OFFLINE = False
CACHE_TRANSFER_MODE = "archive"
REQUIRED_GPU = "L4"
```

4. Run the cache/setup cells once.
5. After the cache is populated, return to the normal token-free settings:

```text
ALLOW_AUTH_REFRESH = False
HF_OFFLINE = True
```

The first refresh downloads models to `/content/hf_home` instead of directly to Drive. That is intentional. The notebook later writes one tar archive to `DRIVE_ROOT/archives/hf_home.tar`, which future sessions can extract much faster. If an archive exists and you need to rebuild it, set `FORCE_AUTH_REFRESH=True`.

## Normal Run-All Workflow

1. Open the restricted shared notebook.
2. Runtime -> Change runtime type -> GPU -> L4.
3. Runtime -> Run all.
4. Inspect rollout and diagnostic videos inline.
5. Find persisted outputs in:

```text
DRIVE_ROOT/outputs/eval/pi05_libero/<timestamp>/
```

Useful default smoke-test controls:

```text
SUITE = "libero_spatial"
TASK_IDS = "[0]"
ANALYSIS_TASK_ID = 0
EPISODES = 1
EVAL_PROGRESS_SECONDS = 30
CAPTURE_ACTIVATIONS = True
CAPTURE_PARAM_STATS = False
CAPTURE_MAX_CHUNKS = 40
REPORT_MAX_ROWS = 80
GENERATE_DIAGNOSTIC_VIDEO = False
DISPLAY_INDIVIDUAL_LAYER_GRAPHS = False
LAYER_GRAPH_LIMIT = 6
```

Full benchmark controls:

```text
SUITE = "libero_spatial,libero_object,libero_goal,libero_10"
TASK_IDS = ""
EPISODES = 10
CAPTURE_ACTIVATIONS = False
```

The eval cell streams the model/simulator output and prints a heartbeat every `EVAL_PROGRESS_SECONDS`. The heartbeat includes elapsed time, completed rollout videos versus expected task episodes, output directory size, activation chunks captured, latest eval batch progress, latest rollout step progress, and the last useful log line. If the run fails, the notebook prints `nvidia-smi` plus the last lines of both `colab_launcher.log` and `run.log` before raising the error.

## Inline Investigation Outputs

The notebook display cell runs automatically after eval and shows:

- Rollout video from the simulator.
- Optional four-panel diagnostic video with rollout, camera tensors, signed action chunk, and expert activation heatmaps.
- Chunk matrix PNG: one row per policy call / 10-action execution window.
- Family heatmaps for `vision`, `prefix` language/task, and `expert` action layers over chunks.
- Interactive HTML report with dropdowns for activation family, metric, and layer.
- Standalone PNGs for each expert transformer layer showing activation over policy chunks.

The chunk matrix row is the first granular view to inspect. Each row contains:

```text
chunk/task | simulator third-person frame | input image | input image2 | first 10 actions | expert layer mean | expert layer x denoise
```

Generated report artifacts are persisted under:

```text
outputs/eval/pi05_libero/<timestamp>/analysis/task_<id>_episode_0_colab_report/
```

The main file to open/share is:

```text
task_<id>_episode_0_interactive.html
```

The script that builds this is:

```bash
python scripts/make_pi05_colab_report.py --run latest --task-id 0 --episode 0 --max-rows 80
```

You normally do not need to run that command manually in Colab; the notebook runs it when `CAPTURE_ACTIVATIONS=True`.

For speed, leave `GENERATE_DIAGNOSTIC_VIDEO=False`. Turn it on only when you want the MP4.

## Plaintext Prompt Probe

The final notebook cell is for mechanistic interpretability, not official benchmark scoring.

It resets a chosen LIBERO task, captures the starting camera/state observation, replaces the language prompt with `PROBE_LANGUAGE`, and asks Pi0.5 for one `[50, 7]` action chunk.

Saved probe artifacts:

```text
outputs/probes/<suite>/task_<id>/render.png
outputs/probes/<suite>/task_<id>/observation_images_image.png
outputs/probes/<suite>/task_<id>/observation_images_image2.png
outputs/probes/<suite>/task_<id>/action_chunk.json
```

Low-level commands such as `move +x` or `close gripper` may be out of distribution for the LIBERO-tuned checkpoint. Use the probe to compare inputs, action chunks, and activation traces, not as a formal success metric.

## If The Cache Cell Looks Stuck

Stop the old run and use the latest notebook from `main`. The previous version copied `DRIVE_ROOT/hf_home` file-by-file, which can look frozen on Colab.

If you see this older error:

```text
CalledProcessError: ... pip install -q -U huggingface_hub[hf_transfer]
```

Use the latest notebook. That extra install was removed because the setup cell already installs `huggingface_hub` through LeRobot and installs `hf-transfer` directly.

If you see this older error:

```text
CalledProcessError: ... python3 -m pip install -q -U uv
```

Use the latest notebook. Colab can block system `pip install` on Python 3.12, so the setup cell now installs the standalone `uv` binary under `/content/uv-bin` instead of modifying system Python packages.

If you see this older error:

```text
CalledProcessError: ... /usr/local/bin/uv venv /content/lerobot-venv --python /usr/bin/python3
```

Use the latest notebook. Colab can ship its own `/usr/local/bin/uv`; the setup cell now ignores that binary, installs official `uv` into `/content/uv-bin`, and recreates `/content/lerobot-venv` with `--clear`.

Use these controls for the first successful cache fill:

```text
CACHE_TRANSFER_MODE = "archive"
ALLOW_AUTH_REFRESH = True
FORCE_AUTH_REFRESH = False
HF_OFFLINE = False
```

After the archive exists:

```text
CACHE_TRANSFER_MODE = "archive"
ALLOW_AUTH_REFRESH = False
FORCE_AUTH_REFRESH = False
HF_OFFLINE = True
```

The fixed cache cell prints section headers, start/finish timestamps, elapsed time, `pv` transfer progress for tar archives, `rsync --info=progress2` for Drive folder copies, and `du -sh` summaries. Hugging Face refreshes also print one line per repo with estimated remote size plus a 15-second heartbeat showing cache growth, recent rate, average rate, elapsed time, and ETA.

## Security Checklist

- No Hugging Face token is committed to this repo.
- The notebook default is `HF_OFFLINE=True`.
- The shared Drive folder stores model caches and outputs, not credentials.
- Authenticated refresh uses only each user's private Colab Secret named `HF_TOKEN`.
- Output directories may contain experiment data and videos; keep the Drive folder restricted.

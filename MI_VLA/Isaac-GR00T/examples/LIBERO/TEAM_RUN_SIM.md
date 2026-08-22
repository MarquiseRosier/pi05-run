# Team Guide: Run the LIBERO Simulator from Scratch (Windows)

This repo has been verified on **Windows + NVIDIA GPU + conda `groot`**. The stack is two processes:

```
Terminal 1 (conda groot, uses GPU)     Terminal 2 (libero_uv venv)
run_gr00t_server.py  ──────────────►   rollout_policy.py
Loads weights, listens on 5555         MuJoCo / LIBERO sim client
```

The official Linux / `uv` flow is in [README.md](README.md) in this folder. The steps below are the path this team uses to reproduce the setup from zero.

---

## 0. Machine requirements

- Windows 10/11, NVIDIA GPU, **16GB+ VRAM** recommended (RTX 4090 or similar)
- [Anaconda](https://www.anaconda.com/download) or Miniconda
- About **20GB** free disk (weights are ~6.5GB; the VLM backbone downloads extra Cosmo weights on first load)
- A Hugging Face account

---

## 1. Clone the repo

```powershell
git clone --recurse-submodules <repo-url> MI_VLA
cd MI_VLA
```

If you already cloned without submodules:

```powershell
git submodule update --init Isaac-GR00T/external_dependencies/LIBERO
```

All later commands assume:

```powershell
cd Isaac-GR00T
```

---

## 2. Get Hugging Face access (do this first)

Both models are **gated**. Opening the page is not enough — you must request access while logged in, then use a token on this machine.

1. Create / log in to a [Hugging Face](https://huggingface.co/join) account in the browser.
2. Open each model page and click **Agree and access repository** (or **Request access**), then submit the license form:
   - [nvidia/GR00T-N1.7-LIBERO](https://huggingface.co/nvidia/GR00T-N1.7-LIBERO) — LIBERO finetuned weights
   - [nvidia/Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B) — gated VLM backbone; **every checkpoint loads this on first use**
3. Wait until the page says you have access (NVIDIA license accept is usually instant; if it says “pending”, wait for approval).
4. Create a token: [Hugging Face → Settings → Access Tokens](https://huggingface.co/settings/tokens) → **New token** → permission **Read** → copy it.
5. After the `groot` env exists (next section), log in on this machine:

```powershell
conda activate groot
huggingface-cli login
```

Paste the token when prompted. Or set an env var (do not commit it):

```powershell
$env:HF_TOKEN = "<your-token>"
```

Without both page approvals **and** a local login, download / server start fails with `GatedRepoError` / `401`.

---

## 3. Install the inference env (conda `groot`, once)

Must be **Python 3.12** (some wheels are missing on 3.13).

```powershell
conda create -n groot python=3.12 -y
conda activate groot
conda install -c conda-forge "ffmpeg<8" -y
pip install -e .
```

`torchcodec` only supports FFmpeg 4–7. Missing `flash-attn` on Windows is expected; the server falls back to `sdpa`.

Sanity check:

```powershell
python -c "import gr00t, torch; print('gr00t ok, cuda=', torch.cuda.is_available())"
```

This should print `cuda= True`.

---

## 4. Download the LIBERO weights

Hugging Face **does not** accept a nested path like `nvidia/GR00T-N1.7-LIBERO/libero_10` as `--model-path`. Download locally first.

```powershell
conda activate groot
hf download nvidia/GR00T-N1.7-LIBERO `
  --include "libero_10/config.json" `
  --include "libero_10/embodiment_id.json" `
  --include "libero_10/model-*.safetensors" `
  --include "libero_10/model.safetensors.index.json" `
  --include "libero_10/processor_config.json" `
  --include "libero_10/statistics.json" `
  --local-dir checkpoints/GR00T-N1.7-LIBERO
```

You should end up with these files (the two `.safetensors` shards total about **6.5GB**):

```
checkpoints/GR00T-N1.7-LIBERO/libero_10/
  config.json
  embodiment_id.json
  model-00001-of-00002.safetensors
  model-00002-of-00002.safetensors
  model.safetensors.index.json
  processor_config.json
  statistics.json
```

These weights are already in `.gitignore`. **Do not push them to GitHub.**

---

## 5. Install the LIBERO sim env (once)

The sim client is a separate venv. Do not mix it with `groot`.

Activate `groot` first (the script uses its Python 3.12 to create the venv), then:

```powershell
powershell -ExecutionPolicy Bypass -File gr00t/eval/sim/LIBERO/setup_libero.ps1
```

Success looks like `Env OK` and `Setup done.`

Sim Python path:

```
gr00t/eval/sim/LIBERO/libero_uv/.venv/Scripts/python.exe
```

`.venv` is ignored. Do not commit it.

---

## 6. Run with two terminals

### Terminal 1: Policy server (it is normal for this window to stay occupied)

```powershell
cd Isaac-GR00T
conda activate groot
python gr00t/eval/run_gr00t_server.py `
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 `
  --embodiment-tag LIBERO_PANDA `
  --use-sim-policy-wrapper
```

Wait until:

```
Server is ready and listening on tcp://0.0.0.0:5555
```

The first launch also downloads Cosmos-Reason2-2B and can take several minutes. Ignore `flash_attn is not installed`.

Stop the server with `Ctrl+C` in this window. If that does nothing, close the terminal.

### Terminal 2: Sim client

On Windows, pass `--video-dir` explicitly (the default is the Linux path `/tmp/...`). Start with `1` parallel env:

```powershell
cd Isaac-GR00T
$py = "gr00t/eval/sim/LIBERO/libero_uv/.venv/Scripts/python.exe"
$env:MUJOCO_GL = "wgl"
& $py gr00t/eval/rollout_policy.py `
  --n-episodes 1 `
  --n-envs 1 `
  --policy-client-host 127.0.0.1 `
  --policy-client-port 5555 `
  --max-episode-steps 720 `
  --n-action-steps 8 `
  --env-name libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it `
  --video-dir tmp_sim_videos
```

When it finishes you will see `success rate`. Videos land in `Isaac-GR00T/tmp_sim_videos/`.

---

## 7. Switch tasks

Change `--env-name` to any task below. The full list is at the end of [README.md](README.md).

Good starter tasks:

```
libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
libero_sim/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate
libero_sim/turn_on_the_stove
```

You do **not** need to restart the server. Only change the client's `--env-name`.

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `GatedRepoError` / `401` | Accept both HF models, then `huggingface-cli login` |
| `Address already in use` | Port 5555 is taken by an old server. Close that terminal, or use `--port 5556` (change the client too) |
| Server terminal never returns a prompt | Expected; it is listening in the foreground. `Ctrl+C` or close the window |
| `Could not load libtorchcodec` / FFmpeg | After `conda activate groot`, run `conda install -c conda-forge "ffmpeg<8" -y` |
| Sim errors about EGL / OpenGL | Windows needs `MUJOCO_GL=wgl`. Re-run `setup_libero.ps1` |
| Video write to `/tmp/...` fails | Add `--video-dir tmp_sim_videos` on the client |
| Skip video for now | `--video-dir none` |
| Large `n-envs` hangs or crashes | On Windows start with `--n-envs 1` |
| Push rejected because files are too large | `checkpoints/` and `*.safetensors` are gitignored; do not `git add` them |

---

## 9. Do not commit these

- `Isaac-GR00T/checkpoints/` (weights)
- `Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv/` (sim env)
- `Isaac-GR00T/tmp_sim_videos/` (eval videos; optional to ignore)

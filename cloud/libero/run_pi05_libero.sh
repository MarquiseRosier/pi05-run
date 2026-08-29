#!/usr/bin/env bash
set -euo pipefail

TASKS="${1:-libero_spatial}"
EPISODES="${2:-1}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/eval/pi05_libero}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_ID}"
POLICY_PATH="${POLICY_PATH:-lerobot/pi05_libero_finetuned}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-1000}"
MIN_GPU_MEM_GB="${MIN_GPU_MEM_GB:-0}"
MIN_HOST_RAM_GB="${MIN_HOST_RAM_GB:-0}"
if [[ -z "${DTYPE:-}" ]]; then
  if [[ "${DEVICE}" == "cpu" ]]; then
    DTYPE="float32"
  else
    DTYPE="bfloat16"
  fi
fi
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_PARALLEL_TASKS="${MAX_PARALLEL_TASKS:-1}"
USE_ASYNC_ENVS="${USE_ASYNC_ENVS:-false}"
TASK_IDS="${TASK_IDS:-}"
CAPTURE_ACTIVATIONS="${CAPTURE_ACTIVATIONS:-0}"

mkdir -p "${OUTPUT_DIR}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export MPLBACKEND="Agg"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

host_ram_kb="$(awk '/MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null || printf '0')"
host_ram_gb=$(( (host_ram_kb + 1024 * 1024 - 1) / (1024 * 1024) ))
printf "host_ram_gib %s\n" "${host_ram_gb}"
if (( MIN_HOST_RAM_GB > 0 && host_ram_gb < MIN_HOST_RAM_GB )); then
  cat >&2 <<EOF
ERROR: Host RAM is too small for Pi0.5 LIBERO.
Detected: ~${host_ram_gb} GiB
Required: >= ${MIN_HOST_RAM_GB} GiB

In Colab, switch to a high-RAM L4 or A100 runtime. T4/low-RAM runtimes
commonly exit 137 while loading the Pi0.5 policy.
EOF
  exit 2
fi

if [[ "${DEVICE}" == cuda* ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: DEVICE=${DEVICE}, but nvidia-smi is not available." >&2
    exit 2
  fi
  gpu_csv="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -n1)"
  gpu_name="$(printf '%s' "${gpu_csv%,*}" | sed 's/^ *//;s/ *$//')"
  gpu_mem_mb="$(printf '%s' "${gpu_csv##*,}" | sed 's/^ *//;s/ *$//')"
  gpu_mem_gb=$(( (gpu_mem_mb + 1023) / 1024 ))
  printf "gpu_info %s | vram_gib %s\n" "${gpu_name}" "${gpu_mem_gb}"
  if (( MIN_GPU_MEM_GB > 0 && gpu_mem_gb < MIN_GPU_MEM_GB )); then
    cat >&2 <<EOF
ERROR: GPU VRAM is too small for Pi0.5 LIBERO.
Detected: ${gpu_name} with ~${gpu_mem_gb} GiB
Required: >= ${MIN_GPU_MEM_GB} GiB

Use an L4/A100-class GPU for Colab. T4 usually dies with exit 137 during
policy loading before LIBERO rollout begins.
EOF
    exit 2
  fi
fi

if [[ "${CAPTURE_ACTIVATIONS}" == "1" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  export LEROBOT_CAPTURE_ACTIVATIONS=1
  export LEROBOT_CAPTURE_DIR="${LEROBOT_CAPTURE_DIR:-${OUTPUT_DIR}/activation_capture}"
  export CAPTURE_MAX_CHUNKS="${CAPTURE_MAX_CHUNKS:-80}"
  export CAPTURE_LAYER_STRIDE="${CAPTURE_LAYER_STRIDE:-1}"
  export CAPTURE_MAX_BINS="${CAPTURE_MAX_BINS:-64}"
  export CAPTURE_FAMILIES="${CAPTURE_FAMILIES:-vision,prefix,expert,projection}"
  export CAPTURE_PARAM_STATS="${CAPTURE_PARAM_STATS:-0}"
  export CAPTURE_FEEDBACK_TRACE="${CAPTURE_FEEDBACK_TRACE:-1}"
  export CAPTURE_ENV_STEPS="${CAPTURE_ENV_STEPS:-1}"
  export CAPTURE_ENV_STEP_IMAGES="${CAPTURE_ENV_STEP_IMAGES:-0}"
  export CAPTURE_ENV_STEP_IMAGE_EVERY_N="${CAPTURE_ENV_STEP_IMAGE_EVERY_N:-10}"
  export CAPTURE_MAX_ENV_STEP_IMAGES="${CAPTURE_MAX_ENV_STEP_IMAGES:-80}"
  export CAPTURE_TOKEN_IDS="${CAPTURE_TOKEN_IDS:-1}"
  export CAPTURE_DECODE_LANGUAGE="${CAPTURE_DECODE_LANGUAGE:-0}"
  export CAPTURE_BATCH_TENSOR_SUMMARY="${CAPTURE_BATCH_TENSOR_SUMMARY:-1}"
  export CAPTURE_DENOISE_TRACE="${CAPTURE_DENOISE_TRACE:-1}"
  export CAPTURE_MAX_TENSOR_VALUES="${CAPTURE_MAX_TENSOR_VALUES:-64}"
  export PI05_PROMPT_FEEDBACK_MODE="${PI05_PROMPT_FEEDBACK_MODE:-off}"
  CAPTURE_PYTHONPATH="${SCRIPT_DIR}/activation_capture${PYTHONPATH:+:${PYTHONPATH}}"
fi

PYTHON="${PYTHON:-python}"
LIBERO_CONFIG_DIR="${LIBERO_CONFIG_PATH:-${HOME}/.libero}"
LIBERO_DATASET_DIR="${LIBERO_DATASET_DIR:-${HOME}/groot-run/data/libero/datasets}"
LIBERO_ROOT="$("${PYTHON}" - <<'PY'
from pathlib import Path
import site

roots = []
for path in site.getsitepackages():
    roots.append(Path(path))
user_site = site.getusersitepackages()
if user_site:
    roots.append(Path(user_site))

for root in roots:
    candidate = root / "libero" / "libero"
    if candidate.exists():
        print(candidate)
        break
else:
    raise SystemExit("Could not find installed libero package path")
PY
)"

mkdir -p "${LIBERO_CONFIG_DIR}" "${LIBERO_DATASET_DIR}"
if [[ ! -f "${LIBERO_CONFIG_DIR}/config.yaml" ]]; then
  cat > "${LIBERO_CONFIG_DIR}/config.yaml" <<YAML
assets: ${LIBERO_ROOT}/assets
bddl_files: ${LIBERO_ROOT}/bddl_files
benchmark_root: ${LIBERO_ROOT}
datasets: ${LIBERO_DATASET_DIR}
init_states: ${LIBERO_ROOT}/init_files
YAML
fi

"${PYTHON}" - <<PY
from pathlib import Path
import os
import shutil
import sys

from huggingface_hub import snapshot_download

libero_root = Path("${LIBERO_ROOT}")
assets_dir = libero_root / "assets"
required = assets_dir / "scenes" / "libero_tabletop_base_style.xml"
offline = os.environ.get("HF_HUB_OFFLINE") == "1"
cache_dir = os.environ.get("HF_HUB_CACHE") or str(Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub")


def copy_snapshot(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        if child.name == ".gitattributes":
            continue
        target = dst / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            shutil.copy2(child, target)


if required.exists():
    print(f"libero_assets OK {required}", flush=True)
else:
    print("LIBERO assets not installed in package; preparing lerobot/libero-assets...", flush=True)
    try:
        snapshot = Path(
            snapshot_download(
                repo_id="lerobot/libero-assets",
                repo_type="dataset",
                cache_dir=cache_dir,
                local_files_only=offline,
            )
        )
    except Exception as exc:
        mode = "offline" if offline else "online"
        raise SystemExit(
            "Missing LIBERO assets. The simulator needs Hugging Face dataset "
            "lerobot/libero-assets. Current mode: "
            f"{mode}. If running offline, refresh the cache once with HF_OFFLINE=False "
            "or run the Colab asset hotfix cell."
        ) from exc
    print(f"Installing LIBERO assets: {snapshot} -> {assets_dir}", flush=True)
    copy_snapshot(snapshot, assets_dir)
    if not required.exists():
        raise SystemExit(f"LIBERO asset install failed; still missing {required}")
    print(f"libero_assets OK {required}", flush=True)
PY

"${PYTHON}" - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device", torch.cuda.get_device_name(0))
PY

"${PYTHON}" - <<'PY'
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

checks = [
    ("lerobot/pi05_libero_finetuned", "config.json"),
    ("google/paligemma-3b-pt-224", "config.json"),
]

for repo_id, filename in checks:
    try:
        hf_hub_download(repo_id=repo_id, filename=filename)
    except GatedRepoError as exc:
        raise SystemExit(
            f"Missing Hugging Face gated access for {repo_id}. "
            f"Open https://huggingface.co/{repo_id}, accept/request access with the same HF account, then rerun."
        ) from exc
    except HfHubHTTPError as exc:
        raise SystemExit(f"Could not access {repo_id}/{filename}: {exc}") from exc
print("hf_access OK")
PY

args=(
  lerobot-eval
  --policy.path="${POLICY_PATH}" \
  --policy.device="${DEVICE}" \
  --policy.dtype="${DTYPE}" \
  --policy.compile_model=false \
  --policy.gradient_checkpointing=false \
  --policy.n_action_steps=10 \
  --env.type=libero \
  --env.task="${TASKS}" \
  --env.max_parallel_tasks="${MAX_PARALLEL_TASKS}" \
  --eval.batch_size="${BATCH_SIZE}" \
  --eval.n_episodes="${EPISODES}" \
  --eval.use_async_envs="${USE_ASYNC_ENVS}" \
  --seed="${SEED}" \
  --output_dir="${OUTPUT_DIR}"
)

if [[ -n "${TASK_IDS}" ]]; then
  args+=(--env.task_ids="${TASK_IDS}")
fi

if [[ "${CAPTURE_ACTIVATIONS}" == "1" ]]; then
  PYTHONPATH="${CAPTURE_PYTHONPATH}" "${args[@]}" 2>&1 | tee "${OUTPUT_DIR}/run.log"
else
  "${args[@]}" 2>&1 | tee "${OUTPUT_DIR}/run.log"
fi

printf "\nSaved results in %s\n" "${OUTPUT_DIR}"

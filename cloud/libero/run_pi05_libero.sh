#!/usr/bin/env bash
set -euo pipefail

TASKS="${1:-libero_spatial}"
EPISODES="${2:-1}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/eval/pi05_libero}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_ID}"
POLICY_PATH="${POLICY_PATH:-lerobot/pi05_libero_finetuned}"
DEVICE="${DEVICE:-cuda}"
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

mkdir -p "${OUTPUT_DIR}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

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

lerobot-eval \
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
  --output_dir="${OUTPUT_DIR}" \
  2>&1 | tee "${OUTPUT_DIR}/run.log"

printf "\nSaved results in %s\n" "${OUTPUT_DIR}"

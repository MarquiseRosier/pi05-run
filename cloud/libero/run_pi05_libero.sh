#!/usr/bin/env bash
set -euo pipefail

TASKS="${1:-libero_spatial}"
EPISODES="${2:-1}"
RUN_ID="$(date -u +%Y%m%d-%H%M%S)"
OUTPUT_DIR="outputs/eval/pi05_libero/${RUN_ID}"

mkdir -p "${OUTPUT_DIR}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device", torch.cuda.get_device_name(0))
PY

lerobot-eval \
  --policy.path=lerobot/pi05_libero_finetuned \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.n_action_steps=10 \
  --env.type=libero \
  --env.task="${TASKS}" \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes="${EPISODES}" \
  --eval.use_async_envs=false \
  --output_dir="${OUTPUT_DIR}" \
  2>&1 | tee "${OUTPUT_DIR}/run.log"

printf "\nSaved results in %s\n" "${OUTPUT_DIR}"

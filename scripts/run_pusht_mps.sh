#!/usr/bin/env bash
set -euo pipefail

EPISODES="${1:-5}"
BATCH_SIZE="${2:-1}"
MODEL_ID="${MODEL_ID:-pbelevich/diffusion_pusht}"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="outputs/eval/pusht/${RUN_ID}"

mkdir -p "${OUTPUT_DIR}"

export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

uv run lerobot-eval \
  --policy.path="${MODEL_ID}" \
  --env.type=pusht \
  --eval.batch_size="${BATCH_SIZE}" \
  --eval.n_episodes="${EPISODES}" \
  --policy.use_amp=false \
  --policy.device=mps \
  --output_dir="${OUTPUT_DIR}" \
  2>&1 | tee "${OUTPUT_DIR}/run.log"

printf "\nSaved results in %s\n" "${OUTPUT_DIR}"

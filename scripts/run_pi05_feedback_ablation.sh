#!/usr/bin/env bash
set -euo pipefail

TASKS="${1:-libero_spatial}"
EPISODES="${2:-1}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/outputs/eval/pi05_libero}"
PI05_EVAL_RUNNER="${PI05_EVAL_RUNNER:-${ROOT}/cloud/libero/run_pi05_libero.sh}"
PI05_ABLATION_ID="${PI05_ABLATION_ID:-$(date -u +%Y%m%d-%H%M%S)-feedback}"
PI05_ABLATION_MODES="${PI05_ABLATION_MODES:-off,last_action,chunk_summary}"
SEED="${SEED:-1000}"
FEEDBACK_ABLATION_MAX_STEPS="${FEEDBACK_ABLATION_MAX_STEPS:-120}"

CAPTURE_ACTIVATIONS="${CAPTURE_ACTIVATIONS:-1}"
CAPTURE_FEEDBACK_TRACE="${CAPTURE_FEEDBACK_TRACE:-1}"
CAPTURE_ENV_STEPS="${CAPTURE_ENV_STEPS:-1}"
CAPTURE_ENV_STEP_IMAGES="${CAPTURE_ENV_STEP_IMAGES:-0}"
CAPTURE_DENOISE_TRACE="${CAPTURE_DENOISE_TRACE:-0}"
CAPTURE_MAX_CHUNKS="${CAPTURE_MAX_CHUNKS:-400}"

mkdir -p "${OUTPUT_ROOT}"

IFS=',' read -r -a modes <<< "${PI05_ABLATION_MODES}"
if [[ "${#modes[@]}" -eq 0 ]]; then
  echo "No feedback modes provided in PI05_ABLATION_MODES." >&2
  exit 2
fi

printf "Pi0.5 feedback ablation id: %s\n" "${PI05_ABLATION_ID}"
printf "tasks=%s episodes=%s task_ids=%s seed=%s modes=%s\n" \
  "${TASKS}" "${EPISODES}" "${TASK_IDS:-}" "${SEED}" "${PI05_ABLATION_MODES}"

index=0
for raw_mode in "${modes[@]}"; do
  mode="$(printf '%s' "${raw_mode}" | xargs)"
  if [[ -z "${mode}" ]]; then
    continue
  fi
  safe_mode="$(printf '%s' "${mode}" | tr -cs 'A-Za-z0-9_.-' '_' | sed 's/^_*//;s/_*$//')"
  run_id="$(printf '%s-%02d-%s' "${PI05_ABLATION_ID}" "${index}" "${safe_mode}")"
  run_dir="${OUTPUT_ROOT}/${run_id}"
  if [[ -e "${run_dir}" ]]; then
    suffix=1
    while [[ -e "${run_dir}-${suffix}" ]]; do
      suffix=$((suffix + 1))
    done
    run_id="${run_id}-${suffix}"
    run_dir="${OUTPUT_ROOT}/${run_id}"
  fi
  mkdir -p "${run_dir}"
  cat > "${run_dir}/feedback_ablation_run.json" <<JSON
{
  "ablation_id": "${PI05_ABLATION_ID}",
  "mode": "${mode}",
  "tasks": "${TASKS}",
  "episodes": ${EPISODES},
  "task_ids": "${TASK_IDS:-}",
  "seed": ${SEED}
}
JSON

  printf "\n=== feedback mode: %s | run: %s ===\n" "${mode}" "${run_id}"
  CAPTURE_ACTIVATIONS="${CAPTURE_ACTIVATIONS}" \
  CAPTURE_FEEDBACK_TRACE="${CAPTURE_FEEDBACK_TRACE}" \
  CAPTURE_ENV_STEPS="${CAPTURE_ENV_STEPS}" \
  CAPTURE_ENV_STEP_IMAGES="${CAPTURE_ENV_STEP_IMAGES}" \
  CAPTURE_DENOISE_TRACE="${CAPTURE_DENOISE_TRACE}" \
  CAPTURE_MAX_CHUNKS="${CAPTURE_MAX_CHUNKS}" \
  PI05_PROMPT_FEEDBACK_MODE="${mode}" \
  RUN_ID="${run_id}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  SEED="${SEED}" \
  "${PI05_EVAL_RUNNER}" "${TASKS}" "${EPISODES}"

  "${ROOT}/scripts/summarize_pi05_feedback_trace.py" \
    --run "${run_dir}" \
    --max-steps "${FEEDBACK_ABLATION_MAX_STEPS}" || true

  index=$((index + 1))
done

"${ROOT}/scripts/compare_pi05_feedback_ablation.py" \
  --ablation-id "${PI05_ABLATION_ID}" \
  --output-root "${OUTPUT_ROOT}"

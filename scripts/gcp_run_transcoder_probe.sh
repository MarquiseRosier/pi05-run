#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PROJECT="${PROJECT:-breadwinner-415122}"
ZONE="${ZONE:-us-central1-a}"
VM_NAME="${VM_NAME:-lerobot-libero-l4}"
TRANSCODER_PROBE_NAME="${TRANSCODER_PROBE_NAME:-transcoder-probe}"
REMOTE_FEATURE_REL="outputs/features/pi05_libero/${TRANSCODER_PROBE_NAME}"
REMOTE_ARCHIVE_REL="${TRANSCODER_PROBE_NAME}/${TRANSCODER_PROBE_NAME}.tar.gz"
LOCAL_ARTIFACT_ROOT="${LOCAL_ARTIFACT_ROOT:-${ROOT}/outputs/${TRANSCODER_PROBE_NAME}}"

POLICY_PATH="${POLICY_PATH:-lerobot/pi05_libero_finetuned}"
FEATURE_PROBE_EPISODES="${FEATURE_PROBE_EPISODES:-0,1,2,3,4}"
FEATURE_PROBE_BATCH_SIZE="${FEATURE_PROBE_BATCH_SIZE:-4}"
FEATURE_PROBE_NUM_WORKERS="${FEATURE_PROBE_NUM_WORKERS:-0}"
FEATURE_PROBE_COLLECTION_MODE="${FEATURE_PROBE_COLLECTION_MODE:-inference}"
FEATURE_PROBE_NUM_INFERENCE_STEPS="${FEATURE_PROBE_NUM_INFERENCE_STEPS:-10}"
FEATURE_PROBE_NOISE_SAMPLES="${FEATURE_PROBE_NOISE_SAMPLES:-1}"
FEATURE_PROBE_TOP_K="${FEATURE_PROBE_TOP_K:-20}"
FEATURE_PROBE_TOP_M_ACTIVE="${FEATURE_PROBE_TOP_M_ACTIVE:-100}"
FEATURE_PROBE_MAX_BATCHES="${FEATURE_PROBE_MAX_BATCHES:-}"
FEATURE_REPORT_MAX_FEATURES="${FEATURE_REPORT_MAX_FEATURES:-200}"
FEATURE_REPORT_TOP_EXAMPLES="${FEATURE_REPORT_TOP_EXAMPLES:-20}"
FEATURE_REPORT_SAVE_THUMBNAILS="${FEATURE_REPORT_SAVE_THUMBNAILS:-1}"
FEATURE_FLOW_TOP_FEATURES_PER_LAYER="${FEATURE_FLOW_TOP_FEATURES_PER_LAYER:-6}"
PI05_LANGFUSE_TRACE="${PI05_LANGFUSE_TRACE:-0}"
GCS_URI="${GCS_URI:-}"

REMOTE_CHECKPOINT="${TRANSCODER_CHECKPOINT:-}"
LOCAL_TRANSCODER_CHECKPOINT="${LOCAL_TRANSCODER_CHECKPOINT:-}"

remote_quote() {
  printf "%q" "$1"
}

"${ROOT}/scripts/gcp_create_l4_vm.sh"
if [[ "${SKIP_GCP_SETUP:-0}" != "1" ]]; then
  "${ROOT}/scripts/gcp_setup_uv_libero.sh"
fi

gcloud compute ssh "${VM_NAME}" \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --command="mkdir -p ~/groot-run/src ~/groot-run/scripts ~/groot-run/cloud ~/groot-run/docs ~/groot-run/outputs ~/groot-run/checkpoints/transcoders"

gcloud compute scp \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --recurse \
  "${ROOT}/src" \
  "${ROOT}/scripts" \
  "${ROOT}/cloud" \
  "${ROOT}/docs" \
  "${ROOT}/pyproject.toml" \
  "${ROOT}/uv.lock" \
  "${ROOT}/README.md" \
  "${VM_NAME}:~/groot-run/"

if [[ -n "${LOCAL_TRANSCODER_CHECKPOINT}" ]]; then
  if [[ ! -f "${LOCAL_TRANSCODER_CHECKPOINT}" ]]; then
    echo "LOCAL_TRANSCODER_CHECKPOINT does not exist: ${LOCAL_TRANSCODER_CHECKPOINT}" >&2
    exit 2
  fi
  checkpoint_name="$(basename "${LOCAL_TRANSCODER_CHECKPOINT}")"
  gcloud compute scp \
    --project="${PROJECT}" \
    --zone="${ZONE}" \
    "${LOCAL_TRANSCODER_CHECKPOINT}" \
    "${VM_NAME}:~/groot-run/checkpoints/transcoders/${checkpoint_name}"
  REMOTE_CHECKPOINT="checkpoints/transcoders/${checkpoint_name}"
fi

if [[ -z "${REMOTE_CHECKPOINT}" ]]; then
  REMOTE_CHECKPOINT="checkpoints/transcoders/step_027233.pt"
fi

langfuse_exports=""
for name in \
  LANGFUSE_PUBLIC_KEY \
  LANGFUSE_SECRET_KEY \
  LANGFUSE_BASE_URL \
  LANGFUSE_HOST \
  PI05_LANGFUSE_USER_ID \
  PI05_LANGFUSE_SESSION_ID \
  PI05_LANGFUSE_TENANT_ID \
  PI05_LANGFUSE_CALLER \
  PI05_LANGFUSE_TAGS \
  PI05_LANGFUSE_ENVIRONMENT \
  PI05_LANGFUSE_MEDIA \
  PI05_LANGFUSE_MAX_IMAGES \
  PI05_LANGFUSE_MAX_MEDIA_BYTES \
  PI05_LANGFUSE_MAX_IMAGE_SIDE
do
  if [[ -n "${!name:-}" ]]; then
    langfuse_exports+="export ${name}=$(remote_quote "${!name}"); "
  fi
done

remote_cmd="
set -euo pipefail
REMOTE_ROOT=\"\$HOME/groot-run\"
REMOTE_FEATURE_DIR=\"\$REMOTE_ROOT/$(remote_quote "${REMOTE_FEATURE_REL}")\"
REMOTE_ARCHIVE=\"\$REMOTE_ROOT/$(remote_quote "${REMOTE_ARCHIVE_REL}")\"
REMOTE_ARCHIVE_DIR=\"\$(dirname \"\$REMOTE_ARCHIVE\")\"
REMOTE_CHECKPOINT_VALUE=$(remote_quote "${REMOTE_CHECKPOINT}")
if [[ \"\$REMOTE_CHECKPOINT_VALUE\" == /* ]]; then
  REMOTE_CHECKPOINT_PATH=\"\$REMOTE_CHECKPOINT_VALUE\"
else
  REMOTE_CHECKPOINT_PATH=\"\$REMOTE_ROOT/\$REMOTE_CHECKPOINT_VALUE\"
fi

cd \"\$REMOTE_ROOT\"
export PATH=\"\$HOME/.local/bin:\$PATH\"
export HF_HOME=\"\${HF_HOME:-\$HOME/.cache/huggingface}\"
export HF_HUB_CACHE=\"\${HF_HUB_CACHE:-\$HF_HOME/hub}\"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_DISABLE_PROGRESS_BARS=0
export TQDM_DISABLE=0
export TQDM_MININTERVAL=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export MPLBACKEND=Agg
export PYTHONPATH=\"\$REMOTE_ROOT/src:\$REMOTE_ROOT/scripts:\${PYTHONPATH:-}\"
export TRANSCODER_PROBE_NAME=$(remote_quote "${TRANSCODER_PROBE_NAME}")
export PI05_LANGFUSE_TRACE=$(remote_quote "${PI05_LANGFUSE_TRACE}")
${langfuse_exports}

if [[ ! -f \"\$REMOTE_CHECKPOINT_PATH\" ]]; then
  echo \"Transcoder checkpoint not found on VM: \$REMOTE_CHECKPOINT_PATH\" >&2
  echo \"Set LOCAL_TRANSCODER_CHECKPOINT=/path/to/step.pt to copy it, or TRANSCODER_CHECKPOINT to an existing VM path.\" >&2
  exit 2
fi

uv pip install --python .venv/bin/python langfuse opencv-python >/tmp/pi05_langfuse_install.log
rm -rf \"\$REMOTE_FEATURE_DIR\"
mkdir -p \"\$REMOTE_FEATURE_DIR\" \"\$REMOTE_ARCHIVE_DIR\"

collect_args=(
  --policy-path $(remote_quote "${POLICY_PATH}")
  --checkpoint \"\$REMOTE_CHECKPOINT_PATH\"
  --output-dir \"\$REMOTE_FEATURE_DIR\"
  --episodes $(remote_quote "${FEATURE_PROBE_EPISODES}")
  --batch-size $(remote_quote "${FEATURE_PROBE_BATCH_SIZE}")
  --num-workers $(remote_quote "${FEATURE_PROBE_NUM_WORKERS}")
  --collection-mode $(remote_quote "${FEATURE_PROBE_COLLECTION_MODE}")
  --num-inference-steps $(remote_quote "${FEATURE_PROBE_NUM_INFERENCE_STEPS}")
  --noise-samples $(remote_quote "${FEATURE_PROBE_NOISE_SAMPLES}")
  --top-k $(remote_quote "${FEATURE_PROBE_TOP_K}")
  --top-m-active $(remote_quote "${FEATURE_PROBE_TOP_M_ACTIVE}")
  --device cuda
  --policy-dtype bfloat16
)
if [[ -n $(remote_quote "${FEATURE_PROBE_MAX_BATCHES}") ]]; then
  collect_args+=(--max-batches $(remote_quote "${FEATURE_PROBE_MAX_BATCHES}"))
fi
if [[ \"\${PI05_LANGFUSE_TRACE}\" == \"1\" ]]; then
  collect_args+=(--langfuse-trace)
fi

.venv/bin/python scripts/collect_pi05_transcoder_features.py \"\${collect_args[@]}\" 2>&1 | tee \"\$REMOTE_FEATURE_DIR/collect.log\"

report_args=(
  --feature-dir \"\$REMOTE_FEATURE_DIR\"
  --max-features $(remote_quote "${FEATURE_REPORT_MAX_FEATURES}")
  --top-examples $(remote_quote "${FEATURE_REPORT_TOP_EXAMPLES}")
)
if [[ $(remote_quote "${FEATURE_REPORT_SAVE_THUMBNAILS}") == \"1\" ]]; then
  report_args+=(--save-thumbnails --device cuda --policy-dtype bfloat16)
fi
.venv/bin/python scripts/make_pi05_feature_report.py \"\${report_args[@]}\" 2>&1 | tee \"\$REMOTE_FEATURE_DIR/feature_report.log\"

flow_args=(
  --feature-dir \"\$REMOTE_FEATURE_DIR\"
  --checkpoint \"\$REMOTE_CHECKPOINT_PATH\"
  --top-features-per-layer $(remote_quote "${FEATURE_FLOW_TOP_FEATURES_PER_LAYER}")
)
if [[ \"\${PI05_LANGFUSE_TRACE}\" == \"1\" ]]; then
  flow_args+=(--langfuse-trace)
fi
.venv/bin/python scripts/make_pi05_transcoder_flow_report.py \"\${flow_args[@]}\" 2>&1 | tee \"\$REMOTE_FEATURE_DIR/flow_report.log\"

tar -C \"\$REMOTE_ROOT/outputs/features/pi05_libero\" -czf \"\$REMOTE_ARCHIVE\" $(remote_quote "${TRANSCODER_PROBE_NAME}")
ls -lh \"\$REMOTE_ARCHIVE\"
if [[ -n $(remote_quote "${GCS_URI}") ]]; then
  gcloud storage cp \"\$REMOTE_ARCHIVE\" $(remote_quote "${GCS_URI%/}/")
fi
"

gcloud compute ssh "${VM_NAME}" \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --command="${remote_cmd}"

mkdir -p "${LOCAL_ARTIFACT_ROOT}"
gcloud compute scp \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  "${VM_NAME}:~/groot-run/${TRANSCODER_PROBE_NAME}/${TRANSCODER_PROBE_NAME}.tar.gz" \
  "${LOCAL_ARTIFACT_ROOT}/"

echo "Remote feature dir: ~/groot-run/outputs/features/pi05_libero/${TRANSCODER_PROBE_NAME}"
echo "Remote archive: ~/groot-run/${TRANSCODER_PROBE_NAME}/${TRANSCODER_PROBE_NAME}.tar.gz"
echo "Local archive: ${LOCAL_ARTIFACT_ROOT}/${TRANSCODER_PROBE_NAME}.tar.gz"

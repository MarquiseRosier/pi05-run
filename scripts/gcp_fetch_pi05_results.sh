#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-breadwinner-415122}"
ZONE="${ZONE:-us-central1-a}"
VM_NAME="${VM_NAME:-lerobot-libero-l4}"
RUN_ID="${1:-latest}"
LOCAL_ROOT="${LOCAL_ROOT:-outputs/eval/pi05_libero}"

if [[ "${RUN_ID}" != "latest" && ! "${RUN_ID}" =~ ^[0-9]{8}-[0-9]{6}$ ]]; then
  echo "Run id must be 'latest' or a timestamp like 20260815-082451" >&2
  exit 2
fi

if [[ "${RUN_ID}" == "latest" ]]; then
  RUN_ID="$(gcloud compute ssh "${VM_NAME}" \
    --project="${PROJECT}" \
    --zone="${ZONE}" \
    --command='basename "$(find "$HOME/groot-run/outputs/eval/pi05_libero" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1)"')"
fi

mkdir -p "${LOCAL_ROOT}"

gcloud compute scp \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --recurse "${VM_NAME}:~/groot-run/outputs/eval/pi05_libero/${RUN_ID}" "${LOCAL_ROOT}/"

echo "Fetched ${RUN_ID} into ${LOCAL_ROOT}/${RUN_ID}"

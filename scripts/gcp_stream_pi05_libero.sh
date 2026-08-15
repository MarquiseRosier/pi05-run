#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-breadwinner-415122}"
ZONE="${ZONE:-us-central1-a}"
VM_NAME="${VM_NAME:-lerobot-libero-l4}"
RUN_ID="${1:-latest}"

if [[ "${RUN_ID}" != "latest" && ! "${RUN_ID}" =~ ^[0-9]{8}-[0-9]{6}$ ]]; then
  echo "Run id must be 'latest' or a timestamp like 20260815-082451" >&2
  exit 2
fi

if [[ "${RUN_ID}" == "latest" ]]; then
  REMOTE_COMMAND='set -euo pipefail
RUN_DIR="$(find "$HOME/groot-run/outputs/eval/pi05_libero" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1)"
if [[ -z "${RUN_DIR}" ]]; then
  echo "No pi05_libero runs found." >&2
  exit 1
fi
echo "Streaming ${RUN_DIR}/run.log"
while [[ ! -f "${RUN_DIR}/run.log" ]]; do sleep 2; done
tail -n 120 -F "${RUN_DIR}/run.log"'
else
  REMOTE_COMMAND="set -euo pipefail
RUN_DIR=\"\$HOME/groot-run/outputs/eval/pi05_libero/${RUN_ID}\"
echo \"Streaming \${RUN_DIR}/run.log\"
while [[ ! -f \"\${RUN_DIR}/run.log\" ]]; do sleep 2; done
tail -n 120 -F \"\${RUN_DIR}/run.log\""
fi

gcloud compute ssh "${VM_NAME}" \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --command="${REMOTE_COMMAND}"

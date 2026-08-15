#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-breadwinner-415122}"
ZONE="${ZONE:-us-central1-a}"
VM_NAME="${VM_NAME:-lerobot-libero-l4}"
MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-8}"
DISK_SIZE="${DISK_SIZE:-200GB}"
IMAGE_FAMILY="${IMAGE_FAMILY:-pytorch-2-9-cu129-ubuntu-2404-nvidia-580}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"

STATUS="$(gcloud compute instances describe "${VM_NAME}" \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --format="value(status)" 2>/dev/null || true)"

if [[ -n "${STATUS}" ]]; then
  if [[ "${STATUS}" == "RUNNING" ]]; then
    echo "${VM_NAME} is already RUNNING in ${ZONE}"
  else
    gcloud compute instances start "${VM_NAME}" --project="${PROJECT}" --zone="${ZONE}" --quiet
  fi
  exit 0
fi

gcloud compute instances create "${VM_NAME}" \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --machine-type="${MACHINE_TYPE}" \
  --image-family="${IMAGE_FAMILY}" \
  --image-project="${IMAGE_PROJECT}" \
  --boot-disk-size="${DISK_SIZE}" \
  --boot-disk-type=pd-balanced \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --no-restart-on-failure \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --labels=purpose=lerobot-libero,owner=marquise,managed-by=repo \
  --quiet

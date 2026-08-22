#!/usr/bin/env bash
set -euo pipefail

SUITE="${1:-libero_spatial}"
EPISODES="${2:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GROOT_RUN="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${GROOT_RUN}/.." && pwd)"
ISAAC_ROOT="${ISAAC_ROOT:-${REPO_ROOT}/MI_VLA/Isaac-GR00T}"
TASK_MAP="${TASK_MAP:-${GROOT_RUN}/task_map.json}"

GROOT_PYTHON="${GROOT_PYTHON:-python3}"
LIBERO_PYTHON="${LIBERO_PYTHON:-${ISAAC_ROOT}/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python}"
HOST_PYTHON="${HOST_PYTHON:-python3}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/eval/groot_libero}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_ID}"
MODEL_PATH="${MODEL_PATH:-${ISAAC_ROOT}/checkpoints/GR00T-N1.7-LIBERO/libero_10}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-LIBERO_PANDA}"
POLICY_HOST="${POLICY_HOST:-127.0.0.1}"
POLICY_PORT="${POLICY_PORT:-5555}"
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"
N_ENVS="${N_ENVS:-1}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-720}"
TASK_IDS="${TASK_IDS:-}"
CAPTURE_ACTIVATIONS="${CAPTURE_ACTIVATIONS:-0}"
SERVER_LOG="${OUTPUT_DIR}/server.log"
RUN_LOG="${OUTPUT_DIR}/run.log"
EVAL_INFO="${OUTPUT_DIR}/eval_info.json"

mkdir -p "${OUTPUT_DIR}/videos" "${OUTPUT_DIR}/activation_capture"

if [[ ! -d "${ISAAC_ROOT}" ]]; then
  echo "ERROR: Isaac-GR00T not found at ${ISAAC_ROOT}" >&2
  exit 2
fi
if [[ ! -x "${GROOT_PYTHON}" && ! -f "${GROOT_PYTHON}" ]]; then
  if ! command -v "${GROOT_PYTHON}" >/dev/null 2>&1; then
    echo "ERROR: GROOT_PYTHON not found: ${GROOT_PYTHON}" >&2
    exit 2
  fi
fi
if [[ ! -x "${LIBERO_PYTHON}" ]]; then
  echo "ERROR: LIBERO_PYTHON not found: ${LIBERO_PYTHON}" >&2
  echo "Run MI_VLA/Isaac-GR00T/gr00t/eval/sim/LIBERO/setup_libero.sh first." >&2
  exit 2
fi
if [[ ! -f "${TASK_MAP}" ]]; then
  echo "ERROR: task map missing: ${TASK_MAP}" >&2
  exit 2
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export MPLBACKEND="Agg"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

TASKS_JSON="$("${HOST_PYTHON}" "${SCRIPT_DIR}/resolve_tasks.py" --task-map "${TASK_MAP}" --suite "${SUITE}" --task-ids "${TASK_IDS}")"
printf "resolved_tasks %s\n" "${TASKS_JSON}"

server_pid=""
cleanup() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

wait_for_server() {
  local elapsed=0
  local timeout="${SERVER_READY_TIMEOUT:-600}"
  while (( elapsed < timeout )); do
    if grep -q "listening on" "${SERVER_LOG}" 2>/dev/null; then
      return 0
    fi
    if [[ -n "${server_pid}" ]] && ! kill -0 "${server_pid}" 2>/dev/null; then
      echo "ERROR: GR00T server exited before it was ready. See ${SERVER_LOG}" >&2
      tail -n 80 "${SERVER_LOG}" >&2 || true
      return 1
    fi
    sleep 2
    elapsed=$((elapsed + 2))
    printf "waiting_for_server elapsed=%ss\n" "${elapsed}"
  done
  echo "ERROR: timed out waiting for GR00T server on port ${POLICY_PORT}" >&2
  return 1
}

CAPTURE_PYTHONPATH="${GROOT_RUN}/activation_capture${PYTHONPATH:+:${PYTHONPATH}}"
export GROOT_CAPTURE_DIR="${OUTPUT_DIR}/activation_capture"
export LEROBOT_CAPTURE_DIR="${GROOT_CAPTURE_DIR}"
export CAPTURE_ACTIVATIONS
export GROOT_CAPTURE_ACTIVATIONS="${CAPTURE_ACTIVATIONS}"
export CAPTURE_PARAM_STATS="${CAPTURE_PARAM_STATS:-0}"
export CAPTURE_MAX_CHUNKS="${CAPTURE_MAX_CHUNKS:-40}"
export CAPTURE_LAYER_STRIDE="${CAPTURE_LAYER_STRIDE:-1}"
export CAPTURE_MAX_BINS="${CAPTURE_MAX_BINS:-64}"
export CAPTURE_FAMILIES="${CAPTURE_FAMILIES:-vision,prefix,expert,projection}"

printf "Starting GR00T server model_path=%s\n" "${MODEL_PATH}"
(
  cd "${ISAAC_ROOT}"
  if [[ "${CAPTURE_ACTIVATIONS}" == "1" ]]; then
    export PYTHONPATH="${CAPTURE_PYTHONPATH}"
  fi
  exec "${GROOT_PYTHON}" -u gr00t/eval/run_gr00t_server.py \
    --model-path "${MODEL_PATH}" \
    --embodiment-tag "${EMBODIMENT_TAG}" \
    --use-sim-policy-wrapper \
    --host 0.0.0.0 \
    --port "${POLICY_PORT}"
) >"${SERVER_LOG}" 2>&1 &
server_pid=$!
printf "server_pid %s\n" "${server_pid}"
wait_for_server

{
  echo "run_id ${RUN_ID}"
  echo "suite ${SUITE}"
  echo "episodes ${EPISODES}"
  echo "model_path ${MODEL_PATH}"
  echo "tasks ${TASKS_JSON}"
} | tee -a "${RUN_LOG}"

"${HOST_PYTHON}" - "${OUTPUT_DIR}" "${TASKS_JSON}" "${EPISODES}" "${LIBERO_PYTHON}" "${ISAAC_ROOT}" \
  "${POLICY_HOST}" "${POLICY_PORT}" "${N_ACTION_STEPS}" "${N_ENVS}" "${MAX_EPISODE_STEPS}" "${RUN_LOG}" "${EVAL_INFO}" <<'PY'
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
tasks = json.loads(sys.argv[2])
episodes = int(sys.argv[3])
libero_python = sys.argv[4]
isaac_root = Path(sys.argv[5])
host = sys.argv[6]
port = sys.argv[7]
n_action_steps = sys.argv[8]
n_envs = sys.argv[9]
max_episode_steps = sys.argv[10]
run_log = Path(sys.argv[11])
eval_info_path = Path(sys.argv[12])

per_task = []
successes = []
for item in tasks:
    suite = item["suite"]
    task_id = int(item["task_id"])
    env_name = item["env_name"]
    video_dir = output_dir / "videos" / f"{suite}_{task_id}"
    video_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        libero_python,
        "-u",
        str(isaac_root / "gr00t/eval/rollout_policy.py"),
        "--n-episodes",
        str(episodes),
        "--n-envs",
        str(n_envs),
        "--policy-client-host",
        host,
        "--policy-client-port",
        str(port),
        "--max-episode-steps",
        str(max_episode_steps),
        "--n-action-steps",
        str(n_action_steps),
        "--env-name",
        env_name,
        "--video-dir",
        str(video_dir),
    ]
    print(f"Running {env_name}", flush=True)
    with run_log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n$ {' '.join(cmd)}\n")
        proc = subprocess.run(cmd, cwd=str(isaac_root), stdout=handle, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"rollout_policy.py failed for {env_name} with code {proc.returncode}")

    videos = sorted(path for path in video_dir.glob("**/*.mp4") if path.is_file())
    renamed = []
    for idx, src in enumerate(videos[:episodes]):
        dest = video_dir / f"eval_episode_{idx}.mp4"
        if src.resolve() != dest.resolve():
            if dest.exists():
                dest.unlink()
            shutil.move(str(src), str(dest))
        renamed.append(str(dest))

    success_rate = None
    text = run_log.read_text(errors="replace")
    for line in reversed(text.splitlines()):
        if "success rate:" in line.lower():
            try:
                success_rate = float(line.split(":")[-1].strip())
            except ValueError:
                success_rate = None
            break
    if success_rate is not None:
        successes.append(success_rate)
    per_task.append(
        {
            "suite": suite,
            "task_id": task_id,
            "task_group": suite,
            "env_name": env_name,
            "n_episodes": episodes,
            "success_rate": success_rate,
            "metrics": {"video_paths": renamed},
        }
    )

overall = sum(successes) / len(successes) if successes else None
payload = {
    "run_id": output_dir.name,
    "output_dir": str(output_dir),
    "n_action_steps": int(n_action_steps),
    "overall_success_rate": overall,
    "per_task": per_task,
}
eval_info_path.write_text(json.dumps(payload, indent=2))
print(f"Saved results in {output_dir}")
print(f"eval_info {eval_info_path}")
if overall is not None:
    print(f"overall_success_rate {overall}")
PY

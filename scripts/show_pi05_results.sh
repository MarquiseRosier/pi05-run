#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-latest}"
ACTION="${2:-summary}"
ROOT="${ROOT:-outputs/eval/pi05_libero}"
PYTHON_BIN="${PYTHON:-}"

if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "Python is required to summarize metrics. Install python3 or set PYTHON=/path/to/python." >&2
    exit 1
  fi
fi

if [[ "${RUN_ID}" != "latest" && ! "${RUN_ID}" =~ ^[0-9]{8}-[0-9]{6}$ ]]; then
  echo "Run id must be 'latest' or a timestamp like 20260815-082451" >&2
  exit 2
fi

if [[ "${RUN_ID}" == "latest" ]]; then
  RUN_DIR="$(find "${ROOT}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1)"
else
  RUN_DIR="${ROOT}/${RUN_ID}"
fi

if [[ -z "${RUN_DIR:-}" || ! -d "${RUN_DIR}" ]]; then
  echo "No local Pi0.5 LIBERO result found under ${ROOT}." >&2
  echo "Fetch one first: ./scripts/gcp_fetch_pi05_results.sh" >&2
  exit 1
fi

"${PYTHON_BIN}" - "${RUN_DIR}" <<'PY'
import ast
import re
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
log_path = run_dir / "run.log"
videos = sorted(run_dir.glob("videos/**/*.mp4"))

print(f"Run dir: {run_dir}")
print(f"Log: {log_path if log_path.exists() else 'missing'}")
print(f"Videos: {len(videos)}")

if not log_path.exists():
    raise SystemExit(0)

ansi = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
lines = ansi.sub("", log_path.read_text(errors="replace")).splitlines()

metrics = None
for i, line in enumerate(lines):
    if "Aggregated Metrics for overall:" not in line:
        continue
    for candidate in lines[i + 1 : i + 6]:
        start = candidate.find("{")
        if start == -1:
            continue
        try:
            metrics = ast.literal_eval(candidate[start:])
            break
        except (SyntaxError, ValueError):
            continue

if metrics:
    print(f"Success: {metrics.get('pc_success')}%")
    print(f"Episodes: {metrics.get('n_episodes')}")
    print(f"Avg sum reward: {metrics.get('avg_sum_reward')}")
    print(f"Eval seconds: {metrics.get('eval_s')}")
    print(f"Seconds per episode: {metrics.get('eval_ep_s')}")
else:
    print("Metrics: not found yet")

for video in videos[:10]:
    print(f"Video: {video}")
if len(videos) > 10:
    print(f"... {len(videos) - 10} more videos")
PY

open_path() {
  local path="$1"
  if command -v open >/dev/null 2>&1; then
    open "${path}"
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "${path}"
  else
    echo "No opener found. Open manually: ${path}" >&2
  fi
}

case "${ACTION}" in
  summary)
    ;;
  open-dir)
    open_path "${RUN_DIR}"
    ;;
  open-video)
    VIDEO="$(find "${RUN_DIR}/videos" -type f -name '*.mp4' 2>/dev/null | sort | head -1)"
    if [[ -z "${VIDEO}" ]]; then
      echo "No videos found in ${RUN_DIR}/videos" >&2
      exit 1
    fi
    open_path "${VIDEO}"
    ;;
  *)
    echo "Action must be one of: summary, open-dir, open-video" >&2
    exit 2
    ;;
esac

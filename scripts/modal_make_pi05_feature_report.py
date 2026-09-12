#!/usr/bin/env python
"""Launch Pi0.5 transcoder feature-report rendering on Modal.

Example:
    modal run scripts/modal_make_pi05_feature_report.py \
      --feature-dir /vol/outputs/features/pi05_libero/pilot_ep0-49_inference10_top20
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import modal


APP_NAME = "pi05-transcoder-feature-report"
REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_REPO = "/workspace/pi05-run"
VOLUME_MOUNT = "/vol"
HF_CACHE_DIR = f"{VOLUME_MOUNT}/hf_cache"
DEFAULT_FEATURE_DIR = f"{VOLUME_MOUNT}/outputs/features/pi05_libero/pilot_ep0-49_inference10_top20"


def _ignore_source(path: Path) -> bool:
    parts = set(path.parts)
    return bool(
        {
            ".git",
            ".venv",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            "pi0.5",
            "outputs",
        }
        & parts
    )


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0", "libsm6", "libxext6")
    .pip_install("uv")
    .add_local_dir(REPO_ROOT, remote_path=REMOTE_REPO, copy=True, ignore=_ignore_source)
    .workdir(REMOTE_REPO)
    .run_commands("uv sync --frozen")
)

app = modal.App(APP_NAME, image=image)
pi05_volume = modal.Volume.from_name("pi05-libero-data", create_if_missing=False)
hf_secret = modal.Secret.from_name("huggingface-token", required_keys=["HF_TOKEN"])


def _append_arg(args: list[str], name: str, value: object | None) -> None:
    if value is None:
        return
    if isinstance(value, str) and value == "":
        return
    args.extend([name, str(value)])


@app.function(
    volumes={VOLUME_MOUNT: pi05_volume},
    secrets=[hf_secret],
    timeout=4 * 60 * 60,
    memory=16384,
)
def report_remote(report_args: list[str]) -> None:
    env = os.environ.copy()
    env.update(
        {
            "HF_HOME": HF_CACHE_DIR,
            "HF_HUB_CACHE": f"{HF_CACHE_DIR}/hub",
            "HF_DATASETS_CACHE": f"{HF_CACHE_DIR}/datasets",
            "HUGGING_FACE_HUB_TOKEN": env["HF_TOKEN"],
            "PYTORCH_ENABLE_MPS_FALLBACK": "0",
            "PYTHONPATH": f"{REMOTE_REPO}/src:{REMOTE_REPO}/scripts",
        }
    )
    os.makedirs(HF_CACHE_DIR, exist_ok=True)

    cmd = [f"{REMOTE_REPO}/.venv/bin/python", "scripts/make_pi05_feature_report.py", *report_args]
    print("$ " + shlex.join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=REMOTE_REPO, env=env, check=False)
    pi05_volume.commit()
    if result.returncode != 0:
        raise SystemExit(result.returncode)


@app.local_entrypoint()
def main(
    feature_dir: str = DEFAULT_FEATURE_DIR,
    output_html: str | None = None,
    candidate_csv: str | None = None,
    candidate_json: str | None = None,
    max_features: int = 200,
    top_examples: int = 20,
    features: str | None = None,
    sort_by: str = "interesting",
    min_frequency: float = 0.0,
    max_frequency: float = 1.0,
    min_max_score: float = 0.0,
    save_thumbnails: bool = False,
    thumbnail_camera: str | None = None,
    batch_size: int = 4,
    num_workers: int = 0,
    local_files_only: bool = False,
) -> None:
    report_args: list[str] = []
    _append_arg(report_args, "--feature-dir", feature_dir)
    _append_arg(report_args, "--output-html", output_html)
    _append_arg(report_args, "--candidate-csv", candidate_csv)
    _append_arg(report_args, "--candidate-json", candidate_json)
    _append_arg(report_args, "--max-features", max_features)
    _append_arg(report_args, "--top-examples", top_examples)
    _append_arg(report_args, "--features", features)
    _append_arg(report_args, "--sort-by", sort_by)
    _append_arg(report_args, "--min-frequency", min_frequency)
    _append_arg(report_args, "--max-frequency", max_frequency)
    _append_arg(report_args, "--min-max-score", min_max_score)
    _append_arg(report_args, "--thumbnail-camera", thumbnail_camera)
    _append_arg(report_args, "--batch-size", batch_size)
    _append_arg(report_args, "--num-workers", num_workers)
    if save_thumbnails:
        report_args.append("--save-thumbnails")
    if local_files_only:
        report_args.append("--local-files-only")

    print(f"Rendering feature report from: {feature_dir}")
    print("Remote command:")
    print("$ " + shlex.join([f"{REMOTE_REPO}/.venv/bin/python", "scripts/make_pi05_feature_report.py", *report_args]))
    call = report_remote.spawn(report_args)
    print(f"Spawned Modal call: {call.object_id}")
    call.get()

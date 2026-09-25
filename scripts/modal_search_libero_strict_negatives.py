#!/usr/bin/env python
"""Launch LIBERO strict-negative prompt search on Modal."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import modal


APP_NAME = "pi05-libero-strict-negative-search"
REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_REPO = "/workspace/pi05-run"
VOLUME_MOUNT = "/vol"
HF_CACHE_DIR = f"{VOLUME_MOUNT}/hf_cache"
DEFAULT_OUTPUT_DIR = f"{VOLUME_MOUNT}/outputs/case_studies/pi05_libero/f9970_strict_negative_prompt_search"


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
    timeout=2 * 60 * 60,
)
def search_remote(search_args: list[str]) -> None:
    env = os.environ.copy()
    env.update(
        {
            "HF_HOME": HF_CACHE_DIR,
            "HF_HUB_CACHE": f"{HF_CACHE_DIR}/hub",
            "HF_DATASETS_CACHE": f"{HF_CACHE_DIR}/datasets",
            "HUGGING_FACE_HUB_TOKEN": env["HF_TOKEN"],
            "PYTHONPATH": f"{REMOTE_REPO}/src:{REMOTE_REPO}/scripts",
        }
    )
    os.makedirs(HF_CACHE_DIR, exist_ok=True)

    cmd = [f"{REMOTE_REPO}/.venv/bin/python", "scripts/search_libero_strict_negatives.py", *search_args]
    print("$ " + shlex.join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=REMOTE_REPO, env=env, check=False)
    pi05_volume.commit()
    if result.returncode != 0:
        raise SystemExit(result.returncode)


@app.local_entrypoint()
def main(
    policy_path: str = "lerobot/pi05_libero_finetuned",
    positive_task: str = "put the black bowl in the bottom drawer of the cabinet and close it",
    output_dir: str = DEFAULT_OUTPUT_DIR,
    episodes: str | None = None,
    top_n: int = 120,
    local_files_only: bool = False,
) -> None:
    search_args: list[str] = []
    _append_arg(search_args, "--policy-path", policy_path)
    _append_arg(search_args, "--positive-task", positive_task)
    _append_arg(search_args, "--output-dir", output_dir)
    _append_arg(search_args, "--episodes", episodes)
    _append_arg(search_args, "--top-n", top_n)
    if local_files_only:
        search_args.append("--local-files-only")

    print("Launching Modal LIBERO prompt search")
    print(f"Volume: pi05-libero-data mounted at {VOLUME_MOUNT}")
    print("Args: " + shlex.join(search_args))
    search_remote.remote(search_args)

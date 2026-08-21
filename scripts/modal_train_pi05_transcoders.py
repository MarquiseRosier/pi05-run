#!/usr/bin/env python
"""Launch Pi0.5 transcoder training on Modal.

Example:
    modal run scripts/modal_train_pi05_transcoders.py --gpu B200 --num-feed-forwards 10
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import modal


APP_NAME = "pi05-transcoder-training"
REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_REPO = "/workspace/pi05-run"
VOLUME_MOUNT = "/vol"
HF_CACHE_DIR = f"{VOLUME_MOUNT}/hf_cache"
DEFAULT_OUTPUT_DIR = f"{VOLUME_MOUNT}/outputs/transcoders/pi05_libero"


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


def _normalized_gpu(gpu: str) -> str:
    gpu = gpu.strip()
    aliases = {
        "a100": "A100",
        "a100-40gb": "A100",
        "a100-80gb": "A100-80GB",
        "b200": "B200",
        "h100": "H100",
        "h200": "H200",
        "l4": "L4",
        "t4": "T4",
    }
    return aliases.get(gpu.lower(), gpu)


def _append_arg(args: list[str], name: str, value: object | None) -> None:
    if value is None:
        return
    args.extend([name, str(value)])


@app.function(
    volumes={VOLUME_MOUNT: pi05_volume},
    secrets=[hf_secret],
    timeout=24 * 60 * 60,
)
def train_remote(train_args: list[str]) -> None:
    env = os.environ.copy()
    env.update(
        {
            "HF_HOME": HF_CACHE_DIR,
            "HF_HUB_CACHE": f"{HF_CACHE_DIR}/hub",
            "HF_DATASETS_CACHE": f"{HF_CACHE_DIR}/datasets",
            "HUGGING_FACE_HUB_TOKEN": env["HF_TOKEN"],
            "PYTORCH_ENABLE_MPS_FALLBACK": "0",
        }
    )
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

    cmd = [f"{REMOTE_REPO}/.venv/bin/python", "scripts/train_pi05_transcoders.py", *train_args]
    print("$ " + shlex.join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=REMOTE_REPO, env=env, check=False)
    pi05_volume.commit()
    if result.returncode != 0:
        raise SystemExit(result.returncode)


@app.local_entrypoint()
def main(
    gpu: str = "B200",
    policy_path: str = "lerobot/pi05_libero_finetuned",
    num_feed_forwards: int = 10,
    batch_size: int = 1,
    episodes: str | None = "0",
    num_workers: int = 0,
    collection_mode: str = "random-timestep",
    num_inference_steps: int | None = None,
    policy_dtype: str = "bfloat16",
    lr: float = 1e-4,
    lambda_l1: float = 1e-4,
    expansion_factor: int = 1,
    buffer_capacity: int = 5000,
    min_buffer_records: int = 50,
    transcoder_batch_size: int = 16,
    transcoder_epochs_per_ff: int = 1,
    grad_clip_norm: float = 1.0,
    save_every: int = 5,
    log_every: int = 1,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    local_files_only: bool = False,
    no_progress: bool = False,
) -> None:
    train_args: list[str] = []
    _append_arg(train_args, "--policy-path", policy_path)
    _append_arg(train_args, "--output-dir", output_dir)
    _append_arg(train_args, "--num-feed-forwards", num_feed_forwards)
    _append_arg(train_args, "--batch-size", batch_size)
    _append_arg(train_args, "--episodes", episodes)
    _append_arg(train_args, "--num-workers", num_workers)
    _append_arg(train_args, "--collection-mode", collection_mode)
    _append_arg(train_args, "--num-inference-steps", num_inference_steps)
    _append_arg(train_args, "--device", "cuda")
    _append_arg(train_args, "--policy-dtype", policy_dtype)
    _append_arg(train_args, "--lr", lr)
    _append_arg(train_args, "--lambda-l1", lambda_l1)
    _append_arg(train_args, "--expansion-factor", expansion_factor)
    _append_arg(train_args, "--buffer-capacity", buffer_capacity)
    _append_arg(train_args, "--min-buffer-records", min_buffer_records)
    _append_arg(train_args, "--transcoder-batch-size", transcoder_batch_size)
    _append_arg(train_args, "--transcoder-epochs-per-ff", transcoder_epochs_per_ff)
    _append_arg(train_args, "--grad-clip-norm", grad_clip_norm)
    _append_arg(train_args, "--save-every", save_every)
    _append_arg(train_args, "--log-every", log_every)
    if local_files_only:
        train_args.append("--local-files-only")
    if no_progress:
        train_args.append("--no-progress")

    gpu_spec = _normalized_gpu(gpu)
    print(f"Launching Modal training on gpu={gpu_spec}")
    print(f"Volume: pi05-libero-data mounted at {VOLUME_MOUNT}")
    print(f"Hugging Face cache: {HF_CACHE_DIR}")
    print(f"Output dir: {output_dir}")
    print("Remote command:")
    print("$ " + shlex.join([f"{REMOTE_REPO}/.venv/bin/python", "scripts/train_pi05_transcoders.py", *train_args]))

    train_remote.with_options(gpu=gpu_spec).remote(train_args)

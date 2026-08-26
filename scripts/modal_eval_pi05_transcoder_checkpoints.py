#!/usr/bin/env python
"""Launch Pi0.5 transcoder checkpoint evaluation on Modal."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import modal


APP_NAME = "pi05-transcoder-checkpoint-eval"
REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_REPO = "/workspace/pi05-run"
VOLUME_MOUNT = "/vol"
HF_CACHE_DIR = f"{VOLUME_MOUNT}/hf_cache"


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
def eval_remote(eval_args: list[str]) -> None:
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

    cmd = [f"{REMOTE_REPO}/.venv/bin/python", "scripts/eval_pi05_transcoder_checkpoints.py", *eval_args]
    print("$ " + shlex.join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=REMOTE_REPO, env=env, check=False)
    pi05_volume.commit()
    if result.returncode != 0:
        raise SystemExit(result.returncode)


@app.local_entrypoint()
def main(
    gpu: str = "B200",
    policy_path: str = "lerobot/pi05_libero_finetuned",
    checkpoint_dir: str = "/vol/outputs/transcoders/pi05_libero/multi_ep0-4_100ff_b4_expansion16_lambda1e-4",
    checkpoint_glob: str = "step_*.pt",
    output_file: str = "/vol/outputs/transcoders/pi05_libero/multi_ep0-4_100ff_b4_expansion16_lambda1e-4/heldout_eval_metrics.jsonl",
    val_episodes: str | None = None,
    test_episodes: str | None = None,
    eval_feed_forwards: int = 5,
    batch_size: int = 4,
    num_workers: int = 0,
    collection_mode: str = "random-timestep",
    num_inference_steps: int | None = None,
    policy_dtype: str = "bfloat16",
    transcoder_batch_size: int = 4096,
    eval_buffer_capacity: int = 0,
    lambda_l1: float | None = None,
    local_files_only: bool = False,
    no_progress: bool = False,
) -> None:
    eval_args: list[str] = []
    _append_arg(eval_args, "--policy-path", policy_path)
    _append_arg(eval_args, "--checkpoint-dir", checkpoint_dir)
    _append_arg(eval_args, "--checkpoint-glob", checkpoint_glob)
    _append_arg(eval_args, "--output-file", output_file)
    _append_arg(eval_args, "--val-episodes", val_episodes)
    _append_arg(eval_args, "--test-episodes", test_episodes)
    _append_arg(eval_args, "--eval-feed-forwards", eval_feed_forwards)
    _append_arg(eval_args, "--batch-size", batch_size)
    _append_arg(eval_args, "--num-workers", num_workers)
    _append_arg(eval_args, "--collection-mode", collection_mode)
    _append_arg(eval_args, "--num-inference-steps", num_inference_steps)
    _append_arg(eval_args, "--device", "cuda")
    _append_arg(eval_args, "--policy-dtype", policy_dtype)
    _append_arg(eval_args, "--transcoder-batch-size", transcoder_batch_size)
    _append_arg(eval_args, "--eval-buffer-capacity", eval_buffer_capacity)
    _append_arg(eval_args, "--lambda-l1", lambda_l1)
    if local_files_only:
        eval_args.append("--local-files-only")
    if no_progress:
        eval_args.append("--no-progress")

    gpu_spec = _normalized_gpu(gpu)
    print(f"Launching Modal checkpoint eval on gpu={gpu_spec}")
    print(f"Volume: pi05-libero-data mounted at {VOLUME_MOUNT}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Output file: {output_file}")
    print("Remote command:")
    print("$ " + shlex.join([f"{REMOTE_REPO}/.venv/bin/python", "scripts/eval_pi05_transcoder_checkpoints.py", *eval_args]))

    call = eval_remote.with_options(gpu=gpu_spec).spawn(eval_args)
    print(f"Spawned Modal call: {call.object_id}")
    print("Use `modal run --detach ...` for laptop-disconnect-safe long runs.")
    call.get()

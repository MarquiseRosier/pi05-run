#!/usr/bin/env python
"""Launch Pi0.5 transcoder feature discovery on Modal.

Example:
    modal run --detach scripts/modal_collect_pi05_transcoder_features.py --gpu B200 --max-batches 10
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import modal


APP_NAME = "pi05-transcoder-feature-discovery"
REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_REPO = "/workspace/pi05-run"
VOLUME_MOUNT = "/vol"
HF_CACHE_DIR = f"{VOLUME_MOUNT}/hf_cache"
DEFAULT_CHECKPOINT = (
    f"{VOLUME_MOUNT}/outputs/transcoders/pi05_libero/"
    "allframes_80-10-10_epoch1_b8_exp16_latest_lambda1e-4/step_027233.pt"
)
DEFAULT_OUTPUT_DIR = (
    f"{VOLUME_MOUNT}/outputs/features/pi05_libero/"
    "train_80_top20_exp16_lambda1e-4"
)


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
    if isinstance(value, str) and value == "":
        return
    args.extend([name, str(value)])


@app.function(
    volumes={VOLUME_MOUNT: pi05_volume},
    secrets=[hf_secret],
    timeout=24 * 60 * 60,
)
def collect_remote(collect_args: list[str]) -> None:
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

    cmd = [f"{REMOTE_REPO}/.venv/bin/python", "scripts/collect_pi05_transcoder_features.py", *collect_args]
    print("$ " + shlex.join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=REMOTE_REPO, env=env, check=False)
    pi05_volume.commit()
    if result.returncode != 0:
        raise SystemExit(result.returncode)


@app.local_entrypoint()
def main(
    gpu: str = "B200",
    policy_path: str = "lerobot/pi05_libero_finetuned",
    checkpoint: str = DEFAULT_CHECKPOINT,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    episodes: str | None = None,
    episode_split: str | None = "80,10,10",
    split: str = "train",
    episode_split_seed: int = 0,
    batch_size: int = 8,
    num_workers: int = 0,
    collection_mode: str = "inference",
    num_inference_steps: int | None = 10,
    noise_samples: int = 1,
    top_k: int = 20,
    firing_threshold: float = 1e-6,
    top_m_active: int = 100,
    max_batches: int | None = None,
    policy_dtype: str = "bfloat16",
    local_files_only: bool = False,
    plan_only: bool = False,
    no_progress: bool = False,
) -> None:
    collect_args: list[str] = []
    _append_arg(collect_args, "--policy-path", policy_path)
    _append_arg(collect_args, "--checkpoint", checkpoint)
    _append_arg(collect_args, "--output-dir", output_dir)
    _append_arg(collect_args, "--episodes", episodes)
    _append_arg(collect_args, "--episode-split", episode_split)
    _append_arg(collect_args, "--split", split)
    _append_arg(collect_args, "--episode-split-seed", episode_split_seed)
    _append_arg(collect_args, "--batch-size", batch_size)
    _append_arg(collect_args, "--num-workers", num_workers)
    _append_arg(collect_args, "--collection-mode", collection_mode)
    _append_arg(collect_args, "--num-inference-steps", num_inference_steps)
    _append_arg(collect_args, "--noise-samples", noise_samples)
    _append_arg(collect_args, "--top-k", top_k)
    _append_arg(collect_args, "--firing-threshold", firing_threshold)
    _append_arg(collect_args, "--top-m-active", top_m_active)
    _append_arg(collect_args, "--max-batches", max_batches)
    _append_arg(collect_args, "--device", "cuda")
    _append_arg(collect_args, "--policy-dtype", policy_dtype)
    if local_files_only:
        collect_args.append("--local-files-only")
    if plan_only:
        collect_args.append("--plan-only")
    if no_progress:
        collect_args.append("--no-progress")

    gpu_spec = _normalized_gpu(gpu)
    print(f"Launching Modal feature discovery on gpu={gpu_spec}")
    print(f"Volume: pi05-libero-data mounted at {VOLUME_MOUNT}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Output dir: {output_dir}")
    print("Remote command:")
    print("$ " + shlex.join([f"{REMOTE_REPO}/.venv/bin/python", "scripts/collect_pi05_transcoder_features.py", *collect_args]))

    call = collect_remote.with_options(gpu=gpu_spec).spawn(collect_args)
    print(f"Spawned Modal call: {call.object_id}")
    print("Use `modal run --detach ...` for laptop-disconnect-safe long runs.")
    call.get()

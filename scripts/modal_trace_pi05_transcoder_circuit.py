#!/usr/bin/env python
"""Launch Pi0.5 transcoder circuit tracing on Modal.

Example:
    modal run --detach scripts/modal_trace_pi05_transcoder_circuit.py --gpu B200 --target L12:tau1:F7584
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import modal


APP_NAME = "pi05-transcoder-circuit-tracing"
REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_REPO = "/workspace/pi05-run"
VOLUME_MOUNT = "/vol"
HF_CACHE_DIR = f"{VOLUME_MOUNT}/hf_cache"
DEFAULT_CHECKPOINT = (
    f"{VOLUME_MOUNT}/outputs/transcoders/pi05_libero/"
    "allframes_80-10-10_epoch1_b8_exp16_latest_lambda1e-4/step_027233.pt"
)
DEFAULT_FEATURE_DIR = (
    f"{VOLUME_MOUNT}/outputs/features/pi05_libero/"
    "train_80_top20_exp16_lambda1e-4"
)
DEFAULT_OUTPUT_DIR = f"{VOLUME_MOUNT}/outputs/circuits/pi05_libero"


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
def trace_remote(trace_args: list[str]) -> None:
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

    cmd = [f"{REMOTE_REPO}/.venv/bin/python", "scripts/trace_pi05_transcoder_circuit.py", *trace_args]
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
    feature_dir: str = DEFAULT_FEATURE_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    target: str = "L12:tau1:F7584",
    top_examples: int = 20,
    trace_mode: str = "diffract-frontier",
    parents_per_node: int = 2,
    max_depth: int = 3,
    max_nodes: int = 100,
    expansion_batch_size: int = 10,
    min_attribution: float = 1e-4,
    node_cumulative_threshold: float = 0.8,
    edge_cumulative_threshold: float = 0.98,
    first_layer: int = 0,
    source_policy: str = "all-earlier",
    edge_score_metric: str = "mean_abs",
    example_top_m: int | None = None,
    episodes: str | None = None,
    batch_size: int = 1,
    num_workers: int = 0,
    num_inference_steps: int = 10,
    policy_dtype: str = "bfloat16",
    local_files_only: bool = False,
    plan_only: bool = False,
    no_progress: bool = False,
) -> None:
    trace_output_dir = str(Path(output_dir) / target.replace(":", "_"))
    trace_args: list[str] = []
    _append_arg(trace_args, "--policy-path", policy_path)
    _append_arg(trace_args, "--checkpoint", checkpoint)
    _append_arg(trace_args, "--feature-dir", feature_dir)
    _append_arg(trace_args, "--output-dir", trace_output_dir)
    _append_arg(trace_args, "--target", target)
    _append_arg(trace_args, "--top-examples", top_examples)
    _append_arg(trace_args, "--trace-mode", trace_mode)
    _append_arg(trace_args, "--parents-per-node", parents_per_node)
    _append_arg(trace_args, "--max-depth", max_depth)
    _append_arg(trace_args, "--max-nodes", max_nodes)
    _append_arg(trace_args, "--expansion-batch-size", expansion_batch_size)
    _append_arg(trace_args, "--min-attribution", min_attribution)
    _append_arg(trace_args, "--node-cumulative-threshold", node_cumulative_threshold)
    _append_arg(trace_args, "--edge-cumulative-threshold", edge_cumulative_threshold)
    _append_arg(trace_args, "--first-layer", first_layer)
    _append_arg(trace_args, "--source-policy", source_policy)
    _append_arg(trace_args, "--edge-score-metric", edge_score_metric)
    _append_arg(trace_args, "--example-top-m", example_top_m)
    _append_arg(trace_args, "--episodes", episodes)
    _append_arg(trace_args, "--batch-size", batch_size)
    _append_arg(trace_args, "--num-workers", num_workers)
    _append_arg(trace_args, "--num-inference-steps", num_inference_steps)
    _append_arg(trace_args, "--device", "cuda")
    _append_arg(trace_args, "--policy-dtype", policy_dtype)
    if local_files_only:
        trace_args.append("--local-files-only")
    if plan_only:
        trace_args.append("--plan-only")
    if no_progress:
        trace_args.append("--no-progress")

    gpu_spec = _normalized_gpu(gpu)
    print(f"Launching Modal circuit tracing on gpu={gpu_spec}")
    print(f"Volume: pi05-libero-data mounted at {VOLUME_MOUNT}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Feature dir: {feature_dir}")
    print(f"Output dir: {trace_output_dir}")
    print("Remote command:")
    print("$ " + shlex.join([f"{REMOTE_REPO}/.venv/bin/python", "scripts/trace_pi05_transcoder_circuit.py", *trace_args]))

    call = trace_remote.with_options(gpu=gpu_spec).spawn(trace_args)
    print(f"Spawned Modal call: {call.object_id}")
    print("Use `modal run --detach ...` for laptop-disconnect-safe long runs.")
    call.get()

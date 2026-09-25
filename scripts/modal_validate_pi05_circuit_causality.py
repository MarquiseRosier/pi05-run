#!/usr/bin/env python
"""Launch Pi0.5 transcoder circuit causal validation on Modal.

Example:
    modal run scripts/modal_validate_pi05_circuit_causality.py \
      --graph-json /vol/outputs/circuits/pi05_libero_compact_min003_node06_edge09/L11_tau0.7_F9970/graph.json
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import modal


APP_NAME = "pi05-transcoder-causal-validation"
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
    "pilot_ep0-49_inference10_top20"
)
DEFAULT_GRAPH_JSON = (
    f"{VOLUME_MOUNT}/outputs/circuits/pi05_libero_compact_min003_node06_edge09/"
    "L11_tau0.7_F9970/graph.json"
)
DEFAULT_OUTPUT_DIR = (
    f"{VOLUME_MOUNT}/outputs/case_studies/pi05_libero/"
    "f9970_causal_validation_positive_ablation"
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
def validate_remote(validate_args: list[str]) -> None:
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

    cmd = [f"{REMOTE_REPO}/.venv/bin/python", "scripts/validate_pi05_circuit_causality.py", *validate_args]
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
    graph_json: str = DEFAULT_GRAPH_JSON,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    target: str | None = None,
    example_source: str = "top-positive",
    top_examples: int = 20,
    num_examples: int = 20,
    scan_batch_size: int = 8,
    max_scan_batches: int | None = None,
    disable_zero_score_early_stop: bool = False,
    hard_task_queries: str | None = None,
    hard_task_max_per_query: int = 0,
    num_inference_steps: int = 10,
    random_feature_controls: int = 0,
    random_circuit_controls: int = 0,
    random_control_seed: int = 0,
    episodes: str | None = None,
    batch_size: int = 1,
    num_workers: int = 0,
    policy_dtype: str = "bfloat16",
    local_files_only: bool = False,
    plan_only: bool = False,
    no_progress: bool = False,
) -> None:
    validate_args: list[str] = []
    _append_arg(validate_args, "--policy-path", policy_path)
    _append_arg(validate_args, "--checkpoint", checkpoint)
    _append_arg(validate_args, "--feature-dir", feature_dir)
    _append_arg(validate_args, "--graph-json", graph_json)
    _append_arg(validate_args, "--output-dir", output_dir)
    _append_arg(validate_args, "--target", target)
    _append_arg(validate_args, "--example-source", example_source)
    _append_arg(validate_args, "--top-examples", top_examples)
    _append_arg(validate_args, "--num-examples", num_examples)
    _append_arg(validate_args, "--scan-batch-size", scan_batch_size)
    _append_arg(validate_args, "--max-scan-batches", max_scan_batches)
    _append_arg(validate_args, "--hard-task-queries", hard_task_queries)
    _append_arg(validate_args, "--hard-task-max-per-query", hard_task_max_per_query)
    _append_arg(validate_args, "--num-inference-steps", num_inference_steps)
    _append_arg(validate_args, "--random-feature-controls", random_feature_controls)
    _append_arg(validate_args, "--random-circuit-controls", random_circuit_controls)
    _append_arg(validate_args, "--random-control-seed", random_control_seed)
    _append_arg(validate_args, "--episodes", episodes)
    _append_arg(validate_args, "--batch-size", batch_size)
    _append_arg(validate_args, "--num-workers", num_workers)
    _append_arg(validate_args, "--device", "cuda")
    _append_arg(validate_args, "--policy-dtype", policy_dtype)
    if local_files_only:
        validate_args.append("--local-files-only")
    if plan_only:
        validate_args.append("--plan-only")
    if no_progress:
        validate_args.append("--no-progress")
    if disable_zero_score_early_stop:
        validate_args.append("--disable-zero-score-early-stop")

    gpu_spec = _normalized_gpu(gpu)
    print(f"Launching Modal causal validation on gpu={gpu_spec}")
    print(f"Volume: pi05-libero-data mounted at {VOLUME_MOUNT}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Feature dir: {feature_dir}")
    print(f"Graph: {graph_json}")
    print(f"Output dir: {output_dir}")
    print("Remote command:")
    print(
        "$ "
        + shlex.join([f"{REMOTE_REPO}/.venv/bin/python", "scripts/validate_pi05_circuit_causality.py", *validate_args])
    )

    call = validate_remote.with_options(gpu=gpu_spec).spawn(validate_args)
    print(f"Spawned Modal call: {call.object_id}")
    print("Use `modal run --detach ...` for laptop-disconnect-safe long runs.")
    call.get()

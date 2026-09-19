#!/usr/bin/env python
"""Evaluate Pi0.5 transcoder replacement error on identical observations.

Closed-loop rollout success tells us whether replacement still works in the
simulator. This script measures a more paired quantity: for the same LIBERO
observations and the same diffusion RNG state, compare action chunks from:

1. probe mode: original Pi0.5 MLP outputs, trained transcoders observed only
2. replace mode: action-expert MLP outputs substituted by trained transcoders

The output estimates the action error added by replacement before simulator
feedback causes trajectories to diverge.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from lerobot.datasets.factory import make_dataset
from lerobot.policies import make_policy

from pi05_mi.patch_pi05 import Pi05TranscoderContext, install_pi05_action_expert_wrappers
from pi05_mi.transcoders import TimeConditionedTranscoder, TimeConditionedTranscoderConfig
from train_pi05_transcoders import (
    DEFAULT_POLICY_PATH,
    _config_with_episodes,
    _configure_train_config,
    _make_dataloader,
    _make_preprocessor,
    _parse_episode_ids,
    _prepare_raw_batch,
    patch_pi05_checkpoint_key_compat,
    patch_transformers_causal_mask_compat,
    resolve_device,
    resolve_policy_dtype,
)


DEFAULT_OUTPUT_DIR = Path("outputs/eval/pi05_libero/action-equivalence")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", default=DEFAULT_POLICY_PATH)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--episodes", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--n-action-steps", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--policy-dtype", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _load_transcoders(checkpoint_path: Path, *, device: torch.device) -> dict[str, TimeConditionedTranscoder]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    transcoders: dict[str, TimeConditionedTranscoder] = {}
    for name, raw_config in checkpoint["configs"].items():
        config = TimeConditionedTranscoderConfig(**raw_config)
        transcoder = TimeConditionedTranscoder(config)
        transcoder.load_state_dict(checkpoint["state_dicts"][name])
        transcoder.to(device=device, dtype=torch.float32)
        transcoder.eval()
        for parameter in transcoder.parameters():
            parameter.requires_grad_(False)
        transcoders[name] = transcoder
    return transcoders


def _freeze_policy(policy: torch.nn.Module) -> None:
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)


def _capture_rng(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {"cpu": torch.random.get_rng_state()}
    if device.type == "cuda" and torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any], device: torch.device) -> None:
    torch.random.set_rng_state(state["cpu"])
    if device.type == "cuda" and torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def _compare_one(
    *,
    original: np.ndarray,
    replace: np.ndarray,
    batch_index: int,
    item_index: int,
    n_action_steps: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    steps = min(original.shape[0], replace.shape[0])
    dims = min(original.shape[1], replace.shape[1])
    original = original[:steps, :dims]
    replace = replace[:steps, :dims]
    diff = replace - original
    flat_original = original.reshape(-1)
    flat_replace = replace.reshape(-1)
    denom = float(np.linalg.norm(flat_original) * np.linalg.norm(flat_replace))
    executed_steps = min(n_action_steps, steps)
    executed_diff = diff[:executed_steps]
    per_dim_rmse = np.sqrt(np.mean(diff**2, axis=0))
    row = {
        "batch_index": batch_index,
        "item_index": item_index,
        "steps": steps,
        "dims": dims,
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "mae": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "cosine": None if denom == 0 else float(np.dot(flat_original, flat_replace) / denom),
        "executed_rmse": float(np.sqrt(np.mean(executed_diff**2))),
        "executed_mae": float(np.mean(np.abs(executed_diff))),
        "executed_max_abs": float(np.max(np.abs(executed_diff))),
        "original_norm_mean": float(np.mean(np.linalg.norm(original, axis=-1))),
        "replace_norm_mean": float(np.mean(np.linalg.norm(replace, axis=-1))),
        "norm_delta_mean": float(
            np.mean(np.linalg.norm(replace, axis=-1) - np.linalg.norm(original, axis=-1))
        ),
    }
    for dim, value in enumerate(per_dim_rmse.tolist()):
        row[f"dim_{dim}_rmse"] = float(value)
    row.update(metadata)
    return row


def _metadata_for_batch(raw_batch: dict[str, Any], index: int) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in ["episode_index", "frame_index", "timestamp", "task_index"]:
        value = raw_batch.get(key)
        if value is None:
            continue
        try:
            if hasattr(value, "detach"):
                value = value.detach().cpu()
            if hasattr(value, "tolist"):
                value = value.tolist()
            if isinstance(value, list):
                metadata[key] = value[index] if index < len(value) else None
            else:
                metadata[key] = value
        except Exception:
            metadata[key] = str(value)
    task = raw_batch.get("task")
    if isinstance(task, (list, tuple)) and index < len(task):
        metadata["task"] = task[index]
    elif isinstance(task, str):
        metadata["task"] = task
    return metadata


def _mean_ci(values: list[float]) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": None, "std": None, "ci95_low": None, "ci95_high": None}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    half = 1.959963984540054 * std / math.sqrt(arr.size) if arr.size > 1 else 0.0
    return {
        "n": int(arr.size),
        "mean": mean,
        "std": std,
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.9)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(arr.max()),
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"pairs": len(rows)}
    for key in ["rmse", "mae", "max_abs", "cosine", "executed_rmse", "executed_mae", "executed_max_abs"]:
        summary[key] = _mean_ci([float(row[key]) for row in rows if row.get(key) is not None])
    thresholds = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
    for threshold in thresholds:
        values = [float(row["executed_rmse"]) for row in rows if row.get("executed_rmse") is not None]
        summary[f"executed_rmse_le_{threshold:g}"] = None if not values else sum(v <= threshold for v in values) / len(values)
    dim_keys = sorted({key for row in rows for key in row if key.startswith("dim_") and key.endswith("_rmse")})
    summary["per_dim_rmse"] = {key: _mean_ci([float(row[key]) for row in rows]) for key in dim_keys}
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return str(value)


def main() -> None:
    patch_transformers_causal_mask_compat()
    patch_pi05_checkpoint_key_compat()
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("--max-batches must be positive when provided")

    device = resolve_device(args.device)
    args.resolved_device = device
    args.resolved_policy_dtype = resolve_policy_dtype(args.policy_dtype, device)
    torch.manual_seed(args.seed)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = _configure_train_config(args, episodes=None)
    cfg = _config_with_episodes(cfg, args.episodes)
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.policy.device = str(device)
    cfg.policy.dtype = args.resolved_policy_dtype
    cfg.policy.pretrained_path = Path(args.policy_path)
    cfg.policy.compile_model = False
    cfg.policy.gradient_checkpointing = False

    print(f"loading dataset {cfg.dataset.repo_id} episodes={_parse_episode_ids(args.episodes)}", flush=True)
    dataset = make_dataset(cfg)
    dataloader = _make_dataloader(cfg, dataset)
    planned_batches = len(dataloader) if args.max_batches is None else min(args.max_batches, len(dataloader))
    print(f"planned paired batches={planned_batches} batch_size={args.batch_size}", flush=True)

    print("loading preprocessor/tokenizer", flush=True)
    preprocessor = _make_preprocessor(cfg, dataset, args.policy_path)
    print("loading frozen Pi0.5 policy", flush=True)
    policy = make_policy(cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    _freeze_policy(policy)

    print(f"loading transcoders from {args.checkpoint}", flush=True)
    transcoders = _load_transcoders(args.checkpoint, device=device)
    context = Pi05TranscoderContext(
        mode="probe",
        capture_records=False,
        capture_latents=False,
        store_latent_summaries=False,
    )
    _context, wrapped_names = install_pi05_action_expert_wrappers(
        policy,
        context=context,
        transcoders=transcoders,
        mode="probe",
    )
    print(f"wrapped action-expert MLPs={len(wrapped_names)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    started_at = time.time()
    iterator = tqdm(
        enumerate(dataloader, start=1),
        total=planned_batches,
        desc="paired action equivalence",
        disable=args.no_progress,
    )

    for batch_index, raw_batch in iterator:
        if batch_index > planned_batches:
            break
        raw_batch = _prepare_raw_batch(raw_batch, dataset.meta.camera_keys)
        batch = preprocessor(raw_batch)

        rng_state = _capture_rng(device)
        context.mode = "probe"
        context.clear_records()
        with torch.inference_mode():
            original_actions = policy.predict_action_chunk(batch, num_steps=args.num_inference_steps)

        _restore_rng(rng_state, device)
        context.mode = "replace"
        context.clear_records()
        with torch.inference_mode():
            replace_actions = policy.predict_action_chunk(batch, num_steps=args.num_inference_steps)

        original_np = _tensor_to_numpy(original_actions)
        replace_np = _tensor_to_numpy(replace_actions)
        batch_size = min(original_np.shape[0], replace_np.shape[0])
        for item_index in range(batch_size):
            rows.append(
                _compare_one(
                    original=original_np[item_index],
                    replace=replace_np[item_index],
                    batch_index=batch_index,
                    item_index=item_index,
                    n_action_steps=args.n_action_steps,
                    metadata=_metadata_for_batch(raw_batch, item_index),
                )
            )
        iterator.set_postfix(pairs=len(rows))

    summary = {
        "policy_path": args.policy_path,
        "checkpoint": str(args.checkpoint),
        "episodes": _parse_episode_ids(args.episodes),
        "batch_size": args.batch_size,
        "planned_batches": planned_batches,
        "num_inference_steps": args.num_inference_steps,
        "n_action_steps": args.n_action_steps,
        "seed": args.seed,
        "elapsed_s": time.time() - started_at,
        "wrapped_layers": len(wrapped_names),
        "metrics": _aggregate(rows),
        "interpretation": (
            "These are paired same-observation action errors between original/probe and replace modes. "
            "They estimate replacement error before simulator feedback causes closed-loop divergence."
        ),
    }

    _write_csv(args.output_dir / "paired_action_metrics.csv", rows)
    (args.output_dir / "action_equivalence_summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(json.dumps(summary["metrics"], indent=2, default=_json_default), flush=True)
    print("Saved paired action metrics to", args.output_dir, flush=True)


if __name__ == "__main__":
    main()

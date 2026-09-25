#!/usr/bin/env python
"""Rank Pi0.5 transcoder features by fast/slow action-chunk behavior."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from lerobot.datasets.factory import make_dataset
from lerobot.policies import make_policy
from lerobot.utils.constants import ACTION

from train_pi05_transcoders import (
    DEFAULT_POLICY_PATH,
    _config_with_episodes,
    _configure_train_config,
    _episode_summary,
    _make_dataloader,
    _make_preprocessor,
    _parse_episode_ids,
    _prepare_raw_batch,
    patch_pi05_checkpoint_key_compat,
    patch_transformers_causal_mask_compat,
    resolve_device,
    resolve_policy_dtype,
)


DEFAULT_FEATURE_DIR = Path("outputs/features/pi05_libero/pilot_ep0-49_inference10_top20")
DEFAULT_OUTPUT_DIR = Path("outputs/features/pi05_libero/speed_behavior")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", default=DEFAULT_POLICY_PATH)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--episodes", default=None, help="Override feature-dir episodes. Defaults to feature config.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--top-candidates", type=int, default=100)
    parser.add_argument("--top-examples", type=int, default=20)
    parser.add_argument("--min-frequency", type=float, default=0.0)
    parser.add_argument("--min-max-score", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--policy-dtype", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _freeze(module: torch.nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _feature_dir_episodes(feature_dir: Path) -> str | None:
    config_path = feature_dir / "config.json"
    if not config_path.exists():
        return None
    with config_path.open() as f:
        config = json.load(f)
    episodes = config.get("episodes")
    if episodes is None:
        split_info = config.get("split_info") or {}
        selected = split_info.get("train_episodes") if split_info.get("split") == "train" else None
        episodes = selected
    if episodes is None:
        return None
    return ",".join(str(int(episode)) for episode in episodes)


def _read_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            rows[int(row["observation_id"])] = row
    return rows


def _action_speed_metrics(actions: torch.Tensor, *, action_dim: int) -> dict[str, torch.Tensor]:
    action_dim = min(int(action_dim), int(actions.shape[-1]))
    xyz_dim = min(3, action_dim)
    xyz = actions[..., :xyz_dim].float()
    per_step = xyz.pow(2).sum(dim=-1).sqrt()
    return {
        "speed_mean": per_step.mean(dim=-1),
        "speed_sum": per_step.sum(dim=-1),
        "speed_max": per_step.max(dim=-1).values,
    }


def _load_feature_artifacts(feature_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    topk_path = feature_dir / "feature_topk.pt"
    stats_path = feature_dir / "feature_stats.pt"
    if not topk_path.exists() or not stats_path.exists():
        raise FileNotFoundError(f"Expected feature_topk.pt and feature_stats.pt in {feature_dir}")
    return (
        torch.load(topk_path, map_location="cpu", weights_only=False),
        torch.load(stats_path, map_location="cpu", weights_only=False),
    )


def _layer_name_for_index(topk_payload: dict[str, Any], layer: int) -> str:
    for name in topk_payload["layer_names"]:
        if int(topk_payload["layer_indices"][name]) == int(layer):
            return name
    raise KeyError(f"No layer {layer} in feature artifact")


def _summarize_task_counts(observation_ids: torch.Tensor, observations: dict[int, dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for observation_id in observation_ids.tolist():
        if int(observation_id) < 0:
            continue
        task = str(observations.get(int(observation_id), {}).get("task", ""))
        if len(task) > 80:
            task = task[:77] + "..."
        counts[task] = counts.get(task, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return " | ".join(f"{count}x {task}" for task, count in ordered[:4])


def _rank_speed_features(
    *,
    feature_dir: Path,
    output_dir: Path,
    top_candidates: int,
    top_examples: int,
    min_frequency: float,
    min_max_score: float,
) -> None:
    observations = _read_jsonl(feature_dir / "observations.jsonl")
    speed_rows = _read_jsonl(output_dir / "behavior_scores.jsonl")
    speed_by_observation = {
        observation_id: float(row["speed_mean"])
        for observation_id, row in speed_rows.items()
        if math.isfinite(float(row["speed_mean"]))
    }
    global_speeds = torch.tensor(list(speed_by_observation.values()), dtype=torch.float32)
    speed_mean = float(global_speeds.mean())
    speed_std = float(global_speeds.std(unbiased=False).clamp_min(1e-8))

    topk_payload, stats_payload = _load_feature_artifacts(feature_dir)
    rows: list[dict[str, Any]] = []
    for name in topk_payload["layer_names"]:
        layer = int(topk_payload["layer_indices"][name])
        for timestep in sorted(topk_payload["topk"][name]):
            store = topk_payload["topk"][name][timestep]
            stats = stats_payload["stats"][name][timestep]
            scores = store["scores"].float()
            obs_ids = store["observation_ids"].long()
            action_positions = store["action_positions"].long()
            valid = torch.isfinite(scores) & (obs_ids >= 0)
            k = min(int(top_examples), scores.shape[1])
            if k <= 0:
                continue
            selected_valid = valid[:, :k]
            selected_obs = obs_ids[:, :k]
            selected_scores = scores[:, :k]
            speed_values = torch.full_like(selected_scores, torch.nan, dtype=torch.float32)
            for feature_row in range(selected_obs.shape[0]):
                for rank_col in range(selected_obs.shape[1]):
                    observation_id = int(selected_obs[feature_row, rank_col])
                    if observation_id in speed_by_observation:
                        speed_values[feature_row, rank_col] = speed_by_observation[observation_id]

            finite_speed = torch.isfinite(speed_values) & selected_valid
            valid_count = finite_speed.sum(dim=1).clamp_min(1)
            top_speed_mean = torch.where(
                finite_speed,
                speed_values,
                torch.zeros_like(speed_values),
            ).sum(dim=1) / valid_count
            top_speed_std = torch.where(
                finite_speed,
                (speed_values - top_speed_mean[:, None]).pow(2),
                torch.zeros_like(speed_values),
            ).sum(dim=1).div(valid_count).sqrt()
            top_score_mean = torch.where(
                selected_valid,
                selected_scores,
                torch.zeros_like(selected_scores),
            ).sum(dim=1) / selected_valid.sum(dim=1).clamp_min(1)
            max_score = torch.where(valid[:, 0], scores[:, 0], torch.zeros_like(scores[:, 0]))
            frequency = stats["firing_frequency"].float()
            activation_std = stats["std"].float()

            speed_z = (top_speed_mean - speed_mean) / speed_std
            activation_reliability = top_score_mean / (activation_std + 1e-6)
            fast_score = speed_z * torch.log1p(top_score_mean.clamp_min(0.0))
            slow_score = (-speed_z) * torch.log1p(top_score_mean.clamp_min(0.0))
            feature_count = int(scores.shape[0])
            for feature in range(feature_count):
                if float(frequency[feature]) < min_frequency or float(max_score[feature]) < min_max_score:
                    continue
                selected_ids = obs_ids[feature, :k]
                selected_positions = action_positions[feature, :k]
                row = {
                    "layer": layer,
                    "layer_name": name,
                    "timestep": float(timestep),
                    "timestep_key": timestep,
                    "feature": feature,
                    "feature_key": f"L{layer:02d}:tau{float(timestep):.4g}:F{feature}",
                    "top_speed_mean": float(top_speed_mean[feature]),
                    "top_speed_std": float(top_speed_std[feature]),
                    "top_speed_z": float(speed_z[feature]),
                    "top_activation_mean": float(top_score_mean[feature]),
                    "max_activation": float(max_score[feature]),
                    "activation_reliability": float(activation_reliability[feature]),
                    "frequency": float(frequency[feature]),
                    "valid_top_examples": int(finite_speed[feature].sum()),
                    "fast_score": float(fast_score[feature]),
                    "slow_score": float(slow_score[feature]),
                    "top_observation_ids": " ".join(str(int(value)) for value in selected_ids.tolist() if int(value) >= 0),
                    "top_action_positions": " ".join(str(int(value)) for value in selected_positions.tolist() if int(value) >= 0),
                    "top_tasks": _summarize_task_counts(selected_ids, observations),
                }
                rows.append(row)

    fast_rows = sorted(rows, key=lambda row: float(row["fast_score"]), reverse=True)[:top_candidates]
    slow_rows = sorted(rows, key=lambda row: float(row["slow_score"]), reverse=True)[:top_candidates]
    for rank, row in enumerate(fast_rows, start=1):
        row["rank"] = rank
        row["direction"] = "fast"
    for rank, row in enumerate(slow_rows, start=1):
        row["rank"] = rank
        row["direction"] = "slow"

    fieldnames = [
        "direction",
        "rank",
        "feature_key",
        "layer",
        "timestep",
        "feature",
        "fast_score",
        "slow_score",
        "top_speed_mean",
        "top_speed_std",
        "top_speed_z",
        "top_activation_mean",
        "max_activation",
        "activation_reliability",
        "frequency",
        "valid_top_examples",
        "top_observation_ids",
        "top_action_positions",
        "top_tasks",
        "layer_name",
        "timestep_key",
    ]
    for name, selected in (("fast_feature_candidates.csv", fast_rows), ("slow_feature_candidates.csv", slow_rows)):
        with (output_dir / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(selected)

    summary = {
        "speed_mean": speed_mean,
        "speed_std": speed_std,
        "observation_count": len(speed_by_observation),
        "top_candidates": top_candidates,
        "top_examples": top_examples,
        "feature_dir": str(feature_dir),
        "best_fast": fast_rows[:10],
        "best_slow": slow_rows[:10],
    }
    (output_dir / "speed_feature_ranking.json").write_text(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    patch_transformers_causal_mask_compat()
    patch_pi05_checkpoint_key_compat()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    episodes = args.episodes if args.episodes is not None else _feature_dir_episodes(args.feature_dir)
    device = resolve_device(args.device)
    policy_dtype = resolve_policy_dtype(args.policy_dtype, device)
    args.resolved_device = device
    args.resolved_policy_dtype = policy_dtype

    cfg = _configure_train_config(args, episodes=None)
    cfg = _config_with_episodes(cfg, episodes)
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.policy.device = str(device)
    cfg.policy.dtype = policy_dtype
    cfg.policy.pretrained_path = Path(args.policy_path)
    cfg.policy.compile_model = False
    cfg.policy.gradient_checkpointing = False

    print(f"loading dataset {cfg.dataset.repo_id} episodes={_episode_summary(_parse_episode_ids(episodes))}", flush=True)
    dataset = make_dataset(cfg)
    dataloader = _make_dataloader(cfg, dataset)
    planned_batches = len(dataloader) if args.max_batches is None else min(args.max_batches, len(dataloader))
    print(f"speed scoring observations={len(dataset)} batches={len(dataloader)} planned_batches={planned_batches}", flush=True)
    if args.plan_only:
        return

    print("loading preprocessor/tokenizer", flush=True)
    preprocessor = _make_preprocessor(cfg, dataset, args.policy_path)
    print("loading frozen Pi0.5 policy weights", flush=True)
    policy = make_policy(cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    _freeze(policy)
    action_dim = int(policy.config.output_features[ACTION].shape[0])

    scores_path = args.output_dir / "behavior_scores.jsonl"
    if scores_path.exists():
        scores_path.unlink()

    observation_id = 0
    started = time.time()
    iterator = tqdm(
        enumerate(dataloader, start=1),
        total=planned_batches,
        desc="speed feed-forwards",
        disable=args.no_progress,
    )
    with scores_path.open("w", buffering=1) as f:
        for batch_index, raw_batch in iterator:
            if batch_index > planned_batches:
                break
            raw_batch = _prepare_raw_batch(raw_batch, dataset.meta.camera_keys)
            batch = preprocessor(raw_batch)
            torch.manual_seed(args.seed + batch_index)
            with torch.no_grad():
                actions = policy.predict_action_chunk(batch, num_steps=args.num_inference_steps)
            metrics = _action_speed_metrics(actions.detach(), action_dim=action_dim)
            batch_size = int(actions.shape[0])
            for row in range(batch_size):
                metadata: dict[str, Any] = {
                    "observation_id": observation_id,
                    "speed_mean": float(metrics["speed_mean"][row].detach().cpu()),
                    "speed_sum": float(metrics["speed_sum"][row].detach().cpu()),
                    "speed_max": float(metrics["speed_max"][row].detach().cpu()),
                    "action_dim": action_dim,
                }
                for key in ("index", "episode_index", "frame_index", "timestamp", "task", "task_index"):
                    if key not in raw_batch:
                        continue
                    value = raw_batch[key]
                    if isinstance(value, torch.Tensor):
                        selected = value[row]
                        metadata[key] = selected.item() if selected.ndim == 0 else selected.detach().cpu().tolist()
                    elif isinstance(value, (list, tuple)):
                        metadata[key] = value[row]
                    else:
                        metadata[key] = value
                f.write(json.dumps(metadata, sort_keys=True) + "\n")
                observation_id += 1
            if args.no_progress:
                print(f"batch={batch_index} observations={observation_id}", flush=True)
            else:
                iterator.set_postfix(observations=observation_id)

    config = {
        "policy_path": args.policy_path,
        "feature_dir": str(args.feature_dir),
        "episodes": _parse_episode_ids(episodes),
        "batch_size": args.batch_size,
        "planned_batches": planned_batches,
        "num_inference_steps": args.num_inference_steps,
        "seed": args.seed,
        "speed_metric": "mean_t_l2_first_3_action_dims",
        "action_dim": action_dim,
        "device": str(device),
        "policy_dtype": policy_dtype,
        "elapsed_s": time.time() - started,
    }
    (args.output_dir / "speed_config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    _rank_speed_features(
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
        top_candidates=args.top_candidates,
        top_examples=args.top_examples,
        min_frequency=args.min_frequency,
        min_max_score=args.min_max_score,
    )
    print(f"saved speed behavior feature ranking to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

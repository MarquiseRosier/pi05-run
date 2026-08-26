#!/usr/bin/env python
"""Evaluate saved Pi0.5 transcoder checkpoints on held-out episodes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from pi05_mi.buffers import MultiLayerActivationBuffer
from pi05_mi.patch_pi05 import Pi05TranscoderContext, WrappedActionExpertMLP, install_pi05_action_expert_wrappers
from pi05_mi.transcoders import TimeConditionedTranscoder, TimeConditionedTranscoderConfig
from train_pi05_transcoders import (
    DEFAULT_POLICY_PATH,
    _collect_records_for_batch,
    _config_with_episodes,
    _configure_train_config,
    _make_dataloader,
    _make_preprocessor,
    _parse_episode_ids,
    patch_pi05_checkpoint_key_compat,
    patch_transformers_causal_mask_compat,
    resolve_device,
    resolve_policy_dtype,
)
from lerobot.datasets.factory import make_dataset
from lerobot.policies import make_policy
from lerobot.utils.utils import cycle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", default=DEFAULT_POLICY_PATH)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-glob", default="step_*.pt")
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--val-episodes", default=None)
    parser.add_argument("--test-episodes", default=None)
    parser.add_argument("--eval-feed-forwards", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--collection-mode",
        choices=("random-timestep", "inference", "training-forward"),
        default="random-timestep",
    )
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--policy-dtype", default="auto")
    parser.add_argument("--transcoder-batch-size", type=int, default=4096)
    parser.add_argument("--eval-buffer-capacity", type=int, default=0)
    parser.add_argument("--lambda-l1", type=float, default=None, help="Override checkpoint lambda_l1.")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _checkpoint_step(path: Path) -> int:
    stem = path.stem
    if stem.startswith("step_"):
        return int(stem.removeprefix("step_"))
    return -1


def _eval_buffer_capacity(args: argparse.Namespace, policy, feed_forwards: int) -> int:
    if args.eval_buffer_capacity > 0:
        return args.eval_buffer_capacity
    chunk_size = int(getattr(policy.model.config, "chunk_size", 50))
    return max(args.transcoder_batch_size, feed_forwards * args.batch_size * chunk_size)


def _build_transcoders_from_checkpoint(
    checkpoint: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, TimeConditionedTranscoder]:
    transcoders: dict[str, TimeConditionedTranscoder] = {}
    for name, raw_config in checkpoint["configs"].items():
        config = TimeConditionedTranscoderConfig(**raw_config)
        transcoder = TimeConditionedTranscoder(config)
        transcoder.load_state_dict(checkpoint["state_dicts"][name])
        transcoder.to(device=device, dtype=torch.float32)
        transcoder.eval()
        transcoders[name] = transcoder
    return transcoders


def _install_or_update_wrappers(
    policy,
    *,
    transcoders: dict[str, TimeConditionedTranscoder],
    context: Pi05TranscoderContext | None,
) -> tuple[Pi05TranscoderContext, list[str]]:
    if context is None:
        return install_pi05_action_expert_wrappers(policy, transcoders=transcoders, mode="train")

    context.mode = "train"
    context.clear_records()
    wrapped_names: list[str] = []
    for module in policy.model.modules():
        if not isinstance(module, WrappedActionExpertMLP):
            continue
        module.context = context
        module.transcoder = transcoders[module.name]
        wrapped_names.append(module.name)
    if not wrapped_names:
        raise RuntimeError("Expected Pi0.5 action-expert wrappers to already be installed")
    return context, wrapped_names


def _evaluate_split(
    *,
    split: str,
    episodes: str,
    checkpoint: dict[str, Any],
    transcoders: dict[str, TimeConditionedTranscoder],
    policy,
    cfg,
    preprocessor,
    context,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    split_cfg = _config_with_episodes(cfg, episodes)
    dataset = make_dataset(split_cfg)
    data_iter = cycle(_make_dataloader(split_cfg, dataset))
    layer_names = list(transcoders)
    d_model = next(iter(transcoders.values())).config.d_model
    buffers = MultiLayerActivationBuffer(
        layer_names=layer_names,
        d_model=d_model,
        capacity_per_layer=_eval_buffer_capacity(args, policy, args.eval_feed_forwards),
    )

    added_records = 0
    for _ in range(args.eval_feed_forwards):
        added_by_layer = _collect_records_for_batch(
            policy=policy,
            raw_batch=next(data_iter),
            dataset=dataset,
            preprocessor=preprocessor,
            context=context,
            buffers=buffers,
            args=args,
        )
        added_records += sum(added_by_layer.values())

    lambda_l1 = checkpoint["lambda_l1"] if args.lambda_l1 is None else args.lambda_l1
    total_weighted_loss = 0.0
    total_weighted_mse = 0.0
    total_weighted_l1 = 0.0
    total_records = 0
    minibatches = 0
    ready_layers = 0

    expected_minibatches = sum(
        (buffers.buffers[name].size + args.transcoder_batch_size - 1) // args.transcoder_batch_size
        for name in layer_names
        if buffers.buffers[name].size > 0
    )
    with tqdm(
        total=expected_minibatches,
        desc=f"{split} checkpoint eval",
        leave=False,
        disable=args.no_progress,
    ) as bar:
        with torch.no_grad():
            for name in layer_names:
                if buffers.buffers[name].size == 0:
                    continue
                ready_layers += 1
                raw_denom = checkpoint["buffer_stats"][name]["variance_denominator"]
                denominator = torch.tensor(raw_denom, device=device, dtype=torch.float32)
                transcoder = transcoders[name]
                for sample in buffers.epoch_batches(
                    name,
                    args.transcoder_batch_size,
                    device=device,
                    dtype=torch.float32,
                    shuffle=False,
                ):
                    loss = transcoder.loss(
                        sample.x,
                        sample.timestep,
                        sample.y,
                        denominator,
                        lambda_l1=lambda_l1,
                    )
                    record_count = sample.x.shape[0]
                    total_weighted_loss += float(loss.total.detach().cpu()) * record_count
                    total_weighted_mse += float(loss.normalized_mse.detach().cpu()) * record_count
                    total_weighted_l1 += float(loss.l1.detach().cpu()) * record_count
                    total_records += record_count
                    minibatches += 1
                    bar.update(1)

    if total_records == 0:
        raise RuntimeError(f"No activation records collected for {split} episodes={episodes}")

    return {
        "split": split,
        "episodes": _parse_episode_ids(episodes),
        "eval_feed_forwards": args.eval_feed_forwards,
        "added_records": added_records,
        "records": total_records,
        "ready_layers": ready_layers,
        "minibatches": minibatches,
        "loss": total_weighted_loss / total_records,
        "normalized_mse": total_weighted_mse / total_records,
        "l1": total_weighted_l1 / total_records,
        "weighted_l1": lambda_l1 * (total_weighted_l1 / total_records),
        "lambda_l1": lambda_l1,
    }


def main() -> None:
    patch_transformers_causal_mask_compat()
    patch_pi05_checkpoint_key_compat()
    args = parse_args()
    device = resolve_device(args.device)
    args.resolved_device = device
    args.resolved_policy_dtype = resolve_policy_dtype(args.policy_dtype, device)

    split_args = {"val": args.val_episodes, "test": args.test_episodes}
    split_args = {name: episodes for name, episodes in split_args.items() if episodes is not None}
    if not split_args:
        raise ValueError("Provide at least one of --val-episodes or --test-episodes")

    cfg = _configure_train_config(args, episodes=None)
    dataset = make_dataset(cfg)
    preprocessor = _make_preprocessor(cfg, dataset, args.policy_path)
    policy = make_policy(cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)

    checkpoints = sorted(args.checkpoint_dir.glob(args.checkpoint_glob), key=_checkpoint_step)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints matched {args.checkpoint_dir / args.checkpoint_glob}")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    if args.output_file.exists():
        args.output_file.unlink()

    started_at = time.time()
    context: Pi05TranscoderContext | None = None
    for checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        transcoders = _build_transcoders_from_checkpoint(checkpoint, device=device)
        context, _wrapped_names = _install_or_update_wrappers(
            policy,
            transcoders=transcoders,
            context=context,
        )
        for split, episodes in split_args.items():
            metrics = _evaluate_split(
                split=split,
                episodes=episodes,
                checkpoint=checkpoint,
                transcoders=transcoders,
                policy=policy,
                cfg=cfg,
                preprocessor=preprocessor,
                context=context,
                args=args,
                device=device,
            )
            row = {
                "checkpoint": str(checkpoint_path),
                "checkpoint_name": checkpoint_path.name,
                "feed_forward": int(checkpoint.get("feed_forward", _checkpoint_step(checkpoint_path))),
                "elapsed_s": time.time() - started_at,
                "collection_mode": args.collection_mode,
                "policy_path": args.policy_path,
                **metrics,
            }
            with args.output_file.open("a") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
            print(
                f"{split} checkpoint={checkpoint_path.name} "
                f"loss={metrics['loss']:.4f} norm_mse={metrics['normalized_mse']:.4f} "
                f"l1={metrics['l1']:.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()

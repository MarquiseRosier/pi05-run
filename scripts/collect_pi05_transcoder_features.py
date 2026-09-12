#!/usr/bin/env python
"""Collect DifFRACT-style Top-K feature activations for Pi0.5 transcoders."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from lerobot.datasets.factory import make_dataset
from lerobot.policies import make_policy

from pi05_mi.feature_discovery import FeatureDiscoveryCollector, FeatureDiscoveryConfig
from pi05_mi.patch_pi05 import Pi05TranscoderContext, install_pi05_action_expert_wrappers
from pi05_mi.transcoders import TimeConditionedTranscoder, TimeConditionedTranscoderConfig
from train_pi05_transcoders import (
    DEFAULT_POLICY_PATH,
    _all_episode_ids,
    _collect_random_timestep_records,
    _config_with_episodes,
    _configure_train_config,
    _episode_summary,
    _format_episode_ids,
    _make_dataloader,
    _make_preprocessor,
    _parse_episode_ids,
    _prepare_raw_batch,
    _split_episode_ids,
    patch_pi05_checkpoint_key_compat,
    patch_transformers_causal_mask_compat,
    resolve_device,
    resolve_policy_dtype,
)


DEFAULT_OUTPUT_DIR = Path("outputs/features/pi05_libero")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", default=DEFAULT_POLICY_PATH)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained transcoder checkpoint, e.g. step_027233.pt.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--episodes", default=None, help="Comma-separated episode ids. Omit for all episodes.")
    parser.add_argument(
        "--episode-split",
        default=None,
        help="Optional train,val,test split percentages, for example `80,10,10`.",
    )
    parser.add_argument(
        "--split",
        choices=("all", "train", "val", "test"),
        default="all",
        help="Which split to collect when --episode-split is provided.",
    )
    parser.add_argument("--episode-split-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--collection-mode",
        choices=("random-timestep", "inference", "training-forward"),
        default="inference",
    )
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument(
        "--noise-samples",
        type=int,
        default=1,
        help="Number of random noise/timestep draws per observation batch for random-timestep collection.",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--firing-threshold", type=float, default=1e-6)
    parser.add_argument(
        "--top-m-active",
        type=int,
        default=0,
        help="Also track how often a feature is among the top M features of an observation. 0 disables it.",
    )
    parser.add_argument("--max-batches", type=int, default=None, help="Smoke-test cap on dataloader batches.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--policy-dtype", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _freeze_policy(policy: torch.nn.Module) -> None:
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)


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


def _selected_episodes(args: argparse.Namespace, cfg) -> tuple[str | None, dict[str, Any]]:
    if not args.episode_split:
        return args.episodes, {"episodes": args.episodes, "episode_split": None, "split": "manual"}

    source_dataset = make_dataset(cfg)
    train, val, test = _split_episode_ids(
        _all_episode_ids(source_dataset),
        args.episode_split,
        seed=args.episode_split_seed,
    )
    split_map = {"train": train, "val": val, "test": test}
    selected = None if args.split == "all" else _format_episode_ids(split_map[args.split])
    return selected, {
        "episode_split": args.episode_split,
        "episode_split_seed": args.episode_split_seed,
        "split": args.split,
        "train_episodes": train,
        "val_episodes": val,
        "test_episodes": test,
    }


def _collect_for_batch(
    *,
    args: argparse.Namespace,
    policy,
    dataset,
    preprocessor,
    context: Pi05TranscoderContext,
    raw_batch: dict[str, Any],
) -> None:
    raw_batch = _prepare_raw_batch(raw_batch, dataset.meta.camera_keys)
    batch = preprocessor(raw_batch)

    with torch.no_grad():
        if args.collection_mode == "random-timestep":
            for _ in range(args.noise_samples):
                context.clear_records()
                _collect_random_timestep_records(policy, batch)
        elif args.collection_mode == "inference":
            context.clear_records()
            policy.predict_action_chunk(batch, num_steps=args.num_inference_steps)
        else:
            context.clear_records()
            policy.forward(batch)


def main() -> None:
    patch_transformers_causal_mask_compat()
    patch_pi05_checkpoint_key_compat()
    args = parse_args()
    if args.noise_samples <= 0:
        raise ValueError(f"--noise-samples must be positive, got {args.noise_samples}")

    device = resolve_device(args.device)
    args.resolved_device = device
    args.resolved_policy_dtype = resolve_policy_dtype(args.policy_dtype, device)

    cfg = _configure_train_config(args, episodes=None)
    episodes, split_info = _selected_episodes(args, cfg)
    cfg = _config_with_episodes(cfg, episodes)
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.policy.device = str(device)
    cfg.policy.dtype = args.resolved_policy_dtype
    cfg.policy.pretrained_path = Path(args.policy_path)
    cfg.policy.compile_model = False
    cfg.policy.gradient_checkpointing = False

    print(f"loading dataset {cfg.dataset.repo_id} episodes={_episode_summary(_parse_episode_ids(episodes))}", flush=True)
    dataset = make_dataset(cfg)
    dataloader = _make_dataloader(cfg, dataset)
    planned_batches = len(dataloader) if args.max_batches is None else min(args.max_batches, len(dataloader))
    print(
        f"feature discovery frames={len(dataset)} batches={len(dataloader)} planned_batches={planned_batches} "
        f"batch_size={args.batch_size}",
        flush=True,
    )
    if args.plan_only:
        print("plan-only requested; exiting before model load", flush=True)
        return

    print("loading preprocessor/tokenizer", flush=True)
    preprocessor = _make_preprocessor(cfg, dataset, args.policy_path)
    print("loading frozen Pi0.5 policy weights", flush=True)
    policy = make_policy(cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    _freeze_policy(policy)

    print(f"loading transcoders from {args.checkpoint}", flush=True)
    transcoders = _load_transcoders(args.checkpoint, device=device)
    if not transcoders:
        raise RuntimeError("Checkpoint contained no transcoders")
    d_features = {transcoder.config.latent_dim for transcoder in transcoders.values()}
    if len(d_features) != 1:
        raise ValueError(f"Expected all transcoders to share latent dim, got {sorted(d_features)}")
    layer_indices = {name: int(name.split(".layers.", 1)[1].split(".", 1)[0]) for name in transcoders}
    layer_names = sorted(transcoders, key=lambda name: layer_indices[name])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    observations_path = args.output_dir / "observations.jsonl"
    collector = FeatureDiscoveryCollector(
        layer_names=layer_names,
        layer_indices=layer_indices,
        d_features=d_features.pop(),
        config=FeatureDiscoveryConfig(
            top_k=args.top_k,
            firing_threshold=args.firing_threshold,
            top_m_active=args.top_m_active,
        ),
        observations_path=observations_path,
        camera_keys=dataset.meta.camera_keys,
    )
    context = Pi05TranscoderContext(
        mode="probe",
        capture_records=False,
        capture_latents=True,
        latent_top_k=0,
        save_full_latents=False,
        latent_callback=collector.observe_latent,
        store_latent_summaries=False,
    )
    _context, wrapped_names = install_pi05_action_expert_wrappers(
        policy,
        context=context,
        transcoders=transcoders,
        mode="probe",
    )
    if sorted(wrapped_names) != sorted(layer_names):
        print(f"warning: checkpoint transcoders={len(layer_names)} wrapped_mlps={len(wrapped_names)}", flush=True)

    metrics_path = args.output_dir / "collection_metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    started_at = time.time()
    batch_iter = tqdm(
        enumerate(dataloader, start=1),
        total=planned_batches,
        desc="feature feed-forwards",
        disable=args.no_progress,
    )
    try:
        for batch_index, raw_batch in batch_iter:
            if batch_index > planned_batches:
                break
            batch_size = collector.begin_batch(raw_batch)
            _collect_for_batch(
                args=args,
                policy=policy,
                dataset=dataset,
                preprocessor=preprocessor,
                context=context,
                raw_batch=raw_batch,
            )
            updated = collector.end_batch()
            context.clear_records()
            row = {
                "batch_index": batch_index,
                "elapsed_s": time.time() - started_at,
                "observations": collector.observation_count,
                "batch_size": batch_size,
                "layers_updated": len(updated),
                "collection_mode": args.collection_mode,
                "noise_samples": args.noise_samples,
            }
            with metrics_path.open("a") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
            if args.no_progress:
                print(
                    f"batch={batch_index} observations={collector.observation_count} layers_updated={len(updated)}",
                    flush=True,
                )
            else:
                batch_iter.set_postfix(observations=collector.observation_count, layers=len(updated))
    finally:
        collector.close()

    collector.save(
        args.output_dir,
        extra_config={
            "policy_path": args.policy_path,
            "checkpoint": str(args.checkpoint),
            "dataset_repo_id": cfg.dataset.repo_id,
            "episodes": _parse_episode_ids(episodes),
            "split_info": split_info,
            "batch_size": args.batch_size,
            "planned_batches": planned_batches,
            "collection_mode": args.collection_mode,
            "noise_samples": args.noise_samples,
            "num_inference_steps": args.num_inference_steps,
            "device": str(device),
            "policy_dtype": args.resolved_policy_dtype,
        },
    )
    print(
        f"saved feature discovery artifacts to {args.output_dir} "
        f"observations={collector.observation_count} layers={len(layer_names)}",
        flush=True,
    )


if __name__ == "__main__":
    main()

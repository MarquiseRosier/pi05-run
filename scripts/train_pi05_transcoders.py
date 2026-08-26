#!/usr/bin/env python
"""Train DifFRACT-style transcoders for Pi0.5 action-expert MLPs."""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import tempfile
import time
from dataclasses import asdict
from math import ceil
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

# Importing the Pi0.5 config registers the "pi05" policy choice with LeRobot.
import lerobot.policies.pi05.configuration_pi05  # noqa: F401
from huggingface_hub import hf_hub_download
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets import EpisodeAwareSampler
from lerobot.datasets.factory import make_dataset
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks, prepare_attention_masks_4d
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.utils.utils import cycle

from pi05_mi.buffers import MultiLayerActivationBuffer
from pi05_mi.patch_pi05 import install_pi05_action_expert_wrappers, list_pi05_action_expert_mlp_targets
from pi05_mi.transcoders import TimeConditionedTranscoder, TimeConditionedTranscoderConfig


DEFAULT_POLICY_PATH = "lerobot/pi05_libero_finetuned"
DEFAULT_OUTPUT_DIR = Path("outputs/transcoders/pi05_libero")
TRAIN_CONFIG_NAME = "train_config.json"


def patch_transformers_causal_mask_compat() -> None:
    """Patch LeRobot PiGemma for Transformers builds without `cache_position`.

    LeRobot 0.6.1's PiGemma wrapper passes `cache_position` to
    `create_causal_mask`. Some Transformers builds expose a compatible mask
    helper without that keyword. Dropping it is safe here because PiGemma also
    passes `position_ids`, which the installed helper accepts.
    """
    import lerobot.policies.pi_gemma as pi_gemma

    original = pi_gemma.create_causal_mask
    if original is None or "cache_position" in inspect.signature(original).parameters:
        return

    def create_causal_mask_compat(*args, **kwargs):
        kwargs.pop("cache_position", None)
        return original(*args, **kwargs)

    pi_gemma.create_causal_mask = create_causal_mask_compat


def patch_pi05_checkpoint_key_compat() -> None:
    """Patch Pi0.5 checkpoint loading for vision tower key drift.

    The `lerobot/pi05_libero_finetuned` checkpoint stores SigLIP vision keys
    under `vision_tower.vision_model.*`, while the installed LeRobot Pi0.5
    model expects those same parameters directly under `vision_tower.*`.
    LeRobot catches the resulting `load_state_dict` error and continues, so we
    must fix the keys before loading to avoid silently using random vision
    weights.
    """
    from lerobot.policies.pi05 import modeling_pi05

    original = modeling_pi05.PI05Policy._fix_pytorch_state_dict_keys
    if getattr(original, "_pi05_mi_vision_compat", False):
        return

    def _fix_pytorch_state_dict_keys_compat(self, state_dict, model_config):
        fixed_state_dict = original(self, state_dict, model_config)
        remapped_state_dict = {}
        remap_count = 0
        for key, value in fixed_state_dict.items():
            new_key = key.replace(".vision_tower.vision_model.", ".vision_tower.")
            if new_key != key:
                remap_count += 1
            remapped_state_dict[new_key] = value
        if remap_count > 0:
            print(f"Remapped {remap_count} Pi0.5 vision tower checkpoint keys", flush=True)
        return remapped_state_dict

    _fix_pytorch_state_dict_keys_compat._pi05_mi_vision_compat = True
    modeling_pi05.PI05Policy._fix_pytorch_state_dict_keys = _fix_pytorch_state_dict_keys_compat


def resolve_device(device: str) -> torch.device:
    """Resolve `auto` to the best available local PyTorch device."""
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_policy_dtype(policy_dtype: str, device: torch.device) -> str:
    """Choose a conservative policy dtype for the selected backend."""
    if policy_dtype != "auto":
        return policy_dtype
    if device.type == "cuda":
        return "bfloat16"
    return "float32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", default=DEFAULT_POLICY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--num-feed-forwards",
        "--steps",
        dest="num_feed_forwards",
        type=int,
        default=1000,
        help="Number of frozen Pi0.5 record-collection feed-forwards. `--steps` is kept as a deprecated alias.",
    )
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=None,
        help=(
            "Dataset-style training mode: number of full passes over the selected training frames. "
            "When set, --num-feed-forwards is ignored except in logs/checkpoints."
        ),
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Optional smoke-test cap on training batches per epoch. Omit for all selected frames.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--episodes",
        default=None,
        help=(
            "Backward-compatible alias for training episode ids. Prefer --train-episodes when using "
            "validation/test splits."
        ),
    )
    parser.add_argument(
        "--train-episodes",
        default=None,
        help="Optional comma-separated episode ids used for transcoder training, for example `0,1,2,3`.",
    )
    parser.add_argument(
        "--val-episodes",
        default=None,
        help="Optional comma-separated held-out episode ids used for validation.",
    )
    parser.add_argument(
        "--test-episodes",
        default=None,
        help="Optional comma-separated held-out episode ids used once at the end for test metrics.",
    )
    parser.add_argument(
        "--episode-split",
        default=None,
        help=(
            "Optional train,val,test episode percentages, for example `80,10,10`. Used only for omitted "
            "--train-episodes/--val-episodes/--test-episodes."
        ),
    )
    parser.add_argument("--episode-split-seed", type=int, default=0)
    parser.add_argument(
        "--collection-mode",
        choices=("random-timestep", "inference", "training-forward"),
        default="random-timestep",
        help=(
            "How to collect MLP records. `random-timestep` does one denoiser call from sampled action noise "
            "at a sampled flow time and does not use ground-truth actions."
        ),
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Number of denoising steps for collection-mode=inference. Default uses the Pi0.5 config.",
    )
    parser.add_argument("--device", default="auto", help="Policy/transcoder device: auto, cuda, mps, or cpu.")
    parser.add_argument("--policy-dtype", default="auto", help="Policy dtype: auto, bfloat16, float16, or float32.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-l1", type=float, default=1e-4)
    parser.add_argument("--expansion-factor", type=int, default=16)
    parser.add_argument("--time-embedding-dim", type=int, default=None)
    parser.add_argument("--time-hidden-dim", type=int, default=None)
    parser.add_argument("--buffer-capacity", type=int, default=200_000)
    parser.add_argument(
        "--train-record-scope",
        choices=("buffer", "latest"),
        default="buffer",
        help=(
            "`buffer` trains over the rolling replay buffer after each FF. `latest` trains only on the "
            "newly collected records, while the rolling buffer still tracks variance/checkpoint stats."
        ),
    )
    parser.add_argument("--min-buffer-records", type=int, default=4096)
    parser.add_argument("--transcoder-batch-size", type=int, default=4096)
    parser.add_argument(
        "--transcoder-epochs-per-ff",
        type=int,
        default=1,
        help="Number of epochs over each ready transcoder buffer after each frozen Pi0.5 feed-forward.",
    )
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="Run validation every N training feed-forwards. Disabled when 0 or when --val-episodes is omitted.",
    )
    parser.add_argument(
        "--eval-feed-forwards",
        type=int,
        default=5,
        help="Legacy/capped number of held-out validation feed-forwards per validation pass.",
    )
    parser.add_argument(
        "--test-feed-forwards",
        type=int,
        default=None,
        help="Legacy/capped number of held-out test feed-forwards at the end. Defaults to --eval-feed-forwards.",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=None,
        help="Optional cap on validation/test batches. Omit for all selected held-out frames in epoch mode.",
    )
    parser.add_argument(
        "--eval-noise-samples",
        type=int,
        default=1,
        help="Number of random noise/timestep draws per validation/test batch.",
    )
    parser.add_argument(
        "--eval-buffer-capacity",
        type=int,
        default=0,
        help="Per-layer validation/test buffer capacity. Default auto-sizes from batch size and feed-forward count.",
    )
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=None,
        help="JSONL metrics path. Defaults to <output-dir>/metrics.jsonl.",
    )
    parser.add_argument(
        "--test-metrics-file",
        type=Path,
        default=None,
        help="JSON metrics path for final held-out test metrics. Defaults to <output-dir>/test_metrics.json.",
    )
    parser.add_argument("--append-metrics", action="store_true", help="Append to an existing metrics JSONL file.")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-checkpoints", type=int, default=3)
    parser.add_argument("--plan-only", action="store_true", help="Print dataset split/batch counts and exit before model load.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    return parser.parse_args()


def _train_config_path(policy_path: str, *, local_files_only: bool) -> Path:
    path = Path(policy_path)
    if path.is_dir():
        return path / TRAIN_CONFIG_NAME
    if path.is_file():
        return path
    return Path(hf_hub_download(policy_path, TRAIN_CONFIG_NAME, local_files_only=local_files_only))


def _load_train_config(args: argparse.Namespace) -> TrainPipelineConfig:
    config_path = _train_config_path(args.policy_path, local_files_only=args.local_files_only)
    with config_path.open() as f:
        raw_config = json.load(f)

    # Older Pi0.5 checkpoints used `eval_freq`; current LeRobot expects
    # `env_eval_freq`.
    if "eval_freq" in raw_config:
        raw_config.setdefault("env_eval_freq", raw_config["eval_freq"])
        raw_config.pop("eval_freq")

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(raw_config, f)
        migrated_path = Path(f.name)
    return TrainPipelineConfig.from_pretrained(migrated_path)


def _parse_episode_ids(episodes: str | None) -> list[int] | None:
    if episodes is None:
        return None
    return [int(item) for item in episodes.split(",") if item]


def _format_episode_ids(episodes: list[int] | None) -> str | None:
    if episodes is None:
        return None
    return ",".join(str(episode) for episode in episodes)


def _episode_summary(episodes: list[int] | None) -> str:
    if episodes is None:
        return "all"
    if len(episodes) <= 12:
        return str(episodes)
    return f"{len(episodes)} episodes ({episodes[0]}..{episodes[-1]})"


def _training_episodes_arg(args: argparse.Namespace) -> str | None:
    return args.train_episodes if args.train_episodes is not None else args.episodes


def _all_episode_ids(dataset) -> list[int]:
    from_indices = dataset.meta.episodes["dataset_from_index"]
    return list(range(len(from_indices)))


def _split_episode_ids(
    episode_ids: list[int],
    split: str,
    *,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    weights = [float(item) for item in split.split(",") if item]
    if len(weights) != 3:
        raise ValueError(f"--episode-split must have train,val,test percentages, got {split!r}")
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError(f"--episode-split must contain non-negative percentages, got {split!r}")

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(episode_ids), generator=generator).tolist()
    shuffled = [episode_ids[index] for index in permutation]
    total = len(shuffled)
    train_count = int(round(total * weights[0] / sum(weights)))
    val_count = int(round(total * weights[1] / sum(weights)))
    train_count = min(max(train_count, 0), total)
    val_count = min(max(val_count, 0), total - train_count)
    test_count = total - train_count - val_count
    if total >= 3:
        if train_count == 0:
            train_count = 1
        if val_count == 0:
            val_count = 1
        if test_count == 0:
            test_count = 1
        while train_count + val_count + test_count > total:
            if train_count >= val_count and train_count >= test_count and train_count > 1:
                train_count -= 1
            elif val_count >= test_count and val_count > 1:
                val_count -= 1
            else:
                test_count -= 1

    train = sorted(shuffled[:train_count])
    val = sorted(shuffled[train_count : train_count + val_count])
    test = sorted(shuffled[train_count + val_count :])
    return train, val, test


def _latest_buffer_capacity(args: argparse.Namespace, policy) -> int:
    chunk_size = int(getattr(policy.model.config, "chunk_size", 50))
    return max(args.transcoder_batch_size, args.batch_size * chunk_size)


def _configure_train_config(args: argparse.Namespace, *, episodes: str | None) -> TrainPipelineConfig:
    print("loading Pi0.5 training config", flush=True)
    cfg = _load_train_config(args)
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    if episodes is not None:
        cfg.dataset.episodes = _parse_episode_ids(episodes)
    cfg.policy.pretrained_path = Path(args.policy_path)
    cfg.policy.device = str(args.resolved_device)
    cfg.policy.dtype = args.resolved_policy_dtype
    cfg.policy.compile_model = False
    cfg.policy.gradient_checkpointing = False
    return cfg


def _config_with_episodes(cfg: TrainPipelineConfig, episodes: str | None) -> TrainPipelineConfig:
    split_cfg = copy.deepcopy(cfg)
    split_cfg.dataset.episodes = _parse_episode_ids(episodes)
    return split_cfg


def _dataloader_worker_kwargs(cfg: TrainPipelineConfig) -> dict[str, Any]:
    workers_enabled = cfg.num_workers > 0
    return {
        "prefetch_factor": cfg.prefetch_factor if workers_enabled else None,
        "persistent_workers": cfg.persistent_workers and workers_enabled,
        "multiprocessing_context": cfg.dataloader_multiprocessing_context if workers_enabled else None,
    }


def _make_dataloader(cfg: TrainPipelineConfig, dataset) -> torch.utils.data.DataLoader:
    active_cfg = cfg.trainable_config
    if dataset.episodes is None:
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=None,
            drop_n_last_frames=getattr(active_cfg, "drop_n_last_frames", 0),
            shuffle=True,
            seed=cfg.seed if cfg.seed is not None else 0,
            absolute_to_relative_idx=None,
        )
    else:
        sampler = torch.utils.data.SubsetRandomSampler(
            _selected_episode_relative_indices(
                dataset,
                drop_n_last_frames=getattr(active_cfg, "drop_n_last_frames", 0),
            ),
            generator=torch.Generator().manual_seed(cfg.seed if cfg.seed is not None else 0),
        )
    collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
    return torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        sampler=sampler,
        pin_memory=torch.device(cfg.policy.device).type == "cuda",
        drop_last=False,
        collate_fn=collate_fn,
        **_dataloader_worker_kwargs(cfg),
    )


def _selected_episode_relative_indices(dataset, *, drop_n_last_frames: int) -> list[int]:
    """Return valid relative row indices for an episode-filtered LeRobotDataset."""
    absolute_to_relative = dataset.absolute_to_relative_idx
    if absolute_to_relative is None:
        return list(range(len(dataset)))

    indices: list[int] = []
    from_indices = dataset.meta.episodes["dataset_from_index"]
    to_indices = dataset.meta.episodes["dataset_to_index"]
    for episode_idx in dataset.episodes:
        start = int(from_indices[episode_idx])
        stop = int(to_indices[episode_idx]) - drop_n_last_frames
        for absolute_idx in range(start, max(start, stop)):
            relative_idx = absolute_to_relative.get(absolute_idx)
            if relative_idx is not None:
                indices.append(relative_idx)

    if not indices:
        raise ValueError(f"No usable dataset frames found for selected episodes {dataset.episodes}")
    return indices


def _make_preprocessor(cfg: TrainPipelineConfig, dataset, policy_path: str):
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=policy_path,
        pretrained_revision=getattr(cfg.policy, "pretrained_revision", None),
        dataset_stats=dataset.meta.stats,
        dataset_meta=dataset.meta,
        preprocessor_overrides={
            "device_processor": {"device": str(cfg.policy.device)},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )
    return preprocessor


def _build_transcoders(args: argparse.Namespace, targets) -> dict[str, TimeConditionedTranscoder]:
    transcoders = {}
    for target in targets:
        config = TimeConditionedTranscoderConfig(
            d_model=target.d_model,
            expansion_factor=args.expansion_factor,
            time_embedding_dim=args.time_embedding_dim,
            time_hidden_dim=args.time_hidden_dim,
        )
        transcoders[target.name] = TimeConditionedTranscoder(config)
    return transcoders


def _move_transcoders(
    transcoders: dict[str, TimeConditionedTranscoder],
    *,
    device: torch.device,
) -> dict[str, TimeConditionedTranscoder]:
    for transcoder in transcoders.values():
        transcoder.to(device=device, dtype=torch.float32)
        transcoder.train()
    return transcoders


def _freeze_policy(policy: torch.nn.Module) -> None:
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)


def _prepare_raw_batch(batch: dict[str, Any], camera_keys: list[str]) -> dict[str, Any]:
    for camera_key in camera_keys:
        if camera_key in batch and batch[camera_key].dtype == torch.uint8:
            batch[camera_key] = batch[camera_key].to(dtype=torch.float32) / 255.0
    return batch


def _collect_random_timestep_records(policy, batch: dict[str, torch.Tensor]) -> None:
    """Run one Pi0.5 denoiser call at sampled noise and sampled flow time.

    This is the cheap MI collection path: prompt/image/state enter through the
    normal prefix, the action side is sampled noise, and one random flow time is
    sampled from Pi0.5's training-time distribution. Dataset actions are not read.
    """
    images, img_masks = policy._preprocess_images(batch)
    tokens = batch[OBS_LANGUAGE_TOKENS]
    masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    model = policy.model
    batch_size = tokens.shape[0]
    device = tokens.device

    action_noise = model.sample_noise(
        (batch_size, model.config.chunk_size, model.config.max_action_dim),
        device,
    )
    timestep = model.sample_time(batch_size, device)

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, tokens, masks)
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = prepare_attention_masks_4d(prefix_att_2d_masks)

    model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001
    _, past_key_values = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )

    model.denoise_step(
        prefix_pad_masks=prefix_pad_masks,
        past_key_values=past_key_values,
        x_t=action_noise,
        timestep=timestep,
    )


def _collect_records_for_batch(
    *,
    policy,
    raw_batch: dict[str, Any],
    dataset,
    preprocessor,
    context,
    buffers: MultiLayerActivationBuffer,
    args: argparse.Namespace,
    extra_buffers: MultiLayerActivationBuffer | None = None,
) -> dict[str, int]:
    raw_batch = _prepare_raw_batch(raw_batch, dataset.meta.camera_keys)
    batch = preprocessor(raw_batch)

    context.clear_records()
    with torch.no_grad():
        if args.collection_mode == "random-timestep":
            _collect_random_timestep_records(policy, batch)
        elif args.collection_mode == "inference":
            policy.predict_action_chunk(batch, num_steps=args.num_inference_steps)
        else:
            policy.forward(batch)
    if extra_buffers is not None:
        extra_buffers.add_context_records(context, clear_context=False)
    return buffers.add_context_records(context)


def _eval_buffer_capacity(args: argparse.Namespace, policy, batches: int, *, noise_samples: int) -> int:
    if args.eval_buffer_capacity > 0:
        return args.eval_buffer_capacity
    chunk_size = int(getattr(policy.model.config, "chunk_size", 50))
    return max(args.transcoder_batch_size, batches * noise_samples * args.batch_size * chunk_size)


def _planned_eval_batches(batch_source, *, feed_forwards: int | None, max_batches: int | None) -> int:
    if feed_forwards is not None:
        return feed_forwards
    if max_batches is not None:
        return max_batches
    try:
        return len(batch_source)
    except TypeError:
        return 1


def _evaluate_transcoders(
    *,
    split: str,
    transcoders: dict[str, TimeConditionedTranscoder],
    train_buffers: MultiLayerActivationBuffer,
    layer_names: list[str],
    policy,
    batch_source,
    dataset,
    preprocessor,
    context,
    args: argparse.Namespace,
    device: torch.device,
    feed_forwards: int | None = None,
    max_batches: int | None = None,
) -> dict[str, float] | None:
    if feed_forwards is not None and feed_forwards <= 0:
        return None
    if args.eval_noise_samples <= 0:
        raise ValueError(f"--eval-noise-samples must be positive, got {args.eval_noise_samples}")

    first_buffer = train_buffers.buffers[layer_names[0]]
    planned_batches = _planned_eval_batches(batch_source, feed_forwards=feed_forwards, max_batches=max_batches)
    eval_buffers = MultiLayerActivationBuffer(
        layer_names=layer_names,
        d_model=first_buffer.d_model,
        capacity_per_layer=_eval_buffer_capacity(args, policy, planned_batches, noise_samples=args.eval_noise_samples),
    )

    added_records = 0
    previous_modes = {name: transcoder.training for name, transcoder in transcoders.items()}
    for transcoder in transcoders.values():
        transcoder.eval()

    try:
        if feed_forwards is not None:
            batch_iter = (next(batch_source) for _ in range(feed_forwards))
        else:
            batch_iter = iter(batch_source)
        for batch_index, raw_batch in enumerate(batch_iter):
            if max_batches is not None and batch_index >= max_batches:
                break
            for _noise_sample in range(args.eval_noise_samples):
                added_by_layer = _collect_records_for_batch(
                    policy=policy,
                    raw_batch=raw_batch,
                    dataset=dataset,
                    preprocessor=preprocessor,
                    context=context,
                    buffers=eval_buffers,
                    args=args,
                )
                added_records += sum(added_by_layer.values())

        ready_names = [
            name
            for name in layer_names
            if eval_buffers.buffers[name].size > 0 and train_buffers.buffers[name].size >= args.min_buffer_records
        ]
        if not ready_names:
            return None

        total_weighted_loss = 0.0
        total_weighted_mse = 0.0
        total_weighted_l1 = 0.0
        total_records = 0
        minibatches = 0

        expected_minibatches = sum(
            ceil(eval_buffers.buffers[name].size / args.transcoder_batch_size) for name in ready_names
        )
        with tqdm(
            total=expected_minibatches,
            desc=f"{split} minibatches",
            leave=False,
            disable=args.no_progress,
        ) as minibatch_bar:
            with torch.no_grad():
                for name in ready_names:
                    transcoder = transcoders[name]
                    denominator = train_buffers.variance_denominator(name, device=device)
                    for sample in eval_buffers.epoch_batches(
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
                            lambda_l1=args.lambda_l1,
                        )
                        record_count = sample.x.shape[0]
                        total_weighted_loss += float(loss.total.detach().cpu()) * record_count
                        total_weighted_mse += float(loss.normalized_mse.detach().cpu()) * record_count
                        total_weighted_l1 += float(loss.l1.detach().cpu()) * record_count
                        total_records += record_count
                        minibatches += 1
                        minibatch_bar.update(1)

        if total_records == 0:
            return None
        return {
            "loss": total_weighted_loss / total_records,
            "normalized_mse": total_weighted_mse / total_records,
            "l1": total_weighted_l1 / total_records,
            "ready_layers": float(len(ready_names)),
            "minibatches": float(minibatches),
            "records": float(total_records),
            "feed_forwards": float(planned_batches * args.eval_noise_samples),
            "batches": float(planned_batches),
            "noise_samples": float(args.eval_noise_samples),
            "added_records": float(added_records),
        }
    finally:
        context.clear_records()
        for name, transcoder in transcoders.items():
            transcoder.train(previous_modes[name])


def _train_transcoders_for_epochs(
    *,
    transcoders: dict[str, TimeConditionedTranscoder],
    buffers: MultiLayerActivationBuffer,
    denominator_buffers: MultiLayerActivationBuffer | None = None,
    optimizer: torch.optim.Optimizer,
    layer_names: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float] | None:
    denominator_buffers = buffers if denominator_buffers is None else denominator_buffers
    ready_names = [name for name in layer_names if buffers.buffers[name].size >= args.min_buffer_records]
    if not ready_names:
        return None
    if args.transcoder_epochs_per_ff <= 0:
        return None

    expected_minibatches = args.transcoder_epochs_per_ff * sum(
        ceil(buffers.buffers[name].size / args.transcoder_batch_size) for name in ready_names
    )
    total_loss = 0.0
    mse_sum = 0.0
    l1_sum = 0.0
    minibatches = 0

    with tqdm(
        total=expected_minibatches,
        desc="transcoder minibatches",
        leave=False,
        disable=args.no_progress,
    ) as minibatch_bar:
        for _epoch in range(args.transcoder_epochs_per_ff):
            for name in ready_names:
                transcoder = transcoders[name]
                for sample in buffers.epoch_batches(name, args.transcoder_batch_size, device=device, dtype=torch.float32):
                    optimizer.zero_grad(set_to_none=True)
                    loss = transcoder.loss(
                        sample.x,
                        sample.timestep,
                        sample.y,
                        denominator_buffers.variance_denominator(name, device=device),
                        lambda_l1=args.lambda_l1,
                    )
                    loss.total.backward()
                    torch.nn.utils.clip_grad_norm_(transcoder.parameters(), args.grad_clip_norm)
                    optimizer.step()
                    transcoder.renormalize_decoder_columns()

                    loss_value = float(loss.total.detach().cpu())
                    normalized_mse_value = float(loss.normalized_mse.detach().cpu())
                    l1_value = float(loss.l1.detach().cpu())
                    total_loss += loss_value
                    mse_sum += normalized_mse_value
                    l1_sum += l1_value
                    minibatches += 1
                    minibatch_bar.set_postfix(
                        loss=f"{loss_value:.4f}",
                        norm_mse=f"{normalized_mse_value:.4f}",
                        layer=name.rsplit(".", 2)[-2],
                    )
                    minibatch_bar.update(1)

    divisor = max(1, minibatches)
    return {
        "loss": total_loss / divisor,
        "normalized_mse": mse_sum / divisor,
        "l1": l1_sum / divisor,
        "ready_layers": float(len(ready_names)),
        "minibatches": float(minibatches),
    }


def _save_checkpoint(
    *,
    output_dir: Path,
    step: int,
    args: argparse.Namespace,
    transcoders: dict[str, TimeConditionedTranscoder],
    optimizer: torch.optim.Optimizer,
    buffers: MultiLayerActivationBuffer,
    wrapped_names: list[str],
    metrics_history: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"step_{step:06d}.pt"
    torch.save(
        {
            "feed_forward": step,
            "num_feed_forwards": args.num_feed_forwards,
            "policy_path": args.policy_path,
            "collection_mode": args.collection_mode,
            "num_inference_steps": args.num_inference_steps,
            "transcoder_epochs_per_ff": args.transcoder_epochs_per_ff,
            "lambda_l1": args.lambda_l1,
            "wrapped_names": wrapped_names,
            "configs": {name: asdict(transcoder.config) for name, transcoder in transcoders.items()},
            "state_dicts": {name: transcoder.state_dict() for name, transcoder in transcoders.items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "buffer_stats": {name: asdict(stats) for name, stats in buffers.stats().items()},
            "metrics_history": metrics_history,
        },
        checkpoint_path,
    )

    checkpoints = sorted(output_dir.glob("step_*.pt"))
    extra = len(checkpoints) - args.max_checkpoints
    if args.max_checkpoints > 0 and extra > 0:
        for old_checkpoint in checkpoints[:extra]:
            old_checkpoint.unlink()


def _append_metrics_row(metrics_file: Path, row: dict[str, float | int | str | None]) -> None:
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with metrics_file.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    patch_transformers_causal_mask_compat()
    patch_pi05_checkpoint_key_compat()
    args = parse_args()
    if args.metrics_file is None:
        args.metrics_file = args.output_dir / "metrics.jsonl"
    if args.test_metrics_file is None:
        args.test_metrics_file = args.output_dir / "test_metrics.json"
    if not args.append_metrics and args.metrics_file.exists():
        args.metrics_file.unlink()
    device = resolve_device(args.device)
    args.resolved_device = device
    args.resolved_policy_dtype = resolve_policy_dtype(args.policy_dtype, device)
    train_episodes = _training_episodes_arg(args)
    cfg = _configure_train_config(args, episodes=None if args.episode_split is not None else train_episodes)
    if args.episode_split is not None:
        print(f"loading split source dataset {cfg.dataset.repo_id} episodes=all", flush=True)
        split_source_dataset = make_dataset(cfg)
        auto_train, auto_val, auto_test = _split_episode_ids(
            _all_episode_ids(split_source_dataset),
            args.episode_split,
            seed=args.episode_split_seed,
        )
        train_episodes = train_episodes or _format_episode_ids(auto_train)
        args.val_episodes = args.val_episodes or _format_episode_ids(auto_val)
        args.test_episodes = args.test_episodes or _format_episode_ids(auto_test)
        print(
            "episode split "
            f"train={_episode_summary(_parse_episode_ids(train_episodes))} "
            f"val={_episode_summary(_parse_episode_ids(args.val_episodes))} "
            f"test={_episode_summary(_parse_episode_ids(args.test_episodes))}",
            flush=True,
        )
        cfg = _config_with_episodes(cfg, train_episodes)
    print(f"loading dataset {cfg.dataset.repo_id} episodes={_episode_summary(cfg.dataset.episodes)}", flush=True)
    dataset = make_dataset(cfg)
    print("building dataloader", flush=True)
    dataloader = _make_dataloader(cfg, dataset)
    print(f"train frames={len(dataset)} train_batches={len(dataloader)} batch_size={cfg.batch_size}", flush=True)
    data_iter = cycle(dataloader)
    val_data_iter = None
    val_dataloader = None
    val_dataset = None
    if args.val_episodes is not None:
        val_cfg = _config_with_episodes(cfg, args.val_episodes)
        print(
            f"loading validation dataset {val_cfg.dataset.repo_id} episodes={_episode_summary(val_cfg.dataset.episodes)}",
            flush=True,
        )
        val_dataset = make_dataset(val_cfg)
        print("building validation dataloader", flush=True)
        val_dataloader = _make_dataloader(val_cfg, val_dataset)
        print(f"validation frames={len(val_dataset)} validation_batches={len(val_dataloader)}", flush=True)
        val_data_iter = cycle(val_dataloader)
    test_data_iter = None
    test_dataloader = None
    test_dataset = None
    if args.test_episodes is not None:
        test_cfg = _config_with_episodes(cfg, args.test_episodes)
        print(
            f"loading test dataset {test_cfg.dataset.repo_id} episodes={_episode_summary(test_cfg.dataset.episodes)}",
            flush=True,
        )
        test_dataset = make_dataset(test_cfg)
        print("building test dataloader", flush=True)
        test_dataloader = _make_dataloader(test_cfg, test_dataset)
        print(f"test frames={len(test_dataset)} test_batches={len(test_dataloader)}", flush=True)
        test_data_iter = cycle(test_dataloader)
    if args.plan_only:
        print("plan-only requested; exiting before model load", flush=True)
        return
    print("loading preprocessor/tokenizer", flush=True)
    preprocessor = _make_preprocessor(cfg, dataset, args.policy_path)

    print("loading frozen Pi0.5 policy weights", flush=True)
    policy = make_policy(cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    print("freezing Pi0.5 policy", flush=True)
    _freeze_policy(policy)

    targets = list_pi05_action_expert_mlp_targets(policy)
    if not targets:
        raise RuntimeError("No Pi0.5 action-expert MLP targets found")
    d_models = {target.d_model for target in targets}
    if len(d_models) != 1:
        raise ValueError(f"Expected one shared d_model across action-expert MLPs, got {sorted(d_models)}")

    transcoders = _move_transcoders(_build_transcoders(args, targets), device=device)
    context, wrapped_names = install_pi05_action_expert_wrappers(policy, transcoders=transcoders, mode="train")
    buffers = MultiLayerActivationBuffer(
        layer_names=wrapped_names,
        d_model=d_models.pop(),
        capacity_per_layer=args.buffer_capacity,
    )
    optimizer = torch.optim.AdamW(
        [parameter for transcoder in transcoders.values() for parameter in transcoder.parameters()],
        lr=args.lr,
    )

    print(f"training {len(wrapped_names)} Pi0.5 action-expert transcoders")
    print(
        f"dataset={cfg.dataset.repo_id} policy={args.policy_path} "
        f"device={device} policy_dtype={args.resolved_policy_dtype} "
        f"collection_mode={args.collection_mode}"
    )
    if val_dataset is not None and args.eval_every > 0:
        print(
            f"validation episodes={_episode_summary(val_dataset.episodes)} every={args.eval_every} "
            f"eval_feed_forwards={args.eval_feed_forwards}",
            flush=True,
        )
    if test_dataset is not None:
        print(
            f"test episodes={_episode_summary(test_dataset.episodes)} "
            f"test_feed_forwards={args.test_feed_forwards or args.eval_feed_forwards}",
            flush=True,
        )

    last_metrics: dict[str, float] | None = None
    metrics_history: list[dict[str, Any]] = []
    started_at = time.time()

    def iter_training_batches():
        feed_forward_count = 0
        if args.train_epochs is None:
            for _ in range(args.num_feed_forwards):
                feed_forward_count += 1
                yield None, None, feed_forward_count, next(data_iter)
            return

        for epoch in range(1, args.train_epochs + 1):
            for batch_index, raw_batch in enumerate(dataloader, start=1):
                if args.max_train_batches is not None and batch_index > args.max_train_batches:
                    break
                feed_forward_count += 1
                yield epoch, batch_index, feed_forward_count, raw_batch

    if args.train_epochs is None:
        planned_train_batches = args.num_feed_forwards
    else:
        batches_per_epoch = len(dataloader)
        if args.max_train_batches is not None:
            batches_per_epoch = min(batches_per_epoch, args.max_train_batches)
        planned_train_batches = args.train_epochs * batches_per_epoch
        args.num_feed_forwards = planned_train_batches

    feed_forward_iter = tqdm(
        iter_training_batches(),
        total=planned_train_batches,
        desc="Pi0.5 feed-forwards",
        disable=args.no_progress,
    )
    for epoch, batch_index, feed_forward, raw_batch in feed_forward_iter:
        latest_buffers = None
        if args.train_record_scope == "latest":
            latest_buffers = MultiLayerActivationBuffer(
                layer_names=wrapped_names,
                d_model=next(iter(buffers.buffers.values())).d_model,
                capacity_per_layer=_latest_buffer_capacity(args, policy),
            )
        added_by_layer = _collect_records_for_batch(
            policy=policy,
            raw_batch=raw_batch,
            dataset=dataset,
            preprocessor=preprocessor,
            context=context,
            buffers=buffers,
            args=args,
            extra_buffers=latest_buffers,
        )
        train_source_buffers = latest_buffers if latest_buffers is not None else buffers

        metrics = _train_transcoders_for_epochs(
            transcoders=transcoders,
            buffers=train_source_buffers,
            denominator_buffers=buffers,
            optimizer=optimizer,
            layer_names=wrapped_names,
            args=args,
            device=device,
        )
        if metrics is not None:
            last_metrics = metrics

        val_metrics = None
        if (
            val_data_iter is not None
            and val_dataset is not None
            and args.eval_every > 0
            and feed_forward % args.eval_every == 0
        ):
            val_metrics = _evaluate_transcoders(
                split="validation",
                transcoders=transcoders,
                train_buffers=buffers,
                layer_names=wrapped_names,
                policy=policy,
                batch_source=val_data_iter,
                dataset=val_dataset,
                preprocessor=preprocessor,
                context=context,
                args=args,
                device=device,
                feed_forwards=args.eval_feed_forwards,
            )

        filled = min(stats.size for stats in buffers.stats().values())
        added = sum(added_by_layer.values())
        metrics_row: dict[str, float | int | str | None] = {
            "feed_forward": feed_forward,
            "epoch": epoch,
            "batch_index": batch_index,
            "train_mode": "fixed-feed-forwards" if args.train_epochs is None else "epochs",
            "train_record_scope": args.train_record_scope,
            "elapsed_s": time.time() - started_at,
            "added_records": added,
            "min_buffer": filled,
            "collection_mode": args.collection_mode,
            "loss": None,
            "normalized_mse": None,
            "l1": None,
            "ready_layers": 0,
            "minibatches": 0,
            "val_loss": None,
            "val_normalized_mse": None,
            "val_l1": None,
            "val_ready_layers": 0,
            "val_minibatches": 0,
            "val_records": 0,
            "val_feed_forwards": 0,
            "val_added_records": 0,
        }
        if metrics is not None:
            metrics_row.update(
                {
                    "loss": metrics["loss"],
                    "normalized_mse": metrics["normalized_mse"],
                    "l1": metrics["l1"],
                    "ready_layers": int(metrics["ready_layers"]),
                    "minibatches": int(metrics["minibatches"]),
                }
            )
        if val_metrics is not None:
            metrics_row.update(
                {
                    "val_loss": val_metrics["loss"],
                    "val_normalized_mse": val_metrics["normalized_mse"],
                    "val_l1": val_metrics["l1"],
                    "val_ready_layers": int(val_metrics["ready_layers"]),
                    "val_minibatches": int(val_metrics["minibatches"]),
                    "val_records": int(val_metrics["records"]),
                    "val_feed_forwards": int(val_metrics["feed_forwards"]),
                    "val_added_records": int(val_metrics["added_records"]),
                }
            )
        metrics_history.append(metrics_row)
        _append_metrics_row(args.metrics_file, metrics_row)

        if args.log_every > 0 and feed_forward % args.log_every == 0:
            metric_text = "warming buffers"
            if last_metrics is not None:
                metric_text = (
                    f"loss={last_metrics['loss']:.4f} "
                    f"norm_mse={last_metrics['normalized_mse']:.4f} "
                    f"l1={last_metrics['l1']:.4f} "
                    f"ready_layers={int(last_metrics['ready_layers'])} "
                    f"minibatches={int(last_metrics['minibatches'])}"
                )
                feed_forward_iter.set_postfix(
                    loss=f"{last_metrics['loss']:.4f}",
                    min_buffer=filled,
                    ready_layers=int(last_metrics["ready_layers"]),
                )
            progress_line = f"feed_forward={feed_forward} added_records={added} min_buffer={filled} {metric_text}"
            if val_metrics is not None:
                progress_line += (
                    f" val_loss={val_metrics['loss']:.4f} "
                    f"val_norm_mse={val_metrics['normalized_mse']:.4f} "
                    f"val_l1={val_metrics['l1']:.4f}"
                )
            if args.no_progress:
                print(progress_line, flush=True)
            else:
                tqdm.write(progress_line)

        if args.save_every > 0 and feed_forward % args.save_every == 0:
            _save_checkpoint(
                output_dir=args.output_dir,
                step=feed_forward,
                args=args,
                transcoders=transcoders,
                optimizer=optimizer,
                buffers=buffers,
                wrapped_names=wrapped_names,
                metrics_history=metrics_history,
            )

    test_metrics = None
    if val_dataloader is not None and val_dataset is not None and args.train_epochs is not None and args.eval_every <= 0:
        val_metrics = _evaluate_transcoders(
            split="validation",
            transcoders=transcoders,
            train_buffers=buffers,
            layer_names=wrapped_names,
            policy=policy,
            batch_source=val_dataloader,
            dataset=val_dataset,
            preprocessor=preprocessor,
            context=context,
            args=args,
            device=device,
            feed_forwards=None,
            max_batches=args.max_eval_batches,
        )
        if val_metrics is not None:
            final_val_row: dict[str, Any] = {
                "event": "final_validation",
                "feed_forward": args.num_feed_forwards,
                "elapsed_s": time.time() - started_at,
                "collection_mode": args.collection_mode,
                "val_loss": val_metrics["loss"],
                "val_normalized_mse": val_metrics["normalized_mse"],
                "val_l1": val_metrics["l1"],
                "val_ready_layers": int(val_metrics["ready_layers"]),
                "val_minibatches": int(val_metrics["minibatches"]),
                "val_records": int(val_metrics["records"]),
                "val_feed_forwards": int(val_metrics["feed_forwards"]),
                "val_batches": int(val_metrics["batches"]),
                "val_noise_samples": int(val_metrics["noise_samples"]),
                "val_added_records": int(val_metrics["added_records"]),
            }
            metrics_history.append(final_val_row)
            _append_metrics_row(args.metrics_file, final_val_row)
            print(
                f"final validation loss={val_metrics['loss']:.4f} "
                f"norm_mse={val_metrics['normalized_mse']:.4f} "
                f"l1={val_metrics['l1']:.4f} "
                f"records={int(val_metrics['records'])}",
                flush=True,
            )

    if test_data_iter is not None and test_dataset is not None:
        test_metrics = _evaluate_transcoders(
            split="test",
            transcoders=transcoders,
            train_buffers=buffers,
            layer_names=wrapped_names,
            policy=policy,
            batch_source=test_dataloader if args.train_epochs is not None and test_dataloader is not None else test_data_iter,
            dataset=test_dataset,
            preprocessor=preprocessor,
            context=context,
            args=args,
            device=device,
            feed_forwards=None if args.train_epochs is not None else args.test_feed_forwards or args.eval_feed_forwards,
            max_batches=args.max_eval_batches if args.train_epochs is not None else None,
        )
        if test_metrics is not None:
            args.test_metrics_file.parent.mkdir(parents=True, exist_ok=True)
            test_row = {
                "feed_forward": args.num_feed_forwards,
                "elapsed_s": time.time() - started_at,
                "collection_mode": args.collection_mode,
                "episodes": test_dataset.episodes,
                "loss": test_metrics["loss"],
                "normalized_mse": test_metrics["normalized_mse"],
                "l1": test_metrics["l1"],
                "ready_layers": int(test_metrics["ready_layers"]),
                "minibatches": int(test_metrics["minibatches"]),
                "records": int(test_metrics["records"]),
                "feed_forwards": int(test_metrics["feed_forwards"]),
                "batches": int(test_metrics["batches"]),
                "noise_samples": int(test_metrics["noise_samples"]),
                "added_records": int(test_metrics["added_records"]),
            }
            with args.test_metrics_file.open("w") as f:
                json.dump(test_row, f, indent=2, sort_keys=True)
                f.write("\n")
            metrics_history.append({"event": "test", **test_row})
            print(
                f"test loss={test_metrics['loss']:.4f} "
                f"norm_mse={test_metrics['normalized_mse']:.4f} "
                f"l1={test_metrics['l1']:.4f} "
                f"records={int(test_metrics['records'])}",
                flush=True,
            )

    if test_metrics is not None or args.save_every <= 0 or args.num_feed_forwards % args.save_every != 0:
        _save_checkpoint(
            output_dir=args.output_dir,
            step=args.num_feed_forwards,
            args=args,
            transcoders=transcoders,
            optimizer=optimizer,
            buffers=buffers,
            wrapped_names=wrapped_names,
            metrics_history=metrics_history,
        )


if __name__ == "__main__":
    main()

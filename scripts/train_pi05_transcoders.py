#!/usr/bin/env python
"""Train DifFRACT-style transcoders for Pi0.5 action-expert MLPs."""

from __future__ import annotations

import argparse
import inspect
import json
import tempfile
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
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--episodes",
        default=None,
        help="Optional comma-separated dataset episode ids to load, for example `0` or `0,1,2`.",
    )
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
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-checkpoints", type=int, default=3)
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


def _configure_train_config(args: argparse.Namespace) -> TrainPipelineConfig:
    print("loading Pi0.5 training config", flush=True)
    cfg = _load_train_config(args)
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    if args.episodes is not None:
        cfg.dataset.episodes = [int(item) for item in args.episodes.split(",") if item]
    cfg.policy.pretrained_path = Path(args.policy_path)
    cfg.policy.device = str(args.resolved_device)
    cfg.policy.dtype = args.resolved_policy_dtype
    cfg.policy.compile_model = False
    cfg.policy.gradient_checkpointing = False
    return cfg


def _dataloader_worker_kwargs(cfg: TrainPipelineConfig) -> dict[str, Any]:
    workers_enabled = cfg.num_workers > 0
    return {
        "prefetch_factor": cfg.prefetch_factor if workers_enabled else None,
        "persistent_workers": cfg.persistent_workers and workers_enabled,
        "multiprocessing_context": cfg.dataloader_multiprocessing_context if workers_enabled else None,
    }


def _make_dataloader(cfg: TrainPipelineConfig, dataset) -> torch.utils.data.DataLoader:
    active_cfg = cfg.trainable_config
    sampler = EpisodeAwareSampler(
        dataset.meta.episodes["dataset_from_index"],
        dataset.meta.episodes["dataset_to_index"],
        episode_indices_to_use=dataset.episodes,
        drop_n_last_frames=getattr(active_cfg, "drop_n_last_frames", 0),
        shuffle=True,
        seed=cfg.seed if cfg.seed is not None else 0,
        absolute_to_relative_idx=dataset.absolute_to_relative_idx,
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


def _train_transcoders_for_epochs(
    *,
    transcoders: dict[str, TimeConditionedTranscoder],
    buffers: MultiLayerActivationBuffer,
    optimizer: torch.optim.Optimizer,
    layer_names: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float] | None:
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
                        buffers.variance_denominator(name, device=device),
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
    buffers: MultiLayerActivationBuffer,
    wrapped_names: list[str],
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
            "buffer_stats": {name: asdict(stats) for name, stats in buffers.stats().items()},
        },
        checkpoint_path,
    )

    checkpoints = sorted(output_dir.glob("step_*.pt"))
    extra = len(checkpoints) - args.max_checkpoints
    if args.max_checkpoints > 0 and extra > 0:
        for old_checkpoint in checkpoints[:extra]:
            old_checkpoint.unlink()


def main() -> None:
    patch_transformers_causal_mask_compat()
    patch_pi05_checkpoint_key_compat()
    args = parse_args()
    device = resolve_device(args.device)
    args.resolved_device = device
    args.resolved_policy_dtype = resolve_policy_dtype(args.policy_dtype, device)
    cfg = _configure_train_config(args)
    print(f"loading dataset {cfg.dataset.repo_id} episodes={cfg.dataset.episodes}", flush=True)
    dataset = make_dataset(cfg)
    print("building dataloader", flush=True)
    dataloader = _make_dataloader(cfg, dataset)
    data_iter = cycle(dataloader)
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

    last_metrics: dict[str, float] | None = None
    feed_forward_iter = tqdm(
        range(1, args.num_feed_forwards + 1),
        desc="Pi0.5 feed-forwards",
        disable=args.no_progress,
    )
    for feed_forward in feed_forward_iter:
        raw_batch = next(data_iter)
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
        added_by_layer = buffers.add_context_records(context)

        metrics = _train_transcoders_for_epochs(
            transcoders=transcoders,
            buffers=buffers,
            optimizer=optimizer,
            layer_names=wrapped_names,
            args=args,
            device=device,
        )
        if metrics is not None:
            last_metrics = metrics

        if args.log_every > 0 and feed_forward % args.log_every == 0:
            filled = min(stats.size for stats in buffers.stats().values())
            added = sum(added_by_layer.values())
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
                buffers=buffers,
                wrapped_names=wrapped_names,
            )

    _save_checkpoint(
        output_dir=args.output_dir,
        step=args.num_feed_forwards,
        args=args,
        transcoders=transcoders,
        buffers=buffers,
        wrapped_names=wrapped_names,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Inspect Pi0.5 action-expert MLP targets for DifFRACT-style transcoders.

This script avoids loading checkpoint weights. It downloads/reads the small HF
config, instantiates the LeRobot Pi0.5 module tree on the meta device, and
reports the action-expert MLP module names and shapes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download

from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch


DEFAULT_REPO_ID = "lerobot/pi05_libero_finetuned"
DEFAULT_OUTPUT = Path("docs/pi05_stage1_inspection.md")


def _fetch_json(repo_id: str, filename: str, local_files_only: bool) -> tuple[dict[str, Any] | None, Path | None]:
    try:
        path = Path(hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=local_files_only))
    except Exception:
        return None, None
    return json.loads(path.read_text()), path


def _policy_config_from_hf(raw: dict[str, Any]) -> PI05Config:
    """Build only the fields needed for architecture inspection."""
    return PI05Config(
        paligemma_variant=raw["paligemma_variant"],
        action_expert_variant=raw["action_expert_variant"],
        dtype="float32",
        chunk_size=raw["chunk_size"],
        n_action_steps=raw["n_action_steps"],
        max_state_dim=raw["max_state_dim"],
        max_action_dim=raw["max_action_dim"],
        num_inference_steps=raw["num_inference_steps"],
        time_sampling_beta_alpha=raw["time_sampling_beta_alpha"],
        time_sampling_beta_beta=raw["time_sampling_beta_beta"],
        time_sampling_scale=raw["time_sampling_scale"],
        time_sampling_offset=raw["time_sampling_offset"],
        min_period=raw["min_period"],
        max_period=raw["max_period"],
        image_resolution=tuple(raw["image_resolution"]),
        empty_cameras=raw["empty_cameras"],
    )


def inspect(repo_id: str, local_files_only: bool) -> dict[str, Any]:
    config_json, config_path = _fetch_json(repo_id, "config.json", local_files_only)
    if config_json is None:
        raise SystemExit(f"Could not load config.json for {repo_id}")
    train_config_json, train_config_path = _fetch_json(repo_id, "train_config.json", local_files_only)

    cfg = _policy_config_from_hf(config_json)

    with torch.device("meta"):
        model = PI05Pytorch(cfg)

    expert_mlps: list[dict[str, Any]] = []
    target_prefix = "paligemma_with_expert.gemma_expert.model.layers."
    for name, module in model.named_modules():
        if not (name.startswith(target_prefix) and name.endswith(".mlp")):
            continue
        up_proj = getattr(module, "up_proj", None)
        gate_proj = getattr(module, "gate_proj", None)
        down_proj = getattr(module, "down_proj", None)
        expert_mlps.append(
            {
                "name": name,
                "type": type(module).__name__,
                "up_proj_weight": list(up_proj.weight.shape) if up_proj is not None else None,
                "gate_proj_weight": list(gate_proj.weight.shape) if gate_proj is not None else None,
                "down_proj_weight": list(down_proj.weight.shape) if down_proj is not None else None,
            }
        )

    dataset_repo = None
    if train_config_json is not None:
        dataset = train_config_json.get("dataset") or {}
        dataset_repo = dataset.get("repo_id")

    return {
        "repo_id": repo_id,
        "config_path": str(config_path) if config_path is not None else None,
        "train_config_path": str(train_config_path) if train_config_path is not None else None,
        "dataset_repo": dataset_repo,
        "policy": {
            "paligemma_variant": cfg.paligemma_variant,
            "action_expert_variant": cfg.action_expert_variant,
            "chunk_size": cfg.chunk_size,
            "n_action_steps": cfg.n_action_steps,
            "max_action_dim": cfg.max_action_dim,
            "num_inference_steps": cfg.num_inference_steps,
            "min_period": cfg.min_period,
            "max_period": cfg.max_period,
            "input_features": config_json.get("input_features"),
            "output_features": config_json.get("output_features"),
        },
        "expert_mlp_count": len(expert_mlps),
        "expert_mlps": expert_mlps,
    }


def render_markdown(result: dict[str, Any]) -> str:
    policy = result["policy"]
    lines = [
        "# Pi0.5 Stage 1 Inspection",
        "",
        "This report inspects the LeRobot Pi0.5 LIBERO action-expert MLPs targeted by DifFRACT-style transcoders.",
        "It uses the Hugging Face config and a meta-device model instantiation; checkpoint weights are not loaded.",
        "",
        "## Source",
        "",
        f"- HF policy repo: `{result['repo_id']}`",
        f"- Config path: `{result['config_path']}`",
    ]
    if result.get("train_config_path"):
        lines.append(f"- Train config path: `{result['train_config_path']}`")
    if result.get("dataset_repo"):
        lines.append(f"- Training dataset listed by config: `{result['dataset_repo']}`")

    lines += [
        "",
        "## Policy Settings",
        "",
        f"- `paligemma_variant`: `{policy['paligemma_variant']}`",
        f"- `action_expert_variant`: `{policy['action_expert_variant']}`",
        f"- `chunk_size`: `{policy['chunk_size']}` action tokens",
        f"- `n_action_steps`: `{policy['n_action_steps']}`",
        f"- `max_action_dim`: `{policy['max_action_dim']}` internal padded action dim",
        f"- `num_inference_steps`: `{policy['num_inference_steps']}`",
        f"- timestep embedding periods: `{policy['min_period']}` to `{policy['max_period']}`",
        "",
        "The LIBERO checkpoint outputs 7 action dimensions, but Pi0.5 pads actions internally to 32 dimensions before projecting them into the 1024-wide action expert.",
        "",
        "## Target MLPs",
        "",
        f"- Target count: `{result['expert_mlp_count']}` action-expert MLPs",
        "- Target name pattern: `paligemma_with_expert.gemma_expert.model.layers.<idx>.mlp`",
        "- Module type: `GemmaMLP`",
        "- Per-token function shape: `R^1024 -> R^1024`",
        "- Internal MLP expansion: `1024 -> 4096 -> 1024`",
        "",
        "| idx | module | up_proj | gate_proj | down_proj |",
        "| ---: | --- | --- | --- | --- |",
    ]

    for idx, item in enumerate(result["expert_mlps"]):
        lines.append(
            f"| {idx} | `{item['name']}` | `{item['up_proj_weight']}` | "
            f"`{item['gate_proj_weight']}` | `{item['down_proj_weight']}` |"
        )

    num_steps = policy["num_inference_steps"]
    chunk_size = policy["chunk_size"]
    n_layers = result["expert_mlp_count"]
    lines += [
        "",
        "## Call Pattern",
        "",
        "Training forward:",
        "",
        "- `PI05Policy.forward` samples one continuous flow-matching time per batch item.",
        "- `PI05Pytorch.forward` builds noisy actions `x_t = t * noise + (1 - t) * actions`.",
        "- The action suffix has `chunk_size` action tokens.",
        "- Each target MLP is called once for that sampled time.",
        "",
        "Inference forward:",
        "",
        f"- `sample_actions` runs Euler integration for `{num_steps}` denoise steps.",
        "- At step `s`, LeRobot uses `time = 1.0 + s * (-1 / num_steps)`.",
        f"- With `{num_steps}` steps, times are `1.0, 0.9, ..., 0.1`.",
        "- Each denoise step calls every action-expert MLP once.",
        "",
        "Therefore, for one inference action chunk:",
        "",
        f"- MLP calls: `{n_layers} layers * {num_steps} timesteps = {n_layers * num_steps}` expert MLP calls.",
        f"- Per target MLP, DifFRACT-style records: `batch_size * {chunk_size} action tokens * {num_steps} timesteps`.",
        "",
        "## Timestep Source",
        "",
        "- Raw timestep is the flow-matching scalar `t` passed to `embed_suffix(noisy_actions, timestep)`.",
        "- `embed_suffix` creates a sinusoidal time embedding, then applies `time_mlp_in -> SiLU -> time_mlp_out -> SiLU`.",
        "- The resulting `adarms_cond` conditions every action-expert layer through adaptive RMSNorm.",
        "- The actual `GemmaMLP.forward` still receives only the post-attention normalized hidden states `x`; the wrapper must provide raw `t` to the transcoder through side context.",
        "",
        "## Stage 2 Implications",
        "",
        "- Train one timestep-conditioned transcoder per listed MLP.",
        "- Transcoder input record: one token vector `x` with shape `[1024]` plus raw flow timestep `t`.",
        "- Transcoder target: original MLP output vector `y` with shape `[1024]`.",
        "- Capture tensors before/after each listed `GemmaMLP`.",
        "- Capture `t` by wrapping `forward`/`denoise_step` or `embed_suffix`, because the MLP module itself does not receive `t` as an argument.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    result = inspect(args.repo_id, args.local_files_only)
    markdown = render_markdown(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown)

    print(f"repo_id={result['repo_id']}")
    print(f"expert_mlp_count={result['expert_mlp_count']}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()

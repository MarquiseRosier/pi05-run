#!/usr/bin/env python
"""Run positive-example causal ablations for a Pi0.5 transcoder circuit."""

from __future__ import annotations

import argparse
import csv
import html
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import default_collate

from lerobot.datasets.factory import make_dataset
from lerobot.policies import make_policy
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from pi05_mi.circuit_tracing import FeatureNode, TraceForwardCache, parse_feature_key, top_examples_for_target
from pi05_mi.feature_discovery import batch_size_from_raw_batch, observation_metadata
from pi05_mi.patch_pi05 import Pi05TranscoderContext, install_pi05_action_expert_wrappers
from pi05_mi.transcoders import TimeConditionedTranscoder, TimeConditionedTranscoderConfig
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


DEFAULT_CHECKPOINT = Path(
    "outputs/transcoders/pi05_libero/allframes_80-10-10_epoch1_b8_exp16_latest_lambda1e-4/step_027233.pt"
)
DEFAULT_FEATURE_DIR = Path("outputs/features/pi05_libero/train_80_top20_exp16_lambda1e-4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", default=DEFAULT_POLICY_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--graph-json", type=Path, required=True)
    parser.add_argument("--target", default=None, help="Feature key such as L11:tau0.7:F9970. Defaults to graph config.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--example-source",
        choices=("top-positive", "low-target", "hard-task"),
        default="top-positive",
        help="Use target Top-K positives or scan for low-target-activation contrastives.",
    )
    parser.add_argument("--top-examples", type=int, default=20)
    parser.add_argument("--num-examples", type=int, default=20)
    parser.add_argument("--scan-batch-size", type=int, default=8)
    parser.add_argument("--max-scan-batches", type=int, default=None, help="Smoke-test cap for low-target scanning.")
    parser.add_argument(
        "--disable-zero-score-early-stop",
        action="store_true",
        help="Scan all requested batches even after enough zero-score low-target examples are found.",
    )
    parser.add_argument(
        "--hard-task-queries",
        default=(
            "yellow and white mug in the microwave;"
            "white mug on the plate;"
            "pick up the book and place it in the back compartment of the caddy;"
            "put both the alphabet soup and the cream cheese box in the basket"
        ),
        help="Semicolon-separated task substrings used for --example-source hard-task.",
    )
    parser.add_argument(
        "--hard-task-max-per-query",
        type=int,
        default=0,
        help="Max examples per hard-task query. 0 balances automatically from --num-examples.",
    )
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument(
        "--random-feature-controls",
        type=int,
        default=0,
        help="Number of random same-layer/same-timestep single-feature ablation controls.",
    )
    parser.add_argument(
        "--random-circuit-controls",
        type=int,
        default=0,
        help="Number of random same-layer-count circuit ablation controls.",
    )
    parser.add_argument("--random-control-seed", type=int, default=0)
    parser.add_argument("--episodes", default=None, help="Override dataset episodes. Defaults to feature-dir config.")
    parser.add_argument("--batch-size", type=int, default=1, help="Validation currently evaluates one observation at a time.")
    parser.add_argument("--num-workers", type=int, default=0)
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


def _load_transcoders(checkpoint_path: Path, *, device: torch.device) -> dict[str, TimeConditionedTranscoder]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    transcoders: dict[str, TimeConditionedTranscoder] = {}
    for name, raw_config in checkpoint["configs"].items():
        config = TimeConditionedTranscoderConfig(**raw_config)
        transcoder = TimeConditionedTranscoder(config)
        transcoder.load_state_dict(checkpoint["state_dicts"][name])
        transcoder.to(device=device, dtype=torch.float32)
        _freeze(transcoder)
        transcoders[name] = transcoder
    return transcoders


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


def _dataset_index(dataset: Any, observation: dict[str, Any]) -> int | None:
    if "index" in observation and isinstance(observation["index"], int):
        absolute_index = int(observation["index"])
    elif "episode_index" in observation and "frame_index" in observation:
        episode = int(observation["episode_index"])
        frame = int(observation["frame_index"])
        absolute_index = int(dataset.meta.episodes["dataset_from_index"][episode]) + frame
    else:
        return None
    absolute_to_relative = getattr(dataset, "absolute_to_relative_idx", None)
    if absolute_to_relative is None:
        return absolute_index
    return absolute_to_relative.get(absolute_index)


def _collate_one(dataset: Any, index: int) -> dict[str, Any]:
    collate = lerobot_collate_fn if dataset.meta.has_language_columns else default_collate
    return collate([dataset[index]])


def _read_graph(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _target_from_graph(graph: dict[str, Any], explicit: str | None) -> FeatureNode:
    if explicit:
        return parse_feature_key(explicit)
    target_key = graph.get("config", {}).get("target")
    if not target_key:
        target_nodes = [node["node_key"] for node in graph.get("nodes", []) if node.get("kind") == "target"]
        if not target_nodes:
            raise ValueError("No --target provided and graph has no target node")
        target_key = target_nodes[0]
    return parse_feature_key(target_key)


def _intervention_nodes(graph: dict[str, Any], target: FeatureNode) -> dict[str, list[FeatureNode]]:
    parents = [
        parse_feature_key(str(node["node_key"]))
        for node in graph.get("nodes", [])
        if node.get("kind") != "target" and str(node.get("node_key")) != target.key
    ]
    return {
        "baseline": [],
        "target_ablate": [target],
        "parents_ablate": parents,
        "circuit_ablate": [target, *parents],
    }


def _nodes_by_layer(nodes: list[FeatureNode]) -> dict[int, set[int]]:
    grouped: dict[int, set[int]] = {}
    for node in nodes:
        grouped.setdefault(int(node.layer), set()).add(int(node.feature))
    return grouped


def _add_random_control_interventions(
    intervention_nodes: dict[str, list[FeatureNode]],
    *,
    target: FeatureNode,
    d_features: int,
    random_feature_controls: int,
    random_circuit_controls: int,
    seed: int,
) -> dict[str, list[FeatureNode]]:
    out = dict(intervention_nodes)
    protected = {
        (int(node.layer), int(node.feature))
        for nodes in intervention_nodes.values()
        for node in nodes
    }
    generator = torch.Generator().manual_seed(int(seed))

    def sample_feature(layer: int, used: set[tuple[int, int]]) -> int:
        for _ in range(max(1000, d_features * 2)):
            feature = int(torch.randint(0, d_features, (1,), generator=generator).item())
            key = (int(layer), feature)
            if key not in protected and key not in used:
                used.add(key)
                return feature
        raise RuntimeError(f"Could not sample unused random feature for layer {layer}")

    used_single: set[tuple[int, int]] = set()
    for index in range(max(0, int(random_feature_controls))):
        feature = sample_feature(target.layer, used_single)
        out[f"random_feature_{index:02d}"] = [
            FeatureNode(layer=target.layer, timestep=target.timestep, feature=feature)
        ]

    circuit = intervention_nodes.get("circuit_ablate", [])
    layer_counts = Counter(int(node.layer) for node in circuit)
    used_circuit: set[tuple[int, int]] = set()
    for index in range(max(0, int(random_circuit_controls))):
        nodes: list[FeatureNode] = []
        for layer, count in sorted(layer_counts.items()):
            for _ in range(count):
                feature = sample_feature(layer, used_circuit)
                nodes.append(FeatureNode(layer=layer, timestep=target.timestep, feature=feature))
        out[f"random_circuit_{index:02d}"] = nodes
    return out


class TargetScoreScan:
    """Collect max-position target feature scores from probe-mode forwards."""

    def __init__(self, target: FeatureNode):
        self.target = target
        self.pending_scores: torch.Tensor | None = None
        self.pending_positions: torch.Tensor | None = None

    def add(self, _name: str, layer: int, _preactivation: torch.Tensor, latent: torch.Tensor, timestep: torch.Tensor) -> None:
        if int(layer) != int(self.target.layer):
            return
        flat_timestep = timestep.detach().float().reshape(-1)
        if flat_timestep.numel() == 0 or abs(float(flat_timestep[0].cpu()) - float(self.target.timestep)) > 1e-4:
            return
        if latent.ndim != 3:
            raise ValueError(f"Expected latent [batch, positions, features], got {tuple(latent.shape)}")
        values = latent[..., self.target.feature].detach().float().cpu()
        scores, positions = values.max(dim=1)
        self.pending_scores = scores
        self.pending_positions = positions.to(dtype=torch.int64)

    def pop(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pending_scores is None or self.pending_positions is None:
            raise RuntimeError(f"No target score was captured for {self.target.key}")
        scores = self.pending_scores
        positions = self.pending_positions
        self.pending_scores = None
        self.pending_positions = None
        return scores, positions


def _make_intervention(nodes: list[FeatureNode], target_timestep: float):
    grouped = _nodes_by_layer(nodes)

    def intervene(_name: str, layer: int, latent: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        features = grouped.get(int(layer))
        if not features:
            return latent
        flat_timestep = timestep.detach().float().reshape(-1)
        if flat_timestep.numel() == 0 or abs(float(flat_timestep[0].cpu()) - float(target_timestep)) > 1e-4:
            return latent
        modified = latent.clone()
        for feature in features:
            if 0 <= feature < modified.shape[-1]:
                modified[..., feature] = 0
        return modified

    return intervene


def _scan_low_target_examples(
    *,
    args: argparse.Namespace,
    policy,
    dataset,
    dataloader,
    preprocessor,
    context: Pi05TranscoderContext,
    target: FeatureNode,
    exclude_tasks: set[str],
) -> list[dict[str, Any]]:
    previous_mode = context.mode
    previous_callback = context.trace_callback
    previous_capture_traces = context.capture_traces
    previous_intervention = context.latent_intervention
    context.mode = "probe"
    context.capture_traces = True
    context.latent_intervention = None
    scanner = TargetScoreScan(target)
    context.trace_callback = scanner.add

    selected: list[dict[str, Any]] = []
    started_at = time.time()
    try:
        for batch_index, raw_batch in enumerate(dataloader, start=1):
            if args.max_scan_batches is not None and batch_index > args.max_scan_batches:
                break
            prepared = _prepare_raw_batch(raw_batch, dataset.meta.camera_keys)
            batch = preprocessor(prepared)
            context.clear_records()
            with torch.no_grad():
                policy.predict_action_chunk(batch, num_steps=args.num_inference_steps)
            scores, positions = scanner.pop()
            batch_size = batch_size_from_raw_batch(prepared)
            if scores.shape[0] != batch_size:
                raise ValueError(f"Expected {batch_size} target scores, got {scores.shape[0]}")
            for row in range(batch_size):
                metadata = observation_metadata(prepared, row, camera_keys=dataset.meta.camera_keys)
                task = str(metadata.get("task", ""))
                if task in exclude_tasks:
                    continue
                selected.append(
                    {
                        **metadata,
                        "observation_id": int(metadata.get("index", len(selected))),
                        "target_position": int(positions[row]),
                        "score": float(scores[row]),
                        "target_feature_score": float(scores[row]),
                        "selection": "low-target",
                    }
                )
            if args.no_progress or batch_index == 1 or batch_index % 50 == 0:
                print(
                    f"scan batch={batch_index}/{len(dataloader)} candidates={len(selected)} "
                    f"elapsed_s={time.time() - started_at:.1f}",
                    flush=True,
                )
            if not args.disable_zero_score_early_stop:
                zero_count = sum(1 for item in selected if float(item["target_feature_score"]) <= 0.0)
                if zero_count >= args.num_examples:
                    print(
                        f"scan early stop: found {zero_count} zero-score examples for {target.key}; "
                        "zero is the minimum possible ReLU latent score",
                        flush=True,
                    )
                    break
        selected.sort(key=lambda item: (float(item["target_feature_score"]), int(item.get("index", 0))))
        return selected[: args.num_examples]
    finally:
        context.mode = previous_mode
        context.trace_callback = previous_callback
        context.capture_traces = previous_capture_traces
        context.latent_intervention = previous_intervention
        context.clear_records()


def _parse_hard_task_queries(raw: str) -> list[str]:
    queries = [item.strip().lower() for item in raw.split(";") if item.strip()]
    if not queries:
        raise ValueError("--hard-task-queries must contain at least one non-empty query")
    return queries


def _scan_hard_task_examples(
    *,
    args: argparse.Namespace,
    policy,
    dataset,
    dataloader,
    preprocessor,
    context: Pi05TranscoderContext,
    target: FeatureNode,
    exclude_tasks: set[str],
) -> list[dict[str, Any]]:
    previous_mode = context.mode
    previous_callback = context.trace_callback
    previous_capture_traces = context.capture_traces
    previous_intervention = context.latent_intervention
    context.mode = "probe"
    context.capture_traces = True
    context.latent_intervention = None
    scanner = TargetScoreScan(target)
    context.trace_callback = scanner.add

    queries = _parse_hard_task_queries(args.hard_task_queries)
    per_query_limit = args.hard_task_max_per_query
    if per_query_limit <= 0:
        per_query_limit = max(1, (args.num_examples + len(queries) - 1) // len(queries))
    buckets: dict[str, list[dict[str, Any]]] = {query: [] for query in queries}
    started_at = time.time()

    try:
        for batch_index, raw_batch in enumerate(dataloader, start=1):
            if args.max_scan_batches is not None and batch_index > args.max_scan_batches:
                break
            prepared = _prepare_raw_batch(raw_batch, dataset.meta.camera_keys)
            batch = preprocessor(prepared)
            context.clear_records()
            with torch.no_grad():
                policy.predict_action_chunk(batch, num_steps=args.num_inference_steps)
            scores, positions = scanner.pop()
            batch_size = batch_size_from_raw_batch(prepared)
            if scores.shape[0] != batch_size:
                raise ValueError(f"Expected {batch_size} target scores, got {scores.shape[0]}")
            for row in range(batch_size):
                metadata = observation_metadata(prepared, row, camera_keys=dataset.meta.camera_keys)
                task = str(metadata.get("task", ""))
                if task in exclude_tasks:
                    continue
                task_lower = task.lower()
                matched_query = next((query for query in queries if query in task_lower), None)
                if matched_query is None or len(buckets[matched_query]) >= per_query_limit:
                    continue
                buckets[matched_query].append(
                    {
                        **metadata,
                        "observation_id": int(metadata.get("index", sum(len(v) for v in buckets.values()))),
                        "target_position": int(positions[row]),
                        "score": float(scores[row]),
                        "target_feature_score": float(scores[row]),
                        "selection": "hard-task",
                        "matched_query": matched_query,
                    }
                )
            total = sum(len(items) for items in buckets.values())
            if args.no_progress or batch_index == 1 or batch_index % 50 == 0:
                bucket_text = ", ".join(f"{query[:18]}={len(items)}" for query, items in buckets.items())
                print(
                    f"hard-task scan batch={batch_index}/{len(dataloader)} candidates={total} "
                    f"[{bucket_text}] elapsed_s={time.time() - started_at:.1f}",
                    flush=True,
                )
            if total >= args.num_examples or all(len(items) >= per_query_limit for items in buckets.values()):
                print(f"hard-task scan early stop: selected {total} semantic contrastive candidates", flush=True)
                break

        selected: list[dict[str, Any]] = []
        for offset in range(per_query_limit):
            for query in queries:
                if offset < len(buckets[query]):
                    selected.append(buckets[query][offset])
                    if len(selected) >= args.num_examples:
                        return selected
        return selected[: args.num_examples]
    finally:
        context.mode = previous_mode
        context.trace_callback = previous_callback
        context.capture_traces = previous_capture_traces
        context.latent_intervention = previous_intervention
        context.clear_records()


def _run_action_chunk_with_fixed_noise(
    *,
    policy,
    batch: dict[str, torch.Tensor],
    noise: torch.Tensor,
    num_inference_steps: int,
) -> torch.Tensor:
    images, img_masks = policy._preprocess_images(batch)
    tokens = batch[OBS_LANGUAGE_TOKENS]
    masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    model = policy.model

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, tokens, masks)
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks, prepare_attention_masks_4d

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

    x_t = noise.clone()
    dt = -1.0 / int(num_inference_steps)
    for step in range(int(num_inference_steps)):
        timestep_value = 1.0 + step * dt
        timestep = torch.tensor(timestep_value, dtype=torch.float32, device=x_t.device).expand(x_t.shape[0])
        v_t = model.denoise_step(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            x_t=x_t,
            timestep=timestep,
        )
        x_t = x_t + dt * v_t

    original_action_dim = policy.config.output_features[ACTION].shape[0]
    return x_t[:, :, :original_action_dim].detach()


def _target_activation(cache: TraceForwardCache, target: FeatureNode, position: int) -> tuple[float, float]:
    record = cache.get(target.layer, target.timestep)
    if record.latent.ndim != 3:
        raise ValueError(f"Expected target latent rank 3, got {tuple(record.latent.shape)}")
    if position < 0 or position >= record.latent.shape[1]:
        raise ValueError(f"Target position {position} out of range for {target.key}")
    latent = float(record.latent[0, position, target.feature].detach().float().cpu())
    preactivation = float(record.preactivation[0, position, target.feature].detach().float().cpu())
    return latent, preactivation


def _action_metrics(baseline: torch.Tensor, intervention: torch.Tensor) -> dict[str, float]:
    delta = (intervention.float() - baseline.float()).detach().cpu()
    per_position_l2 = delta.pow(2).sum(dim=-1).sqrt()
    return {
        "action_l2": float(delta.pow(2).sum().sqrt()),
        "action_rmse": float(delta.pow(2).mean().sqrt()),
        "action_mean_abs": float(delta.abs().mean()),
        "max_position_l2": float(per_position_l2.max()),
        "mean_position_l2": float(per_position_l2.mean()),
        "max_position": int(per_position_l2.reshape(-1).argmax()),
    }


def _intervention_family(name: str) -> str:
    if name.startswith("random_feature_"):
        return "random_feature_control"
    if name.startswith("random_circuit_"):
        return "random_circuit_control"
    return name


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    preferred = [
        "intervention",
        "intervention_family",
        "observation_id",
        "episode_index",
        "frame_index",
        "task",
        "target_position",
        "baseline_target_latent",
        "intervention_target_latent",
        "target_latent_delta",
        "target_latent_ratio",
        "action_l2",
        "action_rmse",
        "max_position_l2",
        "max_position",
    ]
    fieldnames = [field for field in preferred if field in fields] + [field for field in fields if field not in preferred]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    interventions = sorted({row["intervention"] for row in rows if row["intervention"] != "baseline"})
    for intervention in interventions:
        subset = [row for row in rows if row["intervention"] == intervention]
        if not subset:
            continue
        out.append(
            {
                "intervention": intervention,
                "intervention_family": _intervention_family(intervention),
                "examples": len(subset),
                "mean_action_l2": sum(float(row["action_l2"]) for row in subset) / len(subset),
                "mean_action_rmse": sum(float(row["action_rmse"]) for row in subset) / len(subset),
                "mean_max_position_l2": sum(float(row["max_position_l2"]) for row in subset) / len(subset),
                "mean_target_latent_delta": sum(float(row["target_latent_delta"]) for row in subset) / len(subset),
                "mean_target_latent_ratio": sum(float(row["target_latent_ratio"]) for row in subset) / len(subset),
            }
        )
    return out


def _write_html(path: Path, *, config: dict[str, Any], summary: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    html_rows = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Pi0.5 Circuit Causal Validation</title>",
        "<style>body{font-family:Arial,Helvetica,sans-serif;margin:24px;color:#15171a}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid #e5e7eb;padding:8px;text-align:left;vertical-align:top}th{background:#f6f8fb}.metric{font-family:Menlo,Consolas,monospace}.muted{color:#667085}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px}.card{border:1px solid #d9dee7;padding:10px;border-radius:8px;background:#fafbfc}</style>",
        "</head><body><h1>Pi0.5 Circuit Causal Validation</h1>",
        "<p class='muted'>Positive-example ablations in the transcoder local replacement model.</p>",
        "<section class='grid'>",
    ]
    for key in ("target", "graph_json", "feature_dir", "checkpoint", "num_examples", "num_inference_steps"):
        html_rows.append(f"<div class='card'>{html.escape(key)}<br><b class='metric'>{html.escape(str(config.get(key)))}</b></div>")
    html_rows.extend(["</section>", "<h2>Aggregate</h2>", "<table><thead><tr>"])
    summary_fields = list(summary[0]) if summary else []
    html_rows.extend(f"<th>{html.escape(field)}</th>" for field in summary_fields)
    html_rows.append("</tr></thead><tbody>")
    for row in summary:
        html_rows.append("<tr>")
        for field in summary_fields:
            value = row[field]
            text = f"{value:.5g}" if isinstance(value, float) else str(value)
            html_rows.append(f"<td class='metric'>{html.escape(text)}</td>")
        html_rows.append("</tr>")
    html_rows.extend(["</tbody></table>", "<h2>Per Example</h2>", "<table><thead><tr>"])
    fields = [
        "intervention",
        "intervention_family",
        "observation_id",
        "episode_index",
        "frame_index",
        "target_position",
        "baseline_target_latent",
        "intervention_target_latent",
        "target_latent_delta",
        "action_l2",
        "action_rmse",
        "max_position",
        "task",
    ]
    html_rows.extend(f"<th>{html.escape(field)}</th>" for field in fields)
    html_rows.append("</tr></thead><tbody>")
    for row in rows:
        if row["intervention"] == "baseline":
            continue
        html_rows.append("<tr>")
        for field in fields:
            value = row.get(field, "")
            text = f"{value:.5g}" if isinstance(value, float) else str(value)
            html_rows.append(f"<td>{html.escape(text)}</td>")
        html_rows.append("</tr>")
    html_rows.extend(["</tbody></table>", "</body></html>"])
    path.write_text("\n".join(html_rows))


def main() -> None:
    args = parse_args()
    patch_transformers_causal_mask_compat()
    patch_pi05_checkpoint_key_compat()

    graph = _read_graph(args.graph_json)
    target = _target_from_graph(graph, args.target)
    intervention_nodes = _intervention_nodes(graph, target)
    episodes = args.episodes or _feature_dir_episodes(args.feature_dir)

    print("Causal validation plan", flush=True)
    print(f"target={target.key}", flush=True)
    print(f"graph_json={args.graph_json}", flush=True)
    print(f"feature_dir={args.feature_dir}", flush=True)
    print(f"checkpoint={args.checkpoint}", flush=True)
    print(f"episodes={episodes or 'all'}", flush=True)
    print(
        "interventions="
        + ", ".join(f"{name}:{len(nodes)} nodes" for name, nodes in intervention_nodes.items()),
        flush=True,
    )
    if args.plan_only:
        print("plan-only requested; exiting before model load", flush=True)
        return

    positive_examples = top_examples_for_target(feature_dir=args.feature_dir, target=target, top_examples=args.top_examples)
    if not positive_examples:
        raise RuntimeError(f"No top examples found for {target.key}")

    device = resolve_device(args.device)
    args.resolved_device = device
    args.resolved_policy_dtype = resolve_policy_dtype(args.policy_dtype, device)
    cfg = _configure_train_config(args, episodes=episodes)
    cfg.policy.pretrained_path = Path(args.policy_path)
    cfg.policy.device = str(device)
    cfg.policy.dtype = args.resolved_policy_dtype
    cfg.policy.compile_model = False
    cfg.policy.gradient_checkpointing = False

    print(f"loading dataset {cfg.dataset.repo_id} episodes={_episode_summary(_parse_episode_ids(episodes))}", flush=True)
    dataset = make_dataset(cfg)

    print("loading preprocessor/tokenizer", flush=True)
    preprocessor = _make_preprocessor(cfg, dataset, args.policy_path)
    print("loading frozen Pi0.5 policy weights", flush=True)
    policy = make_policy(cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    _freeze(policy)

    print(f"loading transcoders from {args.checkpoint}", flush=True)
    transcoders = _load_transcoders(args.checkpoint, device=device)
    if args.random_feature_controls or args.random_circuit_controls:
        if not transcoders:
            raise RuntimeError("Random controls require loaded transcoders to infer latent dimension")
        d_features = int(next(iter(transcoders.values())).config.latent_dim)
        intervention_nodes = _add_random_control_interventions(
            intervention_nodes,
            target=target,
            d_features=d_features,
            random_feature_controls=args.random_feature_controls,
            random_circuit_controls=args.random_circuit_controls,
            seed=args.random_control_seed,
        )
        print(
            "added random controls: "
            f"single={args.random_feature_controls} circuit={args.random_circuit_controls} "
            f"d_features={d_features} seed={args.random_control_seed}",
            flush=True,
        )
        print(
            "interventions_after_controls="
            + ", ".join(f"{name}:{len(nodes)} nodes" for name, nodes in intervention_nodes.items()),
            flush=True,
        )
    context = Pi05TranscoderContext(
        mode="replace",
        capture_records=False,
        capture_latents=False,
        capture_traces=True,
        store_latent_summaries=False,
    )
    _context, wrapped_names = install_pi05_action_expert_wrappers(
        policy,
        context=context,
        transcoders=transcoders,
        mode="replace",
    )
    print(f"installed {len(wrapped_names)} replacement transcoders", flush=True)

    if args.example_source == "top-positive":
        examples = positive_examples[: args.num_examples]
        for example in examples:
            example["selection"] = "top-positive"
    elif args.example_source == "low-target":
        scan_batch_size = cfg.batch_size
        cfg.batch_size = args.scan_batch_size
        cfg.num_workers = args.num_workers
        dataloader = _make_dataloader(cfg, dataset)
        exclude_tasks = {str(example.get("task", "")) for example in positive_examples if example.get("task")}
        print(
            f"scanning low-target contrastives batches={len(dataloader)} batch_size={args.scan_batch_size} "
            f"exclude_tasks={len(exclude_tasks)}",
            flush=True,
        )
        examples = _scan_low_target_examples(
            args=args,
            policy=policy,
            dataset=dataset,
            dataloader=dataloader,
            preprocessor=preprocessor,
            context=context,
            target=target,
            exclude_tasks=exclude_tasks,
        )
        cfg.batch_size = scan_batch_size
    else:
        scan_batch_size = cfg.batch_size
        cfg.batch_size = args.scan_batch_size
        cfg.num_workers = args.num_workers
        dataloader = _make_dataloader(cfg, dataset)
        exclude_tasks = {str(example.get("task", "")) for example in positive_examples if example.get("task")}
        print(
            f"scanning hard semantic contrastives batches={len(dataloader)} batch_size={args.scan_batch_size} "
            f"queries={_parse_hard_task_queries(args.hard_task_queries)}",
            flush=True,
        )
        examples = _scan_hard_task_examples(
            args=args,
            policy=policy,
            dataset=dataset,
            dataloader=dataloader,
            preprocessor=preprocessor,
            context=context,
            target=target,
            exclude_tasks=exclude_tasks,
        )
        cfg.batch_size = scan_batch_size

    if not examples:
        raise RuntimeError(f"No validation examples selected for source={args.example_source}")
    example_indices: list[int] = []
    for example in examples:
        index = _dataset_index(dataset, example)
        if index is None:
            raise RuntimeError(f"Could not map observation {example.get('observation_id')} to a dataset row")
        example_indices.append(index)
    selected_path = args.output_dir / "selected_examples.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with selected_path.open("w") as f:
        for example in examples:
            f.write(json.dumps(example, sort_keys=True) + "\n")
    print(f"selected {len(examples)} examples source={args.example_source} wrote {selected_path}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    started_at = time.time()

    for offset, (example, index) in enumerate(zip(examples, example_indices, strict=True), start=1):
        raw_batch = _collate_one(dataset, index)
        raw_batch = _prepare_raw_batch(raw_batch, dataset.meta.camera_keys)
        batch = preprocessor(raw_batch)
        batch_size = batch[OBS_LANGUAGE_TOKENS].shape[0]
        actions_shape = (batch_size, policy.model.config.chunk_size, policy.model.config.max_action_dim)
        noise = policy.model.sample_noise(actions_shape, device)
        target_position = int(example["target_position"])

        baseline_cache = TraceForwardCache()
        context.trace_callback = baseline_cache.add
        context.latent_intervention = None
        context.clear_records()
        with torch.no_grad():
            baseline_actions = _run_action_chunk_with_fixed_noise(
                policy=policy,
                batch=batch,
                noise=noise,
                num_inference_steps=args.num_inference_steps,
            )
        baseline_latent, baseline_preactivation = _target_activation(baseline_cache, target, target_position)
        baseline_row = {
            "example_source": args.example_source,
            "selection": example.get("selection", args.example_source),
            "target_feature_score": example.get("target_feature_score", example.get("score")),
            "intervention": "baseline",
            "intervention_family": "baseline",
            "observation_id": int(example["observation_id"]),
            "episode_index": example.get("episode_index"),
            "frame_index": example.get("frame_index"),
            "timestamp": example.get("timestamp"),
            "task": example.get("task", ""),
            "target_position": target_position,
            "baseline_target_latent": baseline_latent,
            "intervention_target_latent": baseline_latent,
            "target_latent_delta": 0.0,
            "target_latent_ratio": 1.0,
            "baseline_target_preactivation": baseline_preactivation,
            "intervention_target_preactivation": baseline_preactivation,
            "action_l2": 0.0,
            "action_rmse": 0.0,
            "action_mean_abs": 0.0,
            "max_position_l2": 0.0,
            "mean_position_l2": 0.0,
            "max_position": 0,
        }
        rows.append(baseline_row)

        for intervention, nodes in intervention_nodes.items():
            if intervention == "baseline":
                continue
            cache = TraceForwardCache()
            context.trace_callback = cache.add
            context.latent_intervention = _make_intervention(nodes, target.timestep)
            context.clear_records()
            with torch.no_grad():
                intervention_actions = _run_action_chunk_with_fixed_noise(
                    policy=policy,
                    batch=batch,
                    noise=noise,
                    num_inference_steps=args.num_inference_steps,
                )
            latent, preactivation = _target_activation(cache, target, target_position)
            metric_row = {
                **baseline_row,
                "intervention": intervention,
                "intervention_family": _intervention_family(intervention),
                "intervention_target_latent": latent,
                "target_latent_delta": latent - baseline_latent,
                "target_latent_ratio": latent / baseline_latent if abs(baseline_latent) > 1e-8 else 0.0,
                "intervention_target_preactivation": preactivation,
                "target_preactivation_delta": preactivation - baseline_preactivation,
                "ablated_nodes": len(nodes),
                **_action_metrics(baseline_actions, intervention_actions),
            }
            rows.append(metric_row)

        context.latent_intervention = None
        context.trace_callback = None
        if args.no_progress or offset == 1 or offset == len(examples):
            print(
                f"example {offset}/{len(examples)} observation={example['observation_id']} "
                f"elapsed_s={time.time() - started_at:.1f}",
                flush=True,
            )

    summary = _aggregate(rows)
    config = {
        "target": target.key,
        "graph_json": str(args.graph_json),
        "feature_dir": str(args.feature_dir),
        "checkpoint": str(args.checkpoint),
        "num_examples": len(examples),
        "example_source": args.example_source,
        "num_inference_steps": args.num_inference_steps,
        "interventions": {name: [node.key for node in nodes] for name, nodes in intervention_nodes.items()},
    }

    _write_csv(args.output_dir / "causal_validation_rows.csv", rows)
    _write_csv(args.output_dir / "causal_validation_summary.csv", summary)
    with (args.output_dir / "causal_validation_results.json").open("w") as f:
        json.dump({"config": config, "summary": summary, "rows": rows}, f, indent=2, sort_keys=True)
        f.write("\n")
    _write_html(args.output_dir / "causal_validation_report.html", config=config, summary=summary, rows=rows)
    print(f"saved causal validation outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

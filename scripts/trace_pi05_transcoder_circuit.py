#!/usr/bin/env python
"""Trace sparse-feature parents in the Pi0.5 transcoder replacement model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import default_collate

from lerobot.datasets.factory import make_dataset
from lerobot.policies import make_policy
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from pi05_mi.circuit_tracing import (
    FeatureNode,
    TraceForwardCache,
    aggregate_parent_contributions,
    node_influence_scores,
    parent_summary_to_edge,
    parse_feature_key,
    prune_edges_by_cumulative_influence,
    prune_nodes_by_cumulative_influence,
    source_contributions_to_layers,
    source_contributions_to_target,
    top_examples_for_target,
    write_trace_outputs,
)
from pi05_mi.patch_pi05 import Pi05TranscoderContext, install_pi05_action_expert_wrappers
from pi05_mi.transcoders import TimeConditionedTranscoder, TimeConditionedTranscoderConfig
from train_pi05_transcoders import (
    DEFAULT_POLICY_PATH,
    _config_with_episodes,
    _configure_train_config,
    _episode_summary,
    _make_preprocessor,
    _parse_episode_ids,
    _prepare_raw_batch,
    patch_pi05_checkpoint_key_compat,
    patch_transformers_causal_mask_compat,
    resolve_device,
    resolve_policy_dtype,
)


DEFAULT_FEATURE_DIR = Path("outputs/features/pi05_libero/train_80_top20_exp16_lambda1e-4")
DEFAULT_OUTPUT_DIR = Path("outputs/circuits/pi05_libero")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", default=DEFAULT_POLICY_PATH)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--target", required=True, help="Feature key such as L12:tau1:F7584.")
    parser.add_argument("--top-examples", type=int, default=20)
    parser.add_argument(
        "--trace-mode",
        choices=("diffract-frontier", "fixed-k"),
        default="diffract-frontier",
        help="diffract-frontier expands high-influence discovered nodes; fixed-k keeps the original debug trace.",
    )
    parser.add_argument("--parents-per-node", type=int, default=2)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-nodes", type=int, default=100, help="Max expanded nodes / VJP calls in frontier mode.")
    parser.add_argument("--expansion-batch-size", type=int, default=10)
    parser.add_argument("--min-attribution", type=float, default=1e-4)
    parser.add_argument("--node-cumulative-threshold", type=float, default=0.8)
    parser.add_argument("--edge-cumulative-threshold", type=float, default=0.98)
    parser.add_argument("--first-layer", type=int, default=0)
    parser.add_argument(
        "--source-policy",
        choices=("previous-layer", "all-earlier"),
        default="all-earlier",
        help="Which source layers to consider for each target node.",
    )
    parser.add_argument(
        "--edge-score-metric",
        choices=("mean_abs", "mean_abs_frequency"),
        default="mean_abs",
    )
    parser.add_argument(
        "--example-top-m",
        type=int,
        default=None,
        help="Per-example local parent count used for frequency_in_example_topk. Defaults to --parents-per-node.",
    )
    parser.add_argument("--episodes", default=None, help="Override dataset episodes. Defaults to feature-dir config.")
    parser.add_argument("--batch-size", type=int, default=1, help="Tracing currently uses one observation at a time.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-inference-steps", type=int, default=10)
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


def _run_replacement_inference_timestep(
    *,
    policy,
    batch: dict[str, torch.Tensor],
    target_timestep: float,
    num_inference_steps: int,
    trace_cache: TraceForwardCache,
) -> None:
    images, img_masks = policy._preprocess_images(batch)
    tokens = batch[OBS_LANGUAGE_TOKENS]
    masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    model = policy.model
    batch_size = tokens.shape[0]
    device = tokens.device

    with torch.no_grad():
        actions_shape = (batch_size, model.config.chunk_size, model.config.max_action_dim)
        x_t = model.sample_noise(actions_shape, device)
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, tokens, masks)

    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks, prepare_attention_masks_4d

    with torch.no_grad():
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

    dt = -1.0 / num_inference_steps
    matched = False
    for step in range(num_inference_steps):
        timestep_value = 1.0 + step * dt
        timestep = torch.tensor(timestep_value, dtype=torch.float32, device=device).expand(batch_size)
        if abs(timestep_value - target_timestep) <= 1e-4:
            matched = True
            x_t = x_t.detach().requires_grad_(True)
            with torch.enable_grad():
                model.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=x_t,
                    timestep=timestep,
                )
            return

        with torch.no_grad():
            v_t = model.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=timestep,
            )
            x_t = x_t + dt * v_t

    if not matched:
        raise RuntimeError(
            f"Target timestep {target_timestep:.8f} was not visited by {num_inference_steps} inference steps"
        )


def _format_float(value: object, digits: int = 5) -> object:
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return value


def _render_svg_graph(output_dir: Path, *, graph: dict[str, Any]) -> None:
    nodes = graph["nodes"]
    edges = graph["edges"]
    if not nodes:
        return

    nodes_by_key = {node["node_key"]: node for node in nodes}
    layers = sorted({int(node["layer"]) for node in nodes})
    layer_to_x = {
        layer: 80 + idx * 220
        for idx, layer in enumerate(layers)
    }
    by_layer: dict[int, list[dict[str, Any]]] = {}
    for node in nodes:
        by_layer.setdefault(int(node["layer"]), []).append(node)
    for layer_nodes in by_layer.values():
        layer_nodes.sort(key=lambda node: (-float(node.get("influence") or 0.0), int(node["feature"])))

    coords: dict[str, tuple[int, int]] = {}
    max_rows = max(len(layer_nodes) for layer_nodes in by_layer.values())
    width = max(420, 160 + 220 * max(1, len(layers)))
    height = max(260, 130 + 96 * max_rows)
    for layer, layer_nodes in by_layer.items():
        x = layer_to_x[layer]
        for idx, node in enumerate(layer_nodes):
            coords[node["node_key"]] = (x, 74 + idx * 92)

    svg: list[str] = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        "<defs><marker id='arrow' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' orient='auto-start-reverse'><path d='M 0 0 L 10 5 L 0 10 z' fill='#536171'/></marker></defs>",
        "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#172033}.node{fill:#f7fbff;stroke:#1f66c2;stroke-width:1.6}.target{fill:#fff7ed;stroke:#c45a10;stroke-width:2}.edge{stroke:#536171;stroke-width:1.25;fill:none;marker-end:url(#arrow)}.label{font-size:11px;fill:#1e293b}.small{font-size:10px;fill:#586170}.title{font-size:18px;font-weight:700}</style>",
        "<text x='28' y='30' class='title'>Pi0.5 Transcoder Circuit Trace</text>",
    ]
    for edge in edges:
        source = str(edge["source_key"])
        target = str(edge["target_key"])
        if source not in coords or target not in coords:
            continue
        x1, y1 = coords[source]
        x2, y2 = coords[target]
        start_x = x1 + 130
        end_x = x2 - 10
        start_y = y1 + 24
        end_y = y2 + 24
        svg.append(f"<path class='edge' d='M {start_x} {start_y} C {(start_x + end_x) / 2:.1f} {start_y}, {(start_x + end_x) / 2:.1f} {end_y}, {end_x} {end_y}'/>")
        value = edge.get("edge_mass", edge.get("edge_score", edge.get("attribution", 0.0)))
        svg.append(f"<text class='small' x='{(start_x + end_x) / 2 - 16:.1f}' y='{(start_y + end_y) / 2 - 4:.1f}'>{float(value):.3g}</text>")

    for key, (x, y) in coords.items():
        node = nodes_by_key[key]
        cls = "target" if node.get("kind") == "target" else "node"
        influence = node.get("influence")
        svg.append(f"<rect class='{cls}' x='{x}' y='{y}' rx='0' ry='0' width='130' height='52'/>")
        svg.append(f"<text class='label' x='{x + 8}' y='{y + 19}'>L{int(node['layer']):02d}:tau{float(node['timestep']):.4g}</text>")
        svg.append(f"<text class='label' x='{x + 8}' y='{y + 37}'>F{int(node['feature'])}</text>")
        if influence is not None:
            svg.append(f"<text class='small' x='{x + 80}' y='{y + 37}'>I={float(influence):.2g}</text>")
    svg.append("</svg>")

    (output_dir / "circuit_graph.svg").write_text("\n".join(svg))
    (output_dir / "circuit_graph.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Pi0.5 Circuit Graph</title>"
        "<body style='margin:0;background:white'>"
        + "\n".join(svg)
        + "</body>"
    )


def _render_html(output_dir: Path, *, graph: dict[str, Any]) -> None:
    rows = [
        "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Pi0.5 Circuit Trace</title>",
        "<style>body{font-family:Arial,Helvetica,sans-serif;margin:0;background:#fff;color:#15171a}main{max-width:1200px;margin:0 auto;padding:24px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid #e4e8ef;padding:8px;text-align:left}th{background:#f5f7fa}.metric{font-family:Menlo,Consolas,monospace}.muted{color:#657080}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}.stat{border:1px solid #d9dee7;border-radius:8px;padding:10px;background:#fafbfc}</style>",
        "</head><body><main>",
        "<h1>Pi0.5 Transcoder Circuit Trace</h1>",
        "<p class='muted'>Sparse-feature parents ranked by <span class='metric'>activation * gradient</span> inside the transcoder replacement model.</p>",
        "<section class='grid'>",
    ]
    config = graph["config"]
    for label, key in [
        ("Target", "target"),
        ("Trace mode", "trace_mode"),
        ("Top examples", "top_examples"),
        ("Parents per node", "parents_per_node"),
        ("Max depth", "max_depth"),
        ("Max nodes", "max_nodes"),
        ("Inference steps", "num_inference_steps"),
    ]:
        rows.append(f"<div class='stat'>{label}<br><b class='metric'>{config.get(key)}</b></div>")
    rows.extend(["</section>", "<h2>Edges</h2>", "<table><thead><tr>"])
    fields = [
        "depth",
        "rank",
        "source_key",
        "target_key",
        "edge_mass",
        "attribution",
        "prune_edge_score",
        "edge_score",
        "mean_abs_contribution",
        "mean_signed_contribution",
        "frequency_in_example_topk",
        "examples",
    ]
    rows.extend(f"<th>{field}</th>" for field in fields)
    rows.append("</tr></thead><tbody>")
    for edge in graph["edges"]:
        rows.append("<tr>")
        for field in fields:
            value = _format_float(edge.get(field))
            rows.append(f"<td class='metric'>{value}</td>")
        rows.append("</tr>")
    rows.extend(["</tbody></table>", "<h2>Nodes</h2>", "<table><thead><tr>"])
    node_fields = ["depth", "kind", "expanded", "influence", "node_key", "layer", "timestep", "feature"]
    rows.extend(f"<th>{field}</th>" for field in node_fields)
    rows.append("</tr></thead><tbody>")
    for node in graph["nodes"]:
        rows.append("<tr>")
        for field in node_fields:
            rows.append(f"<td class='metric'>{_format_float(node.get(field))}</td>")
        rows.append("</tr>")
    rows.extend(["</tbody></table>", "</main></body></html>"])
    (output_dir / "circuit_report.html").write_text("\n".join(rows))


def _trace_fixed_k(
    *,
    policy,
    preprocessor,
    context: Pi05TranscoderContext,
    raw_batches: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    target: FeatureNode,
    args: argparse.Namespace,
    example_top_m: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    nodes: dict[str, dict[str, Any]] = {
        target.key: {
            "node_key": target.key,
            "layer": target.layer,
            "timestep": target.timestep,
            "feature": target.feature,
            "depth": 0,
            "kind": "target",
            "label": "",
            "expanded": False,
            "influence": 1.0,
        }
    }
    edges: list[dict[str, Any]] = []
    positions_by_node: dict[str, dict[int, int]] = {
        target.key: {int(example["observation_id"]): int(example["target_position"]) for example in examples}
    }
    frontier = [target]

    for depth in range(1, args.max_depth + 1):
        next_frontier: list[FeatureNode] = []
        for current in frontier:
            source_layer = current.layer - 1
            if source_layer < args.first_layer:
                continue

            abs_values: list[torch.Tensor] = []
            signed_values: list[torch.Tensor] = []
            source_positions: list[torch.Tensor] = []
            per_example_top: list[torch.Tensor] = []
            used_observation_ids: list[int] = []

            for example, raw_batch in zip(examples, raw_batches, strict=True):
                observation_id = int(example["observation_id"])
                target_position = positions_by_node.get(current.key, {}).get(observation_id)
                if target_position is None:
                    continue
                batch = preprocessor(raw_batch)
                trace_cache = TraceForwardCache()
                context.trace_callback = trace_cache.add
                context.clear_records()
                _run_replacement_inference_timestep(
                    policy=policy,
                    batch=batch,
                    target_timestep=current.timestep,
                    num_inference_steps=args.num_inference_steps,
                    trace_cache=trace_cache,
                )
                contribution = source_contributions_to_target(
                    cache=trace_cache,
                    source_layer=source_layer,
                    target=current,
                    target_position=target_position,
                    top_m_per_example=example_top_m,
                )
                abs_values.append(contribution["collapsed_abs"])
                signed_values.append(contribution["signed_contribution"])
                source_positions.append(contribution["source_positions"])
                per_example_top.append(contribution["per_example_top"])
                used_observation_ids.append(observation_id)

            parents, parent_positions = aggregate_parent_contributions(
                source_layer=source_layer,
                target=current,
                collapsed_abs_values=abs_values,
                signed_values=signed_values,
                source_positions=source_positions,
                per_example_top=per_example_top,
                parents_per_node=args.parents_per_node,
                edge_score_metric=args.edge_score_metric,
            )
            nodes[current.key]["expanded"] = True
            for rank, parent in enumerate(parents, start=1):
                edge = parent_summary_to_edge(parent, depth=depth, rank=rank)
                edge["edge_mass"] = parent.mean_abs_contribution
                edge["attribution"] = parent.mean_signed_contribution
                edges.append(edge)
                parent_key = parent.source.key
                nodes.setdefault(
                    parent_key,
                    {
                        "node_key": parent_key,
                        "layer": parent.source.layer,
                        "timestep": parent.source.timestep,
                        "feature": parent.source.feature,
                        "depth": depth,
                        "kind": "parent",
                        "label": "",
                        "expanded": False,
                        "influence": None,
                    },
                )
                if parent_key not in positions_by_node:
                    feature_positions = parent_positions[parent.source.feature]
                    positions_by_node[parent_key] = {
                        observation_id: int(position)
                        for observation_id, position in zip(used_observation_ids, feature_positions, strict=True)
                    }
                    next_frontier.append(parent.source)

                print(
                    f"depth={depth} rank={rank} {parent.source.key} -> {current.key} "
                    f"score={parent.edge_score:.5g}",
                    flush=True,
                )
        frontier = next_frontier
        if not frontier:
            break

    influence = node_influence_scores(nodes, edges, target_key=target.key)
    for key, score in influence.items():
        if key in nodes:
            nodes[key]["influence"] = score
    return nodes, edges


def _trace_diffract_frontier(
    *,
    policy,
    preprocessor,
    context: Pi05TranscoderContext,
    raw_batches: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    target: FeatureNode,
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    target_key = target.key
    nodes: dict[str, dict[str, Any]] = {
        target_key: {
            "node_key": target_key,
            "layer": target.layer,
            "timestep": target.timestep,
            "feature": target.feature,
            "depth": 0,
            "kind": "target",
            "label": "",
            "expanded": False,
            "influence": 1.0,
        }
    }
    node_objects: dict[str, FeatureNode] = {target_key: target}
    positions_by_node: dict[str, dict[int, int]] = {
        target_key: {int(example["observation_id"]): int(example["target_position"]) for example in examples}
    }
    position_scores: dict[str, dict[int, float]] = {target_key: {int(example["observation_id"]): float("inf") for example in examples}}
    edges: list[dict[str, Any]] = []
    expanded: set[str] = set()

    def candidate_keys() -> list[str]:
        scores = node_influence_scores(nodes, edges, target_key=target_key)
        for key, score in scores.items():
            if key in nodes:
                nodes[key]["influence"] = score
        candidates = [
            key
            for key, node in nodes.items()
            if key not in expanded and int(node["layer"]) > args.first_layer
        ]
        candidates.sort(key=lambda key: float(nodes[key].get("influence") or 0.0), reverse=True)
        return candidates

    while len(expanded) < args.max_nodes:
        batch_keys = candidate_keys()[: max(1, args.expansion_batch_size)]
        if not batch_keys:
            break
        for current_key in batch_keys:
            if len(expanded) >= args.max_nodes:
                break
            current = node_objects[current_key]
            if args.source_policy == "previous-layer":
                source_layers = range(max(args.first_layer, current.layer - 1), current.layer)
            else:
                source_layers = range(args.first_layer, current.layer)
            by_layer_abs: dict[int, list[torch.Tensor]] = {}
            by_layer_signed: dict[int, list[torch.Tensor]] = {}
            by_layer_positions: dict[int, list[torch.Tensor]] = {}
            by_layer_observation_ids: dict[int, list[int]] = {}

            for example, raw_batch in zip(examples, raw_batches, strict=True):
                observation_id = int(example["observation_id"])
                target_position = positions_by_node.get(current_key, {}).get(observation_id)
                if target_position is None:
                    continue
                batch = preprocessor(raw_batch)
                trace_cache = TraceForwardCache()
                context.trace_callback = trace_cache.add
                context.clear_records()
                _run_replacement_inference_timestep(
                    policy=policy,
                    batch=batch,
                    target_timestep=current.timestep,
                    num_inference_steps=args.num_inference_steps,
                    trace_cache=trace_cache,
                )
                contributions = source_contributions_to_layers(
                    cache=trace_cache,
                    source_layers=source_layers,
                    target=current,
                    target_position=target_position,
                )
                if not contributions:
                    continue
                for source_layer, contribution in contributions.items():
                    by_layer_abs.setdefault(source_layer, []).append(contribution["collapsed_abs"])
                    by_layer_signed.setdefault(source_layer, []).append(contribution["signed_contribution"])
                    by_layer_positions.setdefault(source_layer, []).append(contribution["source_positions"])
                    by_layer_observation_ids.setdefault(source_layer, []).append(observation_id)

            discovered = 0
            for source_layer, abs_values in sorted(by_layer_abs.items()):
                signed_values = by_layer_signed[source_layer]
                source_position_values = by_layer_positions[source_layer]
                if not abs_values:
                    continue
                mean_abs = torch.stack(abs_values, dim=0).float().mean(dim=0)
                mean_signed = torch.stack(signed_values, dim=0).float().mean(dim=0)
                positions = torch.stack(source_position_values, dim=0).to(dtype=torch.int64)
                used_observation_ids = by_layer_observation_ids[source_layer]
                keep_features = torch.nonzero(mean_abs >= args.min_attribution, as_tuple=False).flatten().tolist()
                for feature in keep_features:
                    source = FeatureNode(layer=source_layer, timestep=current.timestep, feature=int(feature))
                    source_key = source.key
                    edge_mass = float(mean_abs[feature])
                    attribution = float(mean_signed[feature])
                    edge = {
                        "source_key": source_key,
                        "target_key": current_key,
                        "source_layer": source.layer,
                        "target_layer": current.layer,
                        "timestep": current.timestep,
                        "source_feature": source.feature,
                        "target_feature": current.feature,
                        "depth": int(nodes[current_key]["depth"]) + 1,
                        "rank": None,
                        "edge_score": edge_mass,
                        "edge_mass": edge_mass,
                        "attribution": attribution,
                        "mean_abs_contribution": edge_mass,
                        "mean_signed_contribution": attribution,
                        "std_signed_contribution": float(torch.stack(signed_values, dim=0).float()[:, feature].std(unbiased=False)),
                        "frequency_in_example_topk": None,
                        "examples": len(abs_values),
                    }
                    edges.append(edge)
                    discovered += 1

                    nodes.setdefault(
                        source_key,
                        {
                            "node_key": source_key,
                            "layer": source.layer,
                            "timestep": source.timestep,
                            "feature": source.feature,
                            "depth": int(nodes[current_key]["depth"]) + 1,
                            "kind": "parent",
                            "label": "",
                            "expanded": False,
                            "influence": None,
                        },
                    )
                    node_objects.setdefault(source_key, source)
                    positions_by_node.setdefault(source_key, {})
                    position_scores.setdefault(source_key, {})
                    for observation_id, position in zip(used_observation_ids, positions[:, feature].tolist(), strict=False):
                        if edge_mass > position_scores[source_key].get(observation_id, float("-inf")):
                            positions_by_node[source_key][observation_id] = int(position)
                            position_scores[source_key][observation_id] = edge_mass

            expanded.add(current_key)
            nodes[current_key]["expanded"] = True
            scores = node_influence_scores(nodes, edges, target_key=target_key)
            for key, score in scores.items():
                if key in nodes:
                    nodes[key]["influence"] = score
            if not args.no_progress:
                print(
                    f"expanded={len(expanded)}/{args.max_nodes} {current_key} "
                    f"new_edges={discovered} graph_nodes={len(nodes)} graph_edges={len(edges)}",
                    flush=True,
                )

    keep_nodes = prune_nodes_by_cumulative_influence(
        nodes,
        edges,
        target_key=target_key,
        threshold=args.node_cumulative_threshold,
    )
    node_pruned = {key: value for key, value in nodes.items() if key in keep_nodes}
    edge_node_pruned = [
        edge
        for edge in edges
        if str(edge["source_key"]) in node_pruned and str(edge["target_key"]) in node_pruned
    ]
    edge_pruned = prune_edges_by_cumulative_influence(
        node_pruned,
        edge_node_pruned,
        target_key=target_key,
        threshold=args.edge_cumulative_threshold,
    )
    referenced = {target_key}
    for edge in edge_pruned:
        referenced.add(str(edge["source_key"]))
        referenced.add(str(edge["target_key"]))
    final_nodes = {key: value for key, value in node_pruned.items() if key in referenced}
    if not args.no_progress:
        print(
            f"pruned graph nodes={len(nodes)}->{len(final_nodes)} edges={len(edges)}->{len(edge_pruned)}",
            flush=True,
        )
    return final_nodes, edge_pruned


def main() -> None:
    patch_transformers_causal_mask_compat()
    patch_pi05_checkpoint_key_compat()
    args = parse_args()
    if args.parents_per_node <= 0:
        raise ValueError("--parents-per-node must be positive")
    if args.max_depth < 0:
        raise ValueError("--max-depth must be non-negative")
    if args.top_examples <= 0:
        raise ValueError("--top-examples must be positive")
    if args.max_nodes <= 0:
        raise ValueError("--max-nodes must be positive")
    if args.expansion_batch_size <= 0:
        raise ValueError("--expansion-batch-size must be positive")
    if args.first_layer < 0:
        raise ValueError("--first-layer must be non-negative")

    target = parse_feature_key(args.target)
    output_dir = args.output_dir or (DEFAULT_OUTPUT_DIR / args.target.replace(":", "_"))
    example_top_m = args.example_top_m if args.example_top_m is not None else args.parents_per_node
    episodes = args.episodes or _feature_dir_episodes(args.feature_dir)

    print("Circuit tracing plan", flush=True)
    print(f"target={target.key}", flush=True)
    print(f"feature_dir={args.feature_dir}", flush=True)
    print(f"checkpoint={args.checkpoint}", flush=True)
    print(f"episodes={episodes or 'all'}", flush=True)
    print(
        f"replacement_model=true trace_mode={args.trace_mode} top_examples={args.top_examples} "
        f"parents_per_node={args.parents_per_node} max_depth={args.max_depth} max_nodes={args.max_nodes} "
        f"min_attribution={args.min_attribution} num_inference_steps={args.num_inference_steps}",
        flush=True,
    )
    if args.plan_only:
        print("plan-only requested; exiting before model load", flush=True)
        return

    examples = top_examples_for_target(
        feature_dir=args.feature_dir,
        target=target,
        top_examples=args.top_examples,
    )
    if not examples:
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
    example_indices = []
    for example in examples:
        index = _dataset_index(dataset, example)
        if index is None:
            raise RuntimeError(f"Could not map observation {example.get('observation_id')} to a dataset row")
        example_indices.append(index)

    print("loading preprocessor/tokenizer", flush=True)
    preprocessor = _make_preprocessor(cfg, dataset, args.policy_path)
    print("loading frozen Pi0.5 policy weights", flush=True)
    policy = make_policy(cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    _freeze(policy)

    print(f"loading transcoders from {args.checkpoint}", flush=True)
    transcoders = _load_transcoders(args.checkpoint, device=device)
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

    started_at = time.time()

    raw_batches = []
    for index in example_indices:
        raw_batch = _collate_one(dataset, index)
        raw_batches.append(_prepare_raw_batch(raw_batch, dataset.meta.camera_keys))

    if args.trace_mode == "fixed-k":
        nodes, edges = _trace_fixed_k(
            policy=policy,
            preprocessor=preprocessor,
            context=context,
            raw_batches=raw_batches,
            examples=examples,
            target=target,
            args=args,
            example_top_m=example_top_m,
        )
    else:
        nodes, edges = _trace_diffract_frontier(
            policy=policy,
            preprocessor=preprocessor,
            context=context,
            raw_batches=raw_batches,
            examples=examples,
            target=target,
            args=args,
        )

    config = {
        "target": target.key,
        "policy_path": args.policy_path,
        "checkpoint": str(args.checkpoint),
        "feature_dir": str(args.feature_dir),
        "episodes": episodes,
        "trace_mode": args.trace_mode,
        "top_examples": args.top_examples,
        "parents_per_node": args.parents_per_node,
        "max_depth": args.max_depth,
        "max_nodes": args.max_nodes,
        "expansion_batch_size": args.expansion_batch_size,
        "min_attribution": args.min_attribution,
        "node_cumulative_threshold": args.node_cumulative_threshold,
        "edge_cumulative_threshold": args.edge_cumulative_threshold,
        "first_layer": args.first_layer,
        "source_policy": args.source_policy,
        "edge_score_metric": args.edge_score_metric,
        "example_top_m": example_top_m,
        "num_inference_steps": args.num_inference_steps,
        "device": str(device),
        "policy_dtype": args.resolved_policy_dtype,
        "elapsed_s": time.time() - started_at,
    }
    write_trace_outputs(output_dir=output_dir, config=config, nodes=nodes, edges=edges)
    graph = json.loads((output_dir / "graph.json").read_text())
    _render_svg_graph(output_dir, graph=graph)
    _render_html(output_dir, graph=graph)
    print(f"saved circuit trace to {output_dir}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Validate Pi0.5 sparse-feature steering against a positive action prototype."""

from __future__ import annotations

import argparse
import csv
import html
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from lerobot.datasets.factory import make_dataset
from lerobot.policies import make_policy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_TOKENS

from pi05_mi.circuit_tracing import FeatureNode, TraceForwardCache, top_examples_for_target
from pi05_mi.patch_pi05 import Pi05TranscoderContext, install_pi05_action_expert_wrappers
from train_pi05_transcoders import (
    DEFAULT_POLICY_PATH,
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
from validate_pi05_circuit_causality import (
    DEFAULT_CHECKPOINT,
    DEFAULT_FEATURE_DIR,
    _add_random_control_interventions,
    _collate_one,
    _dataset_index,
    _feature_dir_episodes,
    _freeze,
    _intervention_nodes,
    _load_transcoders,
    _read_graph,
    _run_action_chunk_with_fixed_noise,
    _scan_hard_task_examples,
    _scan_low_target_examples,
    _target_activation,
    _target_from_graph,
)


DEFAULT_HARD_TASK_QUERIES = (
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate;"
    "pick up the black bowl on the wooden cabinet and place it on the plate;"
    "open the middle drawer of the cabinet;"
    "open the top drawer and put the bowl inside"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", default=DEFAULT_POLICY_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--graph-json", type=Path, default=None)
    parser.add_argument("--target", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--positive-source",
        choices=("top-examples", "behavior-phase-fast", "behavior-event-fast"),
        default="top-examples",
        help="How to select examples used to build the positive action/activation prototype.",
    )
    parser.add_argument("--top-examples", type=int, default=20)
    parser.add_argument("--num-positive-prototype-examples", type=int, default=20)
    parser.add_argument(
        "--negative-source",
        choices=("low-target", "hard-task", "behavior-phase-slow", "behavior-event-slow"),
        default="hard-task",
        help="Negative examples to steer. Default uses matched strict-ish semantic controls.",
    )
    parser.add_argument(
        "--behavior-phase",
        default=None,
        help="Event/phase label used by behavior-conditioned steering sets, e.g. late.",
    )
    parser.add_argument(
        "--behavior-event-bin",
        default=None,
        help="Matched manual event bin, e.g. drawer_close:q2.",
    )
    parser.add_argument("--positive-episodes", default=None)
    parser.add_argument("--negative-episodes", default=None)
    parser.add_argument("--num-negative-examples", type=int, default=20)
    parser.add_argument("--scan-batch-size", type=int, default=8)
    parser.add_argument("--max-scan-batches", type=int, default=None)
    parser.add_argument("--disable-zero-score-early-stop", action="store_true")
    parser.add_argument("--hard-task-queries", default=DEFAULT_HARD_TASK_QUERIES)
    parser.add_argument("--hard-task-max-per-query", type=int, default=5)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument(
        "--steer-mode",
        choices=("set-max", "add", "set"),
        default="set-max",
        help="How to inject positive sparse-feature activations into negative examples.",
    )
    parser.add_argument("--steer-scale", type=float, default=1.0)
    parser.add_argument("--random-feature-controls", type=int, default=5)
    parser.add_argument("--random-circuit-controls", type=int, default=5)
    parser.add_argument("--random-control-seed", type=int, default=9970)
    parser.add_argument("--episodes", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--policy-dtype", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _nodes_by_key(nodes: list[FeatureNode]) -> dict[str, FeatureNode]:
    return {node.key: node for node in nodes}


def _collect_node_values(cache: TraceForwardCache, nodes: list[FeatureNode]) -> dict[str, float]:
    values: dict[str, float] = {}
    for node in nodes:
        record = cache.get(node.layer, node.timestep)
        latent = record.latent.detach().float()
        values[node.key] = float(latent[..., node.feature].amax().cpu())
    return values


def _mean_dict(dicts: list[dict[str, float]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for item in dicts:
        for key, value in item.items():
            grouped[key].append(float(value))
    return {key: sum(values) / len(values) for key, values in grouped.items() if values}


def _read_behavior_observations(feature_dir: Path) -> list[dict[str, Any]]:
    observations_path = feature_dir / "observations.jsonl"
    if not observations_path.exists():
        raise FileNotFoundError(f"Missing behavior observations file: {observations_path}")
    observations: list[dict[str, Any]] = []
    with observations_path.open() as f:
        for line in f:
            if line.strip():
                observations.append(json.loads(line))
    return observations


def _behavior_phase_positive_examples(
    *,
    feature_dir: Path,
    target: FeatureNode,
    phase: str,
    top_examples: int,
    limit: int,
) -> list[dict[str, Any]]:
    examples = top_examples_for_target(feature_dir=feature_dir, target=target, top_examples=top_examples)
    phase_examples = [example for example in examples if str(example.get("event_phase", "")) == phase]
    phase_examples.sort(key=lambda item: (float(item.get("speed_mean", float("-inf"))), float(item.get("score", 0.0))), reverse=True)
    return phase_examples[:limit]


def _behavior_phase_slow_examples(*, feature_dir: Path, phase: str, limit: int) -> list[dict[str, Any]]:
    observations = [row for row in _read_behavior_observations(feature_dir) if str(row.get("event_phase", "")) == phase]
    observations.sort(key=lambda item: float(item.get("speed_mean", float("inf"))))
    out = []
    for row in observations[:limit]:
        item = dict(row)
        item["target_position"] = "infer"
        item["target_feature_score"] = "not_prescanned"
        out.append(item)
    return out


def _episode_set(raw: str | None) -> set[int] | None:
    if raw is None or not raw.strip():
        return None
    return {int(value.strip()) for value in raw.split(",") if value.strip()}


def _behavior_event_examples(
    *,
    feature_dir: Path,
    event_bin: str,
    episodes: set[int] | None,
    limit: int,
    fastest: bool,
    balance_episodes: bool = False,
) -> list[dict[str, Any]]:
    observations = []
    for row in _read_behavior_observations(feature_dir):
        if str(row.get("manual_event_bin", "")) != event_bin:
            continue
        episode = int(row.get("episode_index", -1))
        if episodes is not None and episode not in episodes:
            continue
        observations.append(row)
    observations.sort(key=lambda item: float(item.get("speed_mean", 0.0)), reverse=fastest)
    if balance_episodes:
        by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in observations:
            by_episode[int(row["episode_index"])].append(row)
        observations = []
        offset = 0
        episode_ids = sorted(by_episode)
        while len(observations) < limit:
            added = False
            for episode in episode_ids:
                rows = by_episode[episode]
                if offset < len(rows):
                    observations.append(rows[offset])
                    added = True
                    if len(observations) >= limit:
                        break
            if not added:
                break
            offset += 1
    out = []
    for row in observations[:limit]:
        item = dict(row)
        item["target_position"] = "infer"
        item["target_feature_score"] = "event_matched"
        out.append(item)
    return out


def _infer_target_position(cache: TraceForwardCache, target: FeatureNode) -> int:
    record = cache.get(target.layer, target.timestep)
    latent = record.latent.detach().float()
    if latent.ndim != 3 or latent.shape[0] != 1:
        raise ValueError(f"Expected target latent shaped [1, positions, features], got {tuple(latent.shape)}")
    values = latent[0, :, int(target.feature)]
    return int(values.argmax().cpu())


def _make_steering_intervention(
    nodes: list[FeatureNode],
    *,
    values_by_key: dict[str, float],
    target_timestep: float,
    mode: str,
    scale: float,
):
    by_layer: dict[int, list[FeatureNode]] = defaultdict(list)
    for node in nodes:
        by_layer[int(node.layer)].append(node)

    def intervene(_name: str, layer: int, latent: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        active = by_layer.get(int(layer))
        if not active:
            return latent
        flat_timestep = timestep.detach().float().reshape(-1)
        if flat_timestep.numel() == 0 or abs(float(flat_timestep[0].cpu()) - float(target_timestep)) > 1e-4:
            return latent
        modified = latent.clone()
        for node in active:
            if not (0 <= int(node.feature) < modified.shape[-1]):
                continue
            raw_value = float(values_by_key.get(node.key, 0.0)) * float(scale)
            value = torch.as_tensor(raw_value, dtype=modified.dtype, device=modified.device)
            if mode == "set-max":
                modified[..., node.feature] = torch.maximum(modified[..., node.feature], value)
            elif mode == "add":
                modified[..., node.feature] = modified[..., node.feature] + value
            elif mode == "set":
                modified[..., node.feature] = value
            else:
                raise ValueError(f"Unknown steer mode {mode!r}")
        return modified

    return intervene


def _chunk_metrics(action: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    delta = (action.float() - reference.float()).detach().cpu()
    per_position_l2 = delta.pow(2).sum(dim=-1).sqrt()
    return {
        "distance": float(delta.pow(2).sum().sqrt()),
        "rmse": float(delta.pow(2).mean().sqrt()),
        "max_position_l2": float(per_position_l2.max()),
        "mean_position_l2": float(per_position_l2.mean()),
        "max_position": int(per_position_l2.reshape(-1).argmax()),
    }


def _action_change_metrics(baseline: torch.Tensor, steered: torch.Tensor) -> dict[str, float]:
    delta = (steered.float() - baseline.float()).detach().cpu()
    per_position_l2 = delta.pow(2).sum(dim=-1).sqrt()
    return {
        "action_change_l2": float(delta.pow(2).sum().sqrt()),
        "action_change_rmse": float(delta.pow(2).mean().sqrt()),
        "action_change_max_position_l2": float(per_position_l2.max()),
        "action_change_max_position": int(per_position_l2.reshape(-1).argmax()),
    }


def _action_speed_metrics(action: torch.Tensor) -> dict[str, float]:
    xyz = action.float()[..., : min(3, action.shape[-1])].detach().cpu()
    per_position = xyz.pow(2).sum(dim=-1).sqrt()
    return {
        "speed_mean": float(per_position.mean()),
        "speed_sum": float(per_position.sum()),
        "speed_max": float(per_position.max()),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    preferred = [
        "intervention",
        "family",
        "observation_id",
        "episode_index",
        "frame_index",
        "task",
        "distance_before",
        "distance_after",
        "absolute_improvement",
        "relative_improvement",
        "speed_before",
        "speed_after",
        "speed_delta",
        "speed_gap_closed",
        "action_change_l2",
        "target_latent_before",
        "target_latent_after",
    ]
    fieldnames = [field for field in preferred if field in fields] + [field for field in fields if field not in preferred]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for intervention in sorted({row["intervention"] for row in rows}):
        subset = [row for row in rows if row["intervention"] == intervention]
        if not subset:
            continue
        out.append(
            {
                "intervention": intervention,
                "family": subset[0].get("family", intervention),
                "examples": len(subset),
                "mean_distance_before": sum(float(row["distance_before"]) for row in subset) / len(subset),
                "mean_distance_after": sum(float(row["distance_after"]) for row in subset) / len(subset),
                "mean_absolute_improvement": sum(float(row["absolute_improvement"]) for row in subset) / len(subset),
                "mean_relative_improvement": sum(float(row["relative_improvement"]) for row in subset) / len(subset),
                "mean_action_change_l2": sum(float(row["action_change_l2"]) for row in subset) / len(subset),
                "mean_speed_before": sum(float(row["speed_before"]) for row in subset) / len(subset),
                "mean_speed_after": sum(float(row["speed_after"]) for row in subset) / len(subset),
                "mean_speed_delta": sum(float(row["speed_delta"]) for row in subset) / len(subset),
                "mean_speed_relative_change": sum(float(row["speed_relative_change"]) for row in subset) / len(subset),
                "mean_speed_gap_closed": sum(float(row["speed_gap_closed"]) for row in subset) / len(subset),
                "speed_increase_rate": sum(float(row["speed_direction_success"]) for row in subset) / len(subset),
                "mean_target_latent_delta": sum(float(row["target_latent_after"]) - float(row["target_latent_before"]) for row in subset)
                / len(subset),
            }
        )
    return out


def _write_html(path: Path, *, config: dict[str, Any], summary: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary_fields = list(summary[0]) if summary else []
    example_fields = [
        "intervention",
        "family",
        "observation_id",
        "episode_index",
        "frame_index",
        "distance_before",
        "distance_after",
        "absolute_improvement",
        "relative_improvement",
        "speed_before",
        "speed_after",
        "speed_delta",
        "speed_gap_closed",
        "action_change_l2",
        "task",
    ]
    html_rows = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Pi0.5 Circuit Steering Validation</title>",
        "<style>body{font-family:Arial,Helvetica,sans-serif;margin:24px;color:#15171a}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid #e5e7eb;padding:8px;text-align:left;vertical-align:top}th{background:#f6f8fb}.metric{font-family:Menlo,Consolas,monospace}.muted{color:#667085}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px}.card{border:1px solid #d9dee7;padding:10px;border-radius:8px;background:#fafbfc}</style>",
        "</head><body><h1>Pi0.5 Circuit Steering Validation</h1>",
        "<p class='muted'>Steered matched-negative examples are scored by whole-action-chunk distance to the positive action prototype.</p>",
        "<section class='grid'>",
    ]
    for key in (
        "target",
        "negative_source",
        "behavior_event_bin",
        "steer_mode",
        "steer_scale",
        "num_negative_examples",
        "num_inference_steps",
    ):
        html_rows.append(f"<div class='card'>{html.escape(key)}<br><b class='metric'>{html.escape(str(config.get(key)))}</b></div>")
    html_rows.extend(["</section>", "<h2>Aggregate</h2>", "<table><thead><tr>"])
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
    html_rows.extend(f"<th>{html.escape(field)}</th>" for field in example_fields)
    html_rows.append("</tr></thead><tbody>")
    for row in rows:
        html_rows.append("<tr>")
        for field in example_fields:
            value = row.get(field, "")
            text = f"{value:.5g}" if isinstance(value, float) else str(value)
            html_rows.append(f"<td>{html.escape(text)}</td>")
        html_rows.append("</tr>")
    html_rows.extend(["</tbody></table>", "</body></html>"])
    path.write_text("\n".join(html_rows))


def _intervention_family(name: str) -> str:
    if name.startswith("random_feature_"):
        return "random_feature_control"
    if name.startswith("random_circuit_"):
        return "random_circuit_control"
    return name


def _reference_values_for_randoms(
    interventions: dict[str, list[FeatureNode]],
    values_by_key: dict[str, float],
) -> dict[str, float]:
    out = dict(values_by_key)
    real_nodes = interventions["circuit_steer"]
    by_layer: dict[int, list[float]] = defaultdict(list)
    for node in real_nodes:
        value = values_by_key.get(node.key)
        if value is not None:
            by_layer[int(node.layer)].append(float(value))
    global_mean = sum(values_by_key.values()) / max(1, len(values_by_key))
    for name, nodes in interventions.items():
        if not name.startswith("random_"):
            continue
        for node in nodes:
            layer_values = by_layer.get(int(node.layer))
            out[node.key] = sum(layer_values) / len(layer_values) if layer_values else global_mean
    return out


def main() -> None:
    args = parse_args()
    patch_transformers_causal_mask_compat()
    patch_pi05_checkpoint_key_compat()

    if args.graph_json is not None:
        graph = _read_graph(args.graph_json)
        target = _target_from_graph(graph, args.target)
    else:
        if not args.target:
            raise ValueError("Either --graph-json or --target is required")
        target = _target_from_graph({"config": {"target": args.target}, "nodes": []}, args.target)
        graph = {
            "config": {"target": target.key},
            "nodes": [{"node_key": target.key, "kind": "target"}],
            "edges": [],
        }
    ablation_nodes = _intervention_nodes(graph, target)
    target_nodes = ablation_nodes["target_ablate"]
    parent_nodes = ablation_nodes["parents_ablate"]
    circuit_nodes = ablation_nodes["circuit_ablate"]
    base_interventions = {
        "target_steer": target_nodes,
        "parents_steer": parent_nodes,
        "circuit_steer": circuit_nodes,
    }
    episodes = args.episodes or _feature_dir_episodes(args.feature_dir)

    print("Steering validation plan", flush=True)
    print(f"target={target.key}", flush=True)
    print(f"negative_source={args.negative_source}", flush=True)
    print(f"steer_mode={args.steer_mode} scale={args.steer_scale}", flush=True)
    print(f"episodes={episodes or 'all'}", flush=True)
    if args.plan_only:
        print("plan-only requested; exiting before model load", flush=True)
        return

    if args.positive_source == "behavior-event-fast":
        if not args.behavior_event_bin:
            raise ValueError("--behavior-event-bin is required for --positive-source behavior-event-fast")
        positive_examples = _behavior_event_examples(
            feature_dir=args.feature_dir,
            event_bin=args.behavior_event_bin,
            episodes=_episode_set(args.positive_episodes),
            limit=args.num_positive_prototype_examples,
            fastest=True,
        )
    elif args.positive_source == "behavior-phase-fast":
        if not args.behavior_phase:
            raise ValueError("--behavior-phase is required for --positive-source behavior-phase-fast")
        positive_examples = _behavior_phase_positive_examples(
            feature_dir=args.feature_dir,
            target=target,
            phase=args.behavior_phase,
            top_examples=args.top_examples,
            limit=args.num_positive_prototype_examples,
        )
    else:
        positive_examples = top_examples_for_target(feature_dir=args.feature_dir, target=target, top_examples=args.top_examples)
    if not positive_examples:
        raise RuntimeError(f"No top examples found for {target.key}")
    positive_examples = positive_examples[: args.num_positive_prototype_examples]

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
        d_features = int(next(iter(transcoders.values())).config.latent_dim)
        randomized = _add_random_control_interventions(
            {"baseline": [], "target_ablate": target_nodes, "parents_ablate": parent_nodes, "circuit_ablate": circuit_nodes},
            target=target,
            d_features=d_features,
            random_feature_controls=args.random_feature_controls,
            random_circuit_controls=args.random_circuit_controls,
            seed=args.random_control_seed,
        )
        for name, nodes in randomized.items():
            if name.startswith("random_feature_") or name.startswith("random_circuit_"):
                base_interventions[name.replace("ablate", "steer")] = nodes

    context = Pi05TranscoderContext(
        mode="replace",
        capture_records=False,
        capture_latents=False,
        capture_traces=True,
        store_latent_summaries=False,
    )
    _context, wrapped_names = install_pi05_action_expert_wrappers(policy, context=context, transcoders=transcoders, mode="replace")
    print(f"installed {len(wrapped_names)} replacement transcoders", flush=True)

    positive_actions: list[torch.Tensor] = []
    positive_speeds: list[float] = []
    positive_node_values: list[dict[str, float]] = []
    prototype_nodes = list(_nodes_by_key(circuit_nodes).values())
    for offset, example in enumerate(positive_examples, start=1):
        index = _dataset_index(dataset, example)
        if index is None:
            raise RuntimeError(f"Could not map positive observation {example.get('observation_id')} to dataset row")
        raw_batch = _collate_one(dataset, index)
        raw_batch = _prepare_raw_batch(raw_batch, dataset.meta.camera_keys)
        batch = preprocessor(raw_batch)
        batch_size = batch[OBS_LANGUAGE_TOKENS].shape[0]
        actions_shape = (batch_size, policy.model.config.chunk_size, policy.model.config.max_action_dim)
        torch.manual_seed(100_000 + offset)
        noise = policy.model.sample_noise(actions_shape, device)
        cache = TraceForwardCache()
        context.trace_callback = cache.add
        context.latent_intervention = None
        context.clear_records()
        with torch.no_grad():
            action = _run_action_chunk_with_fixed_noise(
                policy=policy,
                batch=batch,
                noise=noise,
                num_inference_steps=args.num_inference_steps,
            )
        positive_actions.append(action.detach().float().cpu())
        positive_speeds.append(_action_speed_metrics(action)["speed_mean"])
        positive_node_values.append(_collect_node_values(cache, prototype_nodes))
        if not args.no_progress:
            print(f"positive prototype {offset}/{len(positive_examples)}", flush=True)

    positive_reference = torch.cat(positive_actions, dim=0).mean(dim=0, keepdim=True).to(device)
    positive_speed_reference = sum(positive_speeds) / len(positive_speeds)
    real_values = _mean_dict(positive_node_values)
    steering_values = _reference_values_for_randoms(base_interventions, real_values)
    values_path = args.output_dir / "steering_values.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    values_path.write_text(json.dumps(steering_values, indent=2, sort_keys=True))
    print(f"positive prototype from {len(positive_actions)} examples; wrote {values_path}", flush=True)

    scan_batch_size = cfg.batch_size
    cfg.batch_size = args.scan_batch_size
    cfg.num_workers = args.num_workers
    dataloader = _make_dataloader(cfg, dataset)
    exclude_tasks = {str(example.get("task", "")) for example in positive_examples if example.get("task")}
    args.num_examples = args.num_negative_examples
    if args.negative_source == "behavior-event-slow":
        if not args.behavior_event_bin:
            raise ValueError("--behavior-event-bin is required for --negative-source behavior-event-slow")
        negatives = _behavior_event_examples(
            feature_dir=args.feature_dir,
            event_bin=args.behavior_event_bin,
            episodes=_episode_set(args.negative_episodes),
            limit=args.num_negative_examples,
            fastest=False,
            balance_episodes=True,
        )
    elif args.negative_source == "behavior-phase-slow":
        if not args.behavior_phase:
            raise ValueError("--behavior-phase is required for --negative-source behavior-phase-slow")
        negatives = _behavior_phase_slow_examples(
            feature_dir=args.feature_dir,
            phase=args.behavior_phase,
            limit=args.num_negative_examples,
        )
    elif args.negative_source == "low-target":
        negatives = _scan_low_target_examples(
            args=args,
            policy=policy,
            dataset=dataset,
            dataloader=dataloader,
            preprocessor=preprocessor,
            context=context,
            target=target,
            exclude_tasks=exclude_tasks,
        )
    else:
        negatives = _scan_hard_task_examples(
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
    if not negatives:
        raise RuntimeError(f"No negative examples selected for source={args.negative_source}")

    selected_path = args.output_dir / "selected_negative_examples.jsonl"
    with selected_path.open("w") as f:
        for example in negatives:
            f.write(json.dumps(example, sort_keys=True) + "\n")
    print(f"selected {len(negatives)} negatives wrote {selected_path}", flush=True)

    rows: list[dict[str, Any]] = []
    started = time.time()
    for offset, example in enumerate(negatives, start=1):
        index = _dataset_index(dataset, example)
        if index is None:
            raise RuntimeError(f"Could not map negative observation {example.get('observation_id')} to dataset row")
        raw_batch = _collate_one(dataset, index)
        raw_batch = _prepare_raw_batch(raw_batch, dataset.meta.camera_keys)
        batch = preprocessor(raw_batch)
        batch_size = batch[OBS_LANGUAGE_TOKENS].shape[0]
        actions_shape = (batch_size, policy.model.config.chunk_size, policy.model.config.max_action_dim)
        torch.manual_seed(200_000 + offset)
        noise = policy.model.sample_noise(actions_shape, device)
        baseline_cache = TraceForwardCache()
        context.trace_callback = baseline_cache.add
        context.latent_intervention = None
        context.clear_records()
        with torch.no_grad():
            baseline_action = _run_action_chunk_with_fixed_noise(
                policy=policy,
                batch=batch,
                noise=noise,
                num_inference_steps=args.num_inference_steps,
            )
        before_metrics = _chunk_metrics(baseline_action, positive_reference)
        speed_before = _action_speed_metrics(baseline_action)
        if str(example.get("target_position", "")).lower() == "infer":
            target_position = _infer_target_position(baseline_cache, target)
        else:
            target_position = int(example["target_position"])
        target_before, pre_before = _target_activation(baseline_cache, target, target_position)

        for intervention, nodes in base_interventions.items():
            cache = TraceForwardCache()
            context.trace_callback = cache.add
            context.latent_intervention = _make_steering_intervention(
                nodes,
                values_by_key=steering_values,
                target_timestep=target.timestep,
                mode=args.steer_mode,
                scale=args.steer_scale,
            )
            context.clear_records()
            with torch.no_grad():
                steered_action = _run_action_chunk_with_fixed_noise(
                    policy=policy,
                    batch=batch,
                    noise=noise,
                    num_inference_steps=args.num_inference_steps,
                )
            after_metrics = _chunk_metrics(steered_action, positive_reference)
            speed_after = _action_speed_metrics(steered_action)
            target_after, pre_after = _target_activation(cache, target, target_position)
            distance_before = before_metrics["distance"]
            distance_after = after_metrics["distance"]
            absolute_improvement = distance_before - distance_after
            relative_improvement = absolute_improvement / distance_before if distance_before > 1e-8 else 0.0
            speed_delta = speed_after["speed_mean"] - speed_before["speed_mean"]
            speed_relative_change = speed_delta / speed_before["speed_mean"] if speed_before["speed_mean"] > 1e-8 else 0.0
            speed_gap_before = positive_speed_reference - speed_before["speed_mean"]
            speed_gap_after = positive_speed_reference - speed_after["speed_mean"]
            speed_gap_closed = (
                (abs(speed_gap_before) - abs(speed_gap_after)) / abs(speed_gap_before)
                if abs(speed_gap_before) > 1e-8
                else 0.0
            )
            rows.append(
                {
                    "intervention": intervention,
                    "family": _intervention_family(intervention),
                    "observation_id": int(example["observation_id"]),
                    "episode_index": example.get("episode_index"),
                    "frame_index": example.get("frame_index"),
                    "timestamp": example.get("timestamp"),
                    "task": example.get("task", ""),
                    "manual_event_bin": example.get("manual_event_bin"),
                    "target_position": target_position,
                    "target_feature_score": example.get("target_feature_score", example.get("score")),
                    "distance_before": distance_before,
                    "distance_after": distance_after,
                    "absolute_improvement": absolute_improvement,
                    "relative_improvement": relative_improvement,
                    "positive_speed_reference": positive_speed_reference,
                    "speed_before": speed_before["speed_mean"],
                    "speed_after": speed_after["speed_mean"],
                    "speed_delta": speed_delta,
                    "speed_relative_change": speed_relative_change,
                    "speed_gap_before": speed_gap_before,
                    "speed_gap_after": speed_gap_after,
                    "speed_gap_closed": speed_gap_closed,
                    "speed_direction_success": float(speed_delta > 0.0),
                    "speed_max_before": speed_before["speed_max"],
                    "speed_max_after": speed_after["speed_max"],
                    "distance_rmse_before": before_metrics["rmse"],
                    "distance_rmse_after": after_metrics["rmse"],
                    "distance_max_position_before": before_metrics["max_position"],
                    "distance_max_position_after": after_metrics["max_position"],
                    "target_latent_before": target_before,
                    "target_latent_after": target_after,
                    "target_latent_delta": target_after - target_before,
                    "target_preactivation_before": pre_before,
                    "target_preactivation_after": pre_after,
                    "target_preactivation_delta": pre_after - pre_before,
                    "steered_nodes": len(nodes),
                    **_action_change_metrics(baseline_action, steered_action),
                }
            )
        print(f"negative {offset}/{len(negatives)} elapsed_s={time.time() - started:.1f}", flush=True)

    summary = _aggregate(rows)
    config = {
        "target": target.key,
        "graph_json": str(args.graph_json),
        "feature_dir": str(args.feature_dir),
        "checkpoint": str(args.checkpoint),
        "positive_source": args.positive_source,
        "negative_source": args.negative_source,
        "behavior_phase": args.behavior_phase,
        "behavior_event_bin": args.behavior_event_bin,
        "positive_episodes": sorted(_episode_set(args.positive_episodes) or []),
        "negative_episodes": sorted(_episode_set(args.negative_episodes) or []),
        "hard_task_queries": args.hard_task_queries,
        "num_positive_prototype_examples": len(positive_examples),
        "num_negative_examples": len(negatives),
        "positive_speed_reference": positive_speed_reference,
        "num_inference_steps": args.num_inference_steps,
        "steer_mode": args.steer_mode,
        "steer_scale": args.steer_scale,
        "random_feature_controls": args.random_feature_controls,
        "random_circuit_controls": args.random_circuit_controls,
        "random_control_seed": args.random_control_seed,
    }
    _write_csv(args.output_dir / "steering_rows.csv", rows)
    _write_csv(args.output_dir / "steering_summary.csv", summary)
    (args.output_dir / "steering_results.json").write_text(json.dumps({"config": config, "summary": summary, "rows": rows}, indent=2))
    _write_html(args.output_dir / "steering_report.html", config=config, summary=summary, rows=rows)
    print(f"saved steering validation outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

"""DifFRACT-style sparse-feature circuit tracing utilities."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class FeatureNode:
    """A sparse transcoder feature at one action-expert layer and flow time."""

    layer: int
    timestep: float
    feature: int

    @property
    def key(self) -> str:
        return feature_key(self.layer, self.timestep, self.feature)


@dataclass(frozen=True)
class TraceRecord:
    """Gradient-preserving tensors from one layer/timestep in one forward."""

    name: str
    layer: int
    timestep: float
    preactivation: Tensor
    latent: Tensor


@dataclass(frozen=True)
class ParentFeatureSummary:
    """Aggregated parent score for one source feature."""

    source: FeatureNode
    target: FeatureNode
    mean_abs_contribution: float
    mean_signed_contribution: float
    std_signed_contribution: float
    frequency_in_example_topk: float
    edge_score: float
    examples: int


def timestep_key(timestep: float) -> str:
    return f"{float(timestep):.8f}"


def feature_key(layer: int, timestep: float, feature: int) -> str:
    return f"L{int(layer):02d}:tau{float(timestep):.4g}:F{int(feature)}"


def parse_feature_key(raw: str) -> FeatureNode:
    """Parse keys like ``L12:tau1:F7584``."""
    parts = raw.split(":")
    if len(parts) != 3:
        raise ValueError(f"Expected feature key L<layer>:tau<timestep>:F<feature>, got {raw!r}")
    layer_part, timestep_part, feature_part = parts
    if not layer_part.startswith("L") or not timestep_part.startswith("tau") or not feature_part.startswith("F"):
        raise ValueError(f"Expected feature key L<layer>:tau<timestep>:F<feature>, got {raw!r}")
    return FeatureNode(
        layer=int(layer_part[1:]),
        timestep=float(timestep_part[3:]),
        feature=int(feature_part[1:]),
    )


def node_key(node: FeatureNode) -> str:
    return node.key


class TraceForwardCache:
    """Records transcoder tensors for one traced Pi0.5 forward."""

    def __init__(self):
        self.records: dict[tuple[int, str], TraceRecord] = {}
        self.layer_names: dict[int, str] = {}

    def add(self, name: str, layer: int, preactivation: Tensor, latent: Tensor, timestep: Tensor) -> None:
        flat_timestep = timestep.detach().float().reshape(-1)
        if flat_timestep.numel() == 0:
            raise ValueError("Empty timestep tensor")
        value = float(flat_timestep[0].cpu())
        key = (int(layer), timestep_key(value))
        self.layer_names[int(layer)] = name
        self.records[key] = TraceRecord(
            name=name,
            layer=int(layer),
            timestep=value,
            preactivation=preactivation,
            latent=latent,
        )

    def get(self, layer: int, timestep: float) -> TraceRecord:
        key = (int(layer), timestep_key(timestep))
        if key in self.records:
            return self.records[key]

        candidates = [record for (record_layer, _), record in self.records.items() if record_layer == int(layer)]
        if not candidates:
            raise KeyError(f"No trace record for layer {layer}")
        closest = min(candidates, key=lambda record: abs(record.timestep - float(timestep)))
        if abs(closest.timestep - float(timestep)) > 1e-4:
            raise KeyError(
                f"No trace record for layer {layer} timestep {timestep:.8f}; "
                f"closest available timestep is {closest.timestep:.8f}"
            )
        return closest


def _require_rank3(name: str, tensor: Tensor) -> None:
    if tensor.ndim != 3:
        raise ValueError(f"Expected {name} shaped [batch, positions, features], got {tuple(tensor.shape)}")


def source_contributions_to_target(
    *,
    cache: TraceForwardCache,
    source_layer: int,
    target: FeatureNode,
    target_position: int,
    top_m_per_example: int,
) -> dict[str, Tensor]:
    """Return collapsed source-feature contributions for one traced example.

    The contribution is the local replacement-model attribution

    ``source_latent * d(target_preactivation) / d(source_latent)``.

    Position is collapsed by summing signed contributions per source feature.
    The max-absolute source position is retained only as inspection metadata.
    """
    target_record = cache.get(target.layer, target.timestep)
    source_record = cache.get(source_layer, target.timestep)
    _require_rank3("target preactivation", target_record.preactivation)
    _require_rank3("source latent", source_record.latent)
    if target_record.preactivation.shape[0] != 1 or source_record.latent.shape[0] != 1:
        raise ValueError("Circuit tracing currently expects batch_size=1 so target positions stay unambiguous")
    if target_position < 0 or target_position >= target_record.preactivation.shape[1]:
        raise ValueError(
            f"Target position {target_position} out of range for target shape {tuple(target_record.preactivation.shape)}"
        )
    if target.feature < 0 or target.feature >= target_record.preactivation.shape[-1]:
        raise ValueError(
            f"Target feature {target.feature} out of range for target shape {tuple(target_record.preactivation.shape)}"
        )

    target_scalar = target_record.preactivation[0, target_position, target.feature]
    grad = torch.autograd.grad(
        target_scalar,
        source_record.latent,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if grad is None:
        zeros = source_record.latent.detach().new_zeros(source_record.latent.shape[1:])
        contribution = zeros
    else:
        contribution = (source_record.latent * grad)[0]

    signed_contribution = contribution.sum(dim=0)
    collapsed_abs = signed_contribution.abs()
    source_positions = contribution.abs().argmax(dim=0)

    per_example_top = torch.zeros_like(collapsed_abs, dtype=torch.bool)
    if top_m_per_example > 0:
        k = min(int(top_m_per_example), collapsed_abs.numel())
        per_example_top[torch.topk(collapsed_abs, k=k).indices] = True

    return {
        "collapsed_abs": collapsed_abs.detach().cpu().float(),
        "signed_contribution": signed_contribution.detach().cpu().float(),
        "source_positions": source_positions.detach().cpu().to(dtype=torch.int16),
        "per_example_top": per_example_top.detach().cpu(),
    }


def source_contributions_to_layers(
    *,
    cache: TraceForwardCache,
    source_layers: list[int] | range,
    target: FeatureNode,
    target_position: int,
) -> dict[int, dict[str, Tensor]]:
    """Return source-feature contributions from several layers to one target node.

    This is the DifFRACT-frontier primitive: one target scalar VJP gives
    gradients with respect to every selected source layer latent tensor, then
    each source feature attribution is ``sum_p z[p, j] * grad[p, j]``.
    """
    target_record = cache.get(target.layer, target.timestep)
    _require_rank3("target preactivation", target_record.preactivation)
    if target_record.preactivation.shape[0] != 1:
        raise ValueError("Circuit tracing currently expects batch_size=1 so target positions stay unambiguous")
    if target_position < 0 or target_position >= target_record.preactivation.shape[1]:
        raise ValueError(
            f"Target position {target_position} out of range for target shape {tuple(target_record.preactivation.shape)}"
        )
    if target.feature < 0 or target.feature >= target_record.preactivation.shape[-1]:
        raise ValueError(
            f"Target feature {target.feature} out of range for target shape {tuple(target_record.preactivation.shape)}"
        )

    source_records: list[TraceRecord] = []
    for source_layer in source_layers:
        if int(source_layer) >= target.layer:
            continue
        try:
            source_record = cache.get(int(source_layer), target.timestep)
        except KeyError:
            continue
        _require_rank3("source latent", source_record.latent)
        if source_record.latent.shape[0] != 1:
            raise ValueError("Circuit tracing currently expects batch_size=1")
        source_records.append(source_record)

    if not source_records:
        return {}

    target_scalar = target_record.preactivation[0, target_position, target.feature]
    grads = torch.autograd.grad(
        target_scalar,
        [record.latent for record in source_records],
        retain_graph=True,
        allow_unused=True,
    )

    out: dict[int, dict[str, Tensor]] = {}
    for source_record, grad in zip(source_records, grads, strict=False):
        if grad is None:
            contribution = source_record.latent.detach().new_zeros(source_record.latent.shape[1:])
        else:
            contribution = (source_record.latent * grad)[0]
        signed_contribution = contribution.sum(dim=0)
        out[source_record.layer] = {
            "signed_contribution": signed_contribution.detach().cpu().float(),
            "collapsed_abs": signed_contribution.detach().abs().cpu().float(),
            "source_positions": contribution.detach().abs().argmax(dim=0).cpu().to(dtype=torch.int16),
        }
    return out


def aggregate_parent_contributions(
    *,
    source_layer: int,
    target: FeatureNode,
    collapsed_abs_values: list[Tensor],
    signed_values: list[Tensor],
    source_positions: list[Tensor],
    per_example_top: list[Tensor],
    parents_per_node: int,
    edge_score_metric: str = "mean_abs",
) -> tuple[list[ParentFeatureSummary], dict[int, list[int]]]:
    """Aggregate per-example source contributions and return top parent nodes."""
    if not collapsed_abs_values:
        return [], {}
    collapsed_abs = torch.stack(collapsed_abs_values, dim=0).float()
    signed = torch.stack(signed_values, dim=0).float()
    positions = torch.stack(source_positions, dim=0).to(dtype=torch.int64)
    top_hits = torch.stack(per_example_top, dim=0).bool()

    mean_abs = collapsed_abs.mean(dim=0)
    mean_signed = signed.mean(dim=0)
    std_signed = signed.std(dim=0, unbiased=False)
    frequency = top_hits.float().mean(dim=0)

    if edge_score_metric == "mean_abs":
        edge_scores = mean_abs
    elif edge_score_metric == "mean_abs_frequency":
        edge_scores = mean_abs * frequency.clamp_min(1.0 / max(1, collapsed_abs.shape[0]))
    else:
        raise ValueError(f"Unknown edge_score_metric {edge_score_metric!r}")

    k = min(int(parents_per_node), edge_scores.numel())
    top_scores, top_indices = torch.topk(edge_scores, k=k)
    summaries: list[ParentFeatureSummary] = []
    positions_by_feature: dict[int, list[int]] = {}
    for score, feature_index in zip(top_scores.tolist(), top_indices.tolist(), strict=False):
        feature = int(feature_index)
        positions_by_feature[feature] = [int(value) for value in positions[:, feature].tolist()]
        summaries.append(
            ParentFeatureSummary(
                source=FeatureNode(layer=source_layer, timestep=target.timestep, feature=feature),
                target=target,
                mean_abs_contribution=float(mean_abs[feature]),
                mean_signed_contribution=float(mean_signed[feature]),
                std_signed_contribution=float(std_signed[feature]),
                frequency_in_example_topk=float(frequency[feature]),
                edge_score=float(score),
                examples=int(collapsed_abs.shape[0]),
            )
        )
    return summaries, positions_by_feature


def build_node_index(nodes: dict[str, dict[str, Any]], edges: list[dict[str, Any]]) -> tuple[list[str], dict[str, int]]:
    """Assign matrix indices to nodes referenced by ``nodes`` or ``edges``."""
    node_list: list[str] = []
    node_to_idx: dict[str, int] = {}

    def add(key: str) -> None:
        if key not in node_to_idx:
            node_to_idx[key] = len(node_list)
            node_list.append(key)

    for key in nodes:
        add(key)
    for edge in edges:
        add(str(edge["source_key"]))
        add(str(edge["target_key"]))
    return node_list, node_to_idx


def attribution_adjacency(
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    edge_value_key: str = "edge_mass",
) -> tuple[np.ndarray, list[str], dict[str, int]]:
    """Return ``A[i, j] = attribution mass from source node i to target node j``."""
    node_list, node_to_idx = build_node_index(nodes, edges)
    adjacency = np.zeros((len(node_list), len(node_list)), dtype=np.float64)
    for edge in edges:
        source = node_to_idx.get(str(edge["source_key"]))
        target = node_to_idx.get(str(edge["target_key"]))
        if source is None or target is None:
            continue
        value = edge.get(edge_value_key, edge.get("edge_score", edge.get("attribution", 0.0)))
        adjacency[source, target] += abs(float(value))
    return adjacency, node_list, node_to_idx


def normalized_adjacency(adjacency: np.ndarray) -> np.ndarray:
    """Column-normalize absolute incoming attribution mass."""
    adjacency_abs = np.abs(adjacency)
    col_sums = np.maximum(adjacency_abs.sum(axis=0), 1e-8)
    return adjacency_abs / col_sums[None, :]


def indirect_influence(adjacency_norm: np.ndarray) -> np.ndarray:
    """Return total indirect influence ``(I - A)^-1 - I``."""
    n_nodes = adjacency_norm.shape[0]
    identity = np.eye(n_nodes, dtype=np.float64)
    try:
        return np.linalg.inv(identity - adjacency_norm) - identity
    except np.linalg.LinAlgError:
        return np.linalg.pinv(identity - adjacency_norm) - identity


def node_influence_scores(
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    target_key: str,
) -> dict[str, float]:
    """Compute DifFRACT-style indirect node influence on the root target."""
    adjacency, node_list, node_to_idx = attribution_adjacency(nodes, edges)
    if not node_list:
        return {}
    target_idx = node_to_idx.get(target_key)
    if target_idx is None:
        return {key: 0.0 for key in node_list}
    influence = indirect_influence(normalized_adjacency(adjacency))
    weights = np.zeros(len(node_list), dtype=np.float64)
    weights[target_idx] = 1.0
    scores = influence @ weights
    scores[target_idx] = 1.0
    return {key: float(max(0.0, scores[idx])) for key, idx in node_to_idx.items()}


def prune_nodes_by_cumulative_influence(
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    target_key: str,
    threshold: float,
) -> set[str]:
    """Keep the target plus highest-influence nodes up to cumulative threshold."""
    scores = node_influence_scores(nodes, edges, target_key=target_key)
    keep = {target_key}
    candidates = [(key, score) for key, score in scores.items() if key != target_key and score > 0.0]
    candidates.sort(key=lambda item: item[1], reverse=True)
    total = sum(score for _key, score in candidates)
    if total <= 1e-12:
        return keep | {key for key in nodes if key == target_key}
    cumulative = 0.0
    for key, score in candidates:
        keep.add(key)
        cumulative += score
        if cumulative / total >= threshold:
            break
    return keep


def prune_edges_by_cumulative_influence(
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    target_key: str,
    threshold: float,
) -> list[dict[str, Any]]:
    """Keep highest DifFRACT edge-influence edges up to cumulative threshold."""
    if not edges:
        return []
    adjacency, _node_list, node_to_idx = attribution_adjacency(nodes, edges)
    adjacency_norm = normalized_adjacency(adjacency)
    node_scores = node_influence_scores(nodes, edges, target_key=target_key)

    scored_edges: list[tuple[float, dict[str, Any]]] = []
    for edge in edges:
        source_key = str(edge["source_key"])
        target_key_edge = str(edge["target_key"])
        source_idx = node_to_idx.get(source_key)
        target_idx = node_to_idx.get(target_key_edge)
        if source_idx is None or target_idx is None:
            continue
        score = float(adjacency_norm[source_idx, target_idx] * node_scores.get(target_key_edge, 0.0))
        edge = dict(edge)
        edge["prune_edge_score"] = score
        scored_edges.append((score, edge))

    scored_edges.sort(key=lambda item: item[0], reverse=True)
    total = sum(score for score, _edge in scored_edges)
    if total <= 1e-12:
        return [edge for _score, edge in scored_edges]

    kept: list[dict[str, Any]] = []
    cumulative = 0.0
    for score, edge in scored_edges:
        kept.append(edge)
        cumulative += score
        if cumulative / total >= threshold:
            break
    return kept


def load_observations(path: Path) -> dict[int, dict[str, Any]]:
    observations: dict[int, dict[str, Any]] = {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            observations[int(row["observation_id"])] = row
    return observations


def layer_name_for_index(topk_payload: dict[str, Any], layer: int) -> str:
    for name in topk_payload["layer_names"]:
        if int(topk_payload["layer_indices"][name]) == int(layer):
            return name
    raise KeyError(f"No layer {layer} in feature_topk.pt")


def closest_timestep_key(stores: dict[str, Any], timestep: float, *, tolerance: float = 1e-4) -> str:
    key = timestep_key(timestep)
    if key in stores:
        return key
    if not stores:
        raise KeyError("No timesteps are available")
    closest = min(stores, key=lambda item: abs(float(item) - float(timestep)))
    if abs(float(closest) - float(timestep)) > tolerance:
        raise KeyError(
            f"No timestep {float(timestep):.8f}; closest available timestep is {float(closest):.8f}"
        )
    return closest


def top_examples_for_target(
    *,
    feature_dir: Path,
    target: FeatureNode,
    top_examples: int,
) -> list[dict[str, Any]]:
    """Load Top-K observation ids and target positions for a discovered feature."""
    topk_payload = torch.load(feature_dir / "feature_topk.pt", map_location="cpu", weights_only=False)
    observations = load_observations(feature_dir / "observations.jsonl")
    name = layer_name_for_index(topk_payload, target.layer)
    t_key = closest_timestep_key(topk_payload["topk"][name], target.timestep)
    store = topk_payload["topk"][name][t_key]
    limit = min(int(top_examples), store["scores"].shape[1])
    examples: list[dict[str, Any]] = []
    for rank in range(limit):
        score = float(store["scores"][target.feature, rank])
        if not torch.isfinite(torch.tensor(score)):
            continue
        observation_id = int(store["observation_ids"][target.feature, rank])
        if observation_id < 0:
            continue
        row = dict(observations.get(observation_id, {"observation_id": observation_id}))
        row.update(
            {
                "rank": rank + 1,
                "score": score,
                "target_position": int(store["action_positions"][target.feature, rank]),
                "flow_timestep": float(store["flow_timesteps"][target.feature, rank]),
            }
        )
        examples.append(row)
    return examples


def write_trace_outputs(
    *,
    output_dir: Path,
    config: dict[str, Any],
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "trace_config.json").open("w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")
    with (output_dir / "graph.json").open("w") as f:
        json.dump(
            {
                "format_version": 1,
                "description": "Pi0.5 DifFRACT-style local replacement model circuit trace.",
                "config": config,
                "nodes": list(nodes.values()),
                "edges": edges,
            },
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")

    node_fields = [
        "node_key",
        "layer",
        "timestep",
        "feature",
        "depth",
        "kind",
        "label",
        "expanded",
        "influence",
    ]
    with (output_dir / "nodes_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=node_fields)
        writer.writeheader()
        for row in nodes.values():
            writer.writerow({field: row.get(field) for field in node_fields})

    edge_fields = [
        "source_key",
        "target_key",
        "source_layer",
        "target_layer",
        "timestep",
        "source_feature",
        "target_feature",
        "depth",
        "rank",
        "edge_score",
        "edge_mass",
        "attribution",
        "prune_edge_score",
        "mean_abs_contribution",
        "mean_signed_contribution",
        "std_signed_contribution",
        "frequency_in_example_topk",
        "examples",
    ]
    with (output_dir / "edges_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=edge_fields)
        writer.writeheader()
        for row in edges:
            writer.writerow({field: row.get(field) for field in edge_fields})


def parent_summary_to_edge(summary: ParentFeatureSummary, *, depth: int, rank: int) -> dict[str, Any]:
    return {
        "source_key": summary.source.key,
        "target_key": summary.target.key,
        "source_layer": summary.source.layer,
        "target_layer": summary.target.layer,
        "timestep": summary.target.timestep,
        "source_feature": summary.source.feature,
        "target_feature": summary.target.feature,
        "depth": depth,
        "rank": rank,
        **asdict(summary),
        "source": asdict(summary.source),
        "target": asdict(summary.target),
    }

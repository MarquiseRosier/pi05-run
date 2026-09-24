#!/usr/bin/env python
"""Smoke test sparse-feature circuit tracing math on synthetic tensors."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from pi05_mi.circuit_tracing import (
    FeatureNode,
    TraceForwardCache,
    aggregate_parent_contributions,
    node_influence_scores,
    parent_summary_to_edge,
    prune_edges_by_cumulative_influence,
    prune_nodes_by_cumulative_influence,
    source_contributions_to_layers,
    source_contributions_to_target,
    write_trace_outputs,
)


def _make_cache(source_value_a: float, source_value_b: float) -> TraceForwardCache:
    source_latent = torch.zeros(1, 4, 8, requires_grad=True)
    source_latent.data[0, 1, 3] = source_value_a
    source_latent.data[0, 2, 6] = source_value_b

    target_scalar = 2.0 * source_latent[0, 1, 3] - 1.5 * source_latent[0, 2, 6]
    target_mask = torch.zeros(1, 3, 8)
    target_mask[0, 0, 5] = 1.0
    target_preactivation = target_mask * target_scalar

    cache = TraceForwardCache()
    cache.add(
        "paligemma_with_expert.gemma_expert.model.layers.0.mlp",
        0,
        torch.zeros_like(source_latent),
        source_latent,
        torch.tensor([1.0]),
    )
    cache.add(
        "paligemma_with_expert.gemma_expert.model.layers.1.mlp",
        1,
        target_preactivation,
        torch.relu(target_preactivation),
        torch.tensor([1.0]),
    )
    return cache


def _make_multilayer_cache() -> TraceForwardCache:
    source0 = torch.zeros(1, 3, 8, requires_grad=True)
    source1 = torch.zeros(1, 3, 8, requires_grad=True)
    source0.data[0, 0, 2] = 4.0
    source0.data[0, 2, 4] = 2.0
    source1.data[0, 1, 6] = 3.0

    target_scalar = 0.5 * source0[0, 0, 2] + 2.0 * source1[0, 1, 6] - source0[0, 2, 4]
    target_mask = torch.zeros(1, 2, 8)
    target_mask[0, 1, 7] = 1.0
    target_preactivation = target_mask * target_scalar

    cache = TraceForwardCache()
    cache.add("layer0", 0, torch.zeros_like(source0), source0, torch.tensor([0.7]))
    cache.add("layer1", 1, torch.zeros_like(source1), source1, torch.tensor([0.7]))
    cache.add("layer2", 2, target_preactivation, torch.relu(target_preactivation), torch.tensor([0.7]))
    return cache


def main() -> None:
    target = FeatureNode(layer=1, timestep=1.0, feature=5)
    per_example = [
        source_contributions_to_target(
            cache=_make_cache(3.0, 1.0),
            source_layer=0,
            target=target,
            target_position=0,
            top_m_per_example=2,
        ),
        source_contributions_to_target(
            cache=_make_cache(2.0, 4.0),
            source_layer=0,
            target=target,
            target_position=0,
            top_m_per_example=2,
        ),
    ]
    parents, _positions = aggregate_parent_contributions(
        source_layer=0,
        target=target,
        collapsed_abs_values=[row["collapsed_abs"] for row in per_example],
        signed_values=[row["signed_contribution"] for row in per_example],
        source_positions=[row["source_positions"] for row in per_example],
        per_example_top=[row["per_example_top"] for row in per_example],
        parents_per_node=2,
    )
    assert [parent.source.feature for parent in parents] == [3, 6]
    assert parents[0].source.layer == 0
    assert parents[0].target == target

    multilayer_target = FeatureNode(layer=2, timestep=0.7, feature=7)
    layer_contrib = source_contributions_to_layers(
        cache=_make_multilayer_cache(),
        source_layers=range(0, 2),
        target=multilayer_target,
        target_position=1,
    )
    assert set(layer_contrib) == {0, 1}
    assert torch.isclose(layer_contrib[0]["signed_contribution"][2], torch.tensor(2.0))
    assert torch.isclose(layer_contrib[0]["signed_contribution"][4], torch.tensor(-2.0))
    assert torch.isclose(layer_contrib[1]["signed_contribution"][6], torch.tensor(6.0))

    root = Path(tempfile.mkdtemp(prefix="pi05_circuit_trace_"))
    nodes = {
        target.key: {
            "node_key": target.key,
            "layer": target.layer,
            "timestep": target.timestep,
            "feature": target.feature,
            "depth": 0,
            "kind": "target",
            "label": "",
        }
    }
    edges = []
    for rank, parent in enumerate(parents, start=1):
        nodes[parent.source.key] = {
            "node_key": parent.source.key,
            "layer": parent.source.layer,
            "timestep": parent.source.timestep,
            "feature": parent.source.feature,
            "depth": 1,
            "kind": "parent",
            "label": "",
        }
        edges.append(parent_summary_to_edge(parent, depth=1, rank=rank))
    write_trace_outputs(
        output_dir=root,
        config={"target": target.key, "parents_per_node": 2, "max_depth": 1},
        nodes=nodes,
        edges=edges,
    )
    assert (root / "graph.json").exists()
    assert (root / "edges_summary.csv").exists()

    graph_nodes = {
        "T": {"node_key": "T", "layer": 2, "timestep": 1.0, "feature": 0},
        "A": {"node_key": "A", "layer": 1, "timestep": 1.0, "feature": 1},
        "B": {"node_key": "B", "layer": 1, "timestep": 1.0, "feature": 2},
    }
    graph_edges = [
        {"source_key": "A", "target_key": "T", "edge_mass": 0.9},
        {"source_key": "B", "target_key": "T", "edge_mass": 0.1},
    ]
    influence = node_influence_scores(graph_nodes, graph_edges, target_key="T")
    assert influence["T"] == 1.0
    assert influence["A"] > influence["B"]
    kept_nodes = prune_nodes_by_cumulative_influence(graph_nodes, graph_edges, target_key="T", threshold=0.8)
    assert kept_nodes == {"T", "A"}
    kept_edges = prune_edges_by_cumulative_influence(graph_nodes, graph_edges, target_key="T", threshold=0.8)
    assert len(kept_edges) == 1
    assert kept_edges[0]["source_key"] == "A"
    print(f"circuit tracing smoke test passed: {root}")


if __name__ == "__main__":
    main()

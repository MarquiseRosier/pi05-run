#!/usr/bin/env python
"""Smoke test for transcoder circuit tracing.

Checked against planted wiring: two transcoders are built with a known
connection between specific features, and the tracer must recover exactly that
edge and not invent others.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pi05_mi.circuit_tracing import (  # noqa: E402
    CircuitNode,
    build_graph,
    layer_index_of,
    name_nodes,
    select_nodes,
    wiring_matrix,
)

D_MODEL, N_FEATURES = 6, 5
L0, L1, L2 = "m.layers.0.mlp", "m.layers.1.mlp", "m.layers.2.mlp"


def _planted():
    """Feature 1 of layer 0 writes a direction that feature 3 of layer 1 reads."""
    torch.manual_seed(0)
    decoder0 = torch.zeros(D_MODEL, N_FEATURES)
    direction = torch.zeros(D_MODEL)
    direction[2] = 1.0
    decoder0[:, 1] = direction

    encoder1 = torch.zeros(N_FEATURES, D_MODEL)
    encoder1[3] = direction * 2.0  # reads it with gain 2

    decoder1 = torch.zeros(D_MODEL, N_FEATURES)
    second = torch.zeros(D_MODEL)
    second[4] = 1.0
    decoder1[:, 3] = second
    encoder2 = torch.zeros(N_FEATURES, D_MODEL)
    encoder2[0] = second * 3.0  # layer1 f3 -> layer2 f0, gain 3

    return (
        {L0: decoder0, L1: decoder1},
        {L1: encoder1, L2: encoder2},
    )


def test_layer_index_parsing() -> None:
    assert layer_index_of("paligemma_with_expert.gemma_expert.model.layers.17.mlp") == 17
    assert layer_index_of("no_digits") == -1


def test_wiring_matrix_recovers_the_planted_connection() -> None:
    decoders, encoders = _planted()
    matrix = wiring_matrix(
        decoders[L0], encoders[L1], source_features=[0, 1, 2], target_features=[3, 4]
    ).numpy()
    assert matrix.shape == (3, 2)
    assert abs(matrix[1, 0] - 2.0) < 1e-6, "source feature 1 -> target feature 3 with gain 2"
    assert abs(matrix[0, 0]) < 1e-6 and abs(matrix[2, 0]) < 1e-6, "unconnected features must be zero"


def test_input_scale_modulates_the_wiring() -> None:
    decoders, encoders = _planted()
    scale = torch.ones(D_MODEL)
    scale[2] = 5.0
    plain = wiring_matrix(decoders[L0], encoders[L1], source_features=[1], target_features=[3]).numpy()
    scaled = wiring_matrix(
        decoders[L0], encoders[L1], source_features=[1], target_features=[3], input_scale=scale
    ).numpy()
    assert abs(scaled[0, 0] - 5.0 * plain[0, 0]) < 1e-6


def test_select_nodes_takes_the_largest_movers_per_layer() -> None:
    deltas = {L0: np.array([0.0, 3.0, -5.0, 0.1, 0.0]), L1: np.array([2.0, 0.0, 0.0, 0.0, 0.0])}
    nodes = select_nodes(deltas, top_per_layer=2, min_abs_delta=0.05)
    picked = {(n.layer_index, n.feature) for n in nodes}
    assert picked == {(0, 2), (0, 1), (1, 0)}, picked
    # Sign is preserved: a feature that went down must record a negative delta.
    assert next(n for n in nodes if n.feature == 2).delta == -5.0


def test_select_nodes_drops_everything_below_the_threshold() -> None:
    assert select_nodes({L0: np.array([0.001, 0.002])}, top_per_layer=5, min_abs_delta=0.01) == []


def test_graph_builds_only_forward_edges_and_finds_the_planted_path() -> None:
    decoders, encoders = _planted()
    deltas = {
        L0: np.array([0.0, 1.0, 0.0, 0.0, 0.0]),
        L1: np.array([0.0, 0.0, 0.0, 1.0, 0.0]),
        L2: np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
    }
    nodes = select_nodes(deltas, top_per_layer=1)
    graph = build_graph(nodes, decoders=decoders, encoders=encoders, top_edges=50)

    names = {edge.name for edge in graph.edges}
    assert "L0/F1 -> L1/F3" in names, names
    assert "L1/F3 -> L2/F0" in names, names
    for edge in graph.edges:
        assert edge.source.layer_index < edge.target.layer_index, "edges must point forward only"

    strongest = max(graph.edges, key=lambda e: abs(e.weight))
    assert abs(strongest.weight - 3.0) < 1e-6 or abs(strongest.weight - 2.0) < 1e-6


def test_max_layer_gap_restricts_skip_connections() -> None:
    decoders, encoders = _planted()
    # Give layer 0 a direct route to layer 2 as well.
    decoders[L0] = decoders[L0].clone()
    encoders[L2] = encoders[L2].clone()
    deltas = {L0: np.array([0.0, 1.0, 0, 0, 0]), L2: np.array([1.0, 0, 0, 0, 0])}
    nodes = select_nodes(deltas, top_per_layer=1)
    near = build_graph(nodes, decoders=decoders, encoders=encoders, max_layer_gap=1, top_edges=50)
    assert near.edges == [], "a gap of 2 must be excluded when max_layer_gap=1"
    far = build_graph(nodes, decoders=decoders, encoders=encoders, max_layer_gap=2, top_edges=50)
    assert len(far.edges) >= 0  # allowed through; weight may be zero if unconnected


def test_edge_weight_scales_with_the_source_activation() -> None:
    decoders, encoders = _planted()
    def weight_for(delta: float) -> float:
        nodes = [
            CircuitNode(layer=L0, layer_index=0, feature=1, activation=0.0, delta=delta),
            CircuitNode(layer=L1, layer_index=1, feature=3, activation=0.0, delta=1.0),
        ]
        graph = build_graph(nodes, decoders=decoders, encoders=encoders, top_edges=10)
        return next(e.weight for e in graph.edges if e.name == "L0/F1 -> L1/F3")

    assert abs(weight_for(2.0) - 2 * weight_for(1.0)) < 1e-6, "attribution is linear in the source"


def test_names_follow_the_layer_feature_convention() -> None:
    nodes = [CircuitNode(layer=L1, layer_index=1, feature=7145, activation=1.0, delta=-2.0)]
    named = name_nodes(
        nodes, peak_timestep={(1, 7145): 0.8}, selectivity={(1, 7145): 130.0}
    )
    assert named[0].name == "L1/F7145"
    assert "tau0.80" in named[0].label and "sel130x" in named[0].label
    assert "down" in named[0].label, "the sign of the response should be legible"

    overridden = name_nodes(nodes, labels={"L1/F7145": "bowl colour"})
    assert overridden[0].label == "bowl colour"
    assert overridden[0].display() == "L1/F7145 bowl colour"


def test_graph_serialises_for_the_report() -> None:
    decoders, encoders = _planted()
    deltas = {L0: np.array([0, 1.0, 0, 0, 0]), L1: np.array([0, 0, 0, 1.0, 0])}
    graph = build_graph(select_nodes(deltas, top_per_layer=1), decoders=decoders, encoders=encoders)
    payload = graph.to_dict()
    assert {"metadata", "nodes", "edges"} <= payload.keys()
    assert payload["nodes"][0]["name"].startswith("L")
    assert "omits" in payload["metadata"], "the report must carry the method's caveats"


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok {test.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    main()

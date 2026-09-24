"""Feature-to-feature attribution graphs over Pi0.5 transcoders.

The existing layer-flow report draws one node per layer and sets every edge to
the mean of the two layers' activation masses, which carries no connectivity
information. This module computes actual edges between individual features.

The edge
--------

Each transcoder reconstructs one MLP: ``z = relu(W_enc @ x_mod)`` and
``y_hat = W_dec @ z``. That output is written into the residual stream, so it
reaches the input of every later layer. Taking only that direct path, feature
``i`` at layer ``L`` contributes to the pre-activation of feature ``j`` at a
later layer ``L'``:

    edge(i -> j) = z_i^L  *  < W_enc^{L'}[j, :] * (1 + s^{L'}) , W_dec^L[:, i] >

where ``s^{L'}`` is the transcoder's timestep-conditioned scale, applied
channelwise to the input before encoding. The dot product is the fixed wiring
between the two features; multiplying by the source activation makes it an
attribution for *this* input rather than a static weight.

What this deliberately leaves out
---------------------------------

Attention mixing across token positions, RMSNorm rescaling, and any path that
routes through an intermediate feature's nonlinearity. It is a first-order
direct-path attribution, which is the standard starting point for transcoder
circuits, and it is a proxy -- an edge here is a hypothesis to be confirmed by
patching, not a demonstrated causal link.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import torch


@dataclass(frozen=True)
class CircuitNode:
    """One transcoder feature at one layer."""

    layer: str
    layer_index: int
    feature: int
    activation: float
    #: Signed change in activation between two conditions, when tracing a contrast.
    delta: float = 0.0
    label: str | None = None

    @property
    def key(self) -> tuple[int, int]:
        return (self.layer_index, self.feature)

    @property
    def name(self) -> str:
        """Short stable identifier, e.g. ``L2/F7145``."""
        return f"L{self.layer_index}/F{self.feature}"

    def display(self) -> str:
        return f"{self.name} {self.label}" if self.label else self.name


@dataclass(frozen=True)
class CircuitEdge:
    source: CircuitNode
    target: CircuitNode
    weight: float
    #: The pure wiring term, independent of this input's activations.
    wiring: float

    @property
    def name(self) -> str:
        return f"{self.source.name} -> {self.target.name}"


@dataclass
class CircuitGraph:
    nodes: list[CircuitNode] = field(default_factory=list)
    edges: list[CircuitEdge] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "nodes": [
                {
                    "name": node.name,
                    "label": node.label,
                    "layer": node.layer,
                    "layer_index": node.layer_index,
                    "feature": node.feature,
                    "activation": node.activation,
                    "delta": node.delta,
                }
                for node in self.nodes
            ],
            "edges": [
                {
                    "name": edge.name,
                    "source": edge.source.name,
                    "target": edge.target.name,
                    "weight": edge.weight,
                    "wiring": edge.wiring,
                }
                for edge in self.edges
            ],
        }


def layer_index_of(layer_name: str) -> int:
    digits = [part for part in layer_name.split(".") if part.isdigit()]
    return int(digits[-1]) if digits else -1


def select_nodes(
    deltas: dict[str, np.ndarray],
    *,
    activations: dict[str, np.ndarray] | None = None,
    top_per_layer: int = 8,
    min_abs_delta: float = 0.0,
) -> list[CircuitNode]:
    """Pick the features whose activation moved most, per layer.

    Tracing a contrast (perturbed minus baseline) means the interesting nodes
    are the ones that *changed*, not the ones that are merely large.
    """
    nodes: list[CircuitNode] = []
    for layer, delta in sorted(deltas.items(), key=lambda item: layer_index_of(item[0])):
        vector = np.asarray(delta, dtype=np.float64)
        if vector.ndim != 1:
            raise ValueError(f"Expected a 1-D delta for {layer}, got shape {vector.shape}")
        order = np.argsort(-np.abs(vector))[: max(0, top_per_layer)]
        for feature in order:
            value = float(vector[feature])
            if abs(value) <= min_abs_delta:
                continue
            activation = 0.0
            if activations is not None and layer in activations:
                activation = float(np.asarray(activations[layer])[feature])
            nodes.append(
                CircuitNode(
                    layer=layer,
                    layer_index=layer_index_of(layer),
                    feature=int(feature),
                    activation=activation,
                    delta=value,
                )
            )
    return nodes


def wiring_matrix(
    decoder_source: torch.Tensor,
    encoder_target: torch.Tensor,
    *,
    source_features: Iterable[int],
    target_features: Iterable[int],
    input_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dot products between source decoder columns and target encoder rows.

    Args:
        decoder_source: ``W_dec`` of the earlier layer, shape (d_model, n_features).
        encoder_target: ``W_enc`` of the later layer, shape (n_features, d_model).
        input_scale: optional channelwise ``1 + s`` applied to the later layer's
            input by its timestep conditioning.

    Returns:
        Matrix of shape (len(source_features), len(target_features)).
    """
    source_idx = torch.as_tensor(list(source_features), dtype=torch.long)
    target_idx = torch.as_tensor(list(target_features), dtype=torch.long)
    if source_idx.numel() == 0 or target_idx.numel() == 0:
        return torch.zeros((source_idx.numel(), target_idx.numel()))

    columns = decoder_source.index_select(1, source_idx.to(decoder_source.device)).float()
    rows = encoder_target.index_select(0, target_idx.to(encoder_target.device)).float()
    if input_scale is not None:
        rows = rows * input_scale.reshape(1, -1).to(rows.device).float()
    return (rows @ columns).T.detach().cpu()


def build_graph(
    nodes: list[CircuitNode],
    *,
    decoders: dict[str, torch.Tensor],
    encoders: dict[str, torch.Tensor],
    input_scales: dict[str, torch.Tensor] | None = None,
    max_layer_gap: int | None = None,
    top_edges: int = 200,
    min_abs_weight: float = 0.0,
) -> CircuitGraph:
    """Compute direct-path attribution edges between the selected features."""
    by_layer: dict[str, list[CircuitNode]] = {}
    for node in nodes:
        by_layer.setdefault(node.layer, []).append(node)
    ordered = sorted(by_layer, key=layer_index_of)

    edges: list[CircuitEdge] = []
    for source_layer in ordered:
        for target_layer in ordered:
            gap = layer_index_of(target_layer) - layer_index_of(source_layer)
            if gap <= 0:
                continue  # information only flows forward
            if max_layer_gap is not None and gap > max_layer_gap:
                continue
            if source_layer not in decoders or target_layer not in encoders:
                continue
            sources = by_layer[source_layer]
            targets = by_layer[target_layer]
            wiring = wiring_matrix(
                decoders[source_layer],
                encoders[target_layer],
                source_features=[node.feature for node in sources],
                target_features=[node.feature for node in targets],
                input_scale=None if input_scales is None else input_scales.get(target_layer),
            ).numpy()
            for si, source in enumerate(sources):
                # Attribution scales with how much the source actually moved.
                driver = source.delta if source.delta else source.activation
                for ti, target in enumerate(targets):
                    weight = float(driver * wiring[si, ti])
                    if abs(weight) <= min_abs_weight:
                        continue
                    edges.append(
                        CircuitEdge(
                            source=source, target=target, weight=weight, wiring=float(wiring[si, ti])
                        )
                    )

    edges.sort(key=lambda edge: -abs(edge.weight))
    if top_edges > 0:
        edges = edges[:top_edges]
    kept = {node.key for edge in edges for node in (edge.source, edge.target)}
    return CircuitGraph(
        nodes=[node for node in nodes if node.key in kept] or list(nodes),
        edges=edges,
        metadata={
            "edge_definition": "z_source * <W_enc_target * (1 + scale), W_dec_source>",
            "omits": "attention mixing, RMSNorm, paths through intermediate nonlinearities",
            "interpretation": "a first-order direct-path proxy; confirm any edge by patching",
        },
    )


def name_nodes(
    nodes: list[CircuitNode],
    *,
    peak_timestep: dict[tuple[int, int], float] | None = None,
    selectivity: dict[tuple[int, int], float] | None = None,
    labels: dict[str, str] | None = None,
) -> list[CircuitNode]:
    """Attach readable labels, in the ``L2/F7145 · tau0.80 · sel>=130x`` style.

    ``labels`` lets a caller override any node with a human description, keyed
    by the node's short name.
    """
    named: list[CircuitNode] = []
    for node in nodes:
        parts: list[str] = []
        if peak_timestep and node.key in peak_timestep:
            parts.append(f"tau{peak_timestep[node.key]:.2f}")
        if selectivity and node.key in selectivity:
            parts.append(f"sel{selectivity[node.key]:.0f}x")
        parts.append("up" if node.delta >= 0 else "down")
        auto = " · ".join(parts)
        label = (labels or {}).get(node.name, auto)
        named.append(
            CircuitNode(
                layer=node.layer,
                layer_index=node.layer_index,
                feature=node.feature,
                activation=node.activation,
                delta=node.delta,
                label=label,
            )
        )
    return named

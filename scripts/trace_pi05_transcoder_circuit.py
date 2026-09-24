#!/usr/bin/env python
"""Build and draw a feature-level circuit from a counterfactual probe run.

Unlike the layer-flow report, whose nodes are layers and whose edges are the
mean of two layers' activation masses, this traces individual transcoder
features and computes a real attribution between them.

It runs offline from a probe run's artifacts plus the transcoder checkpoint:
the per-feature deltas say which features responded to the perturbation, and
the encoder/decoder weights say how those features are wired to each other.
No policy forward pass is needed.

Nodes are named ``L<layer>/F<feature>`` with a short descriptor, the same
shape of identifier used for transcoder features elsewhere, so a circuit can
be referred to by name once found.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pi05_mi.circuit_tracing import (  # noqa: E402
    build_graph,
    layer_index_of,
    name_nodes,
    select_nodes,
)
from pi05_mi.transcoders import TimeConditionedTranscoder, TimeConditionedTranscoderConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("probe_run", type=Path, help="A counterfactual probe run directory.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--condition", default="target", help="Which probe condition to trace.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--nodes-per-layer", type=int, default=6)
    parser.add_argument("--top-edges", type=int, default=150)
    parser.add_argument("--max-layer-gap", type=int, default=None, help="Omit to allow skip connections.")
    parser.add_argument(
        "--timestep",
        type=float,
        default=0.5,
        help="Diffusion time at which to evaluate the transcoders' input scaling.",
    )
    parser.add_argument("--labels", type=Path, default=None, help="JSON of {node name: human label}.")
    parser.add_argument("--title", default="Pi0.5 Transcoder Circuit")
    return parser.parse_args()


def load_deltas(run_dir: Path, condition: str) -> tuple[dict[str, np.ndarray], dict[tuple[int, int], float]]:
    """Mean per-feature delta per layer, plus each feature's peak denoise step."""
    path = run_dir / "latent_deltas.csv"
    if not path.exists():
        raise SystemExit(f"No latent_deltas.csv in {run_dir}")

    sums: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    width: dict[str, int] = {}
    peak: dict[tuple[int, int], tuple[float, float]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("condition") != condition:
                continue
            layer = row["layer"]
            width[layer] = max(width.get(layer, 0), int(float(row.get("features") or 0)))
            try:
                ids = json.loads(row.get("top_feature_ids") or "[]")
                deltas = json.loads(row.get("top_feature_deltas") or "[]")
                step = float(row.get("denoise_step") or 0)
            except (json.JSONDecodeError, ValueError):
                continue
            for feature, delta in zip(ids, deltas):
                sums[layer][int(feature)].append(float(delta))
                key = (layer_index_of(layer), int(feature))
                if key not in peak or abs(delta) > peak[key][1]:
                    peak[key] = (step, abs(float(delta)))

    if not sums:
        raise SystemExit(f"No rows with condition={condition!r} in {path}")

    vectors: dict[str, np.ndarray] = {}
    for layer, features in sums.items():
        size = width.get(layer) or (max(features) + 1)
        vector = np.zeros(size, dtype=np.float64)
        for feature, values in features.items():
            vector[feature] = float(np.mean(values))
        vectors[layer] = vector
    return vectors, {key: value[0] for key, value in peak.items()}


def load_selectivity(run_dir: Path) -> dict[tuple[int, int], float]:
    path = run_dir / "candidate_features.csv"
    if not path.exists():
        return {}
    out: dict[tuple[int, int], float] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                out[(int(row["layer_index"]), int(row["feature"]))] = float(row["selectivity"])
            except (KeyError, TypeError, ValueError):
                continue
    return out


def load_transcoder_weights(
    checkpoint_path: Path, timestep: float
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return decoder weights, encoder weights, and the input scale at ``timestep``."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    decoders: dict[str, torch.Tensor] = {}
    encoders: dict[str, torch.Tensor] = {}
    scales: dict[str, torch.Tensor] = {}
    for name, raw_config in checkpoint["configs"].items():
        transcoder = TimeConditionedTranscoder(TimeConditionedTranscoderConfig(**raw_config))
        transcoder.load_state_dict(checkpoint["state_dicts"][name])
        transcoder.eval()
        encoders[name] = transcoder.encoder.weight.detach()
        decoders[name] = transcoder.decoder.weight.detach()
        with torch.no_grad():
            scale, _shift = transcoder._time_scale_shift(  # noqa: SLF001 - the public path needs an input tensor
                torch.tensor([timestep], dtype=torch.float32), dtype=torch.float32, device=torch.device("cpu")
            )
        scales[name] = (1.0 + scale.reshape(-1)).detach()
    return decoders, encoders, scales


def render_svg(graph, title: str) -> str:
    nodes = graph.nodes
    edges = graph.edges
    if not nodes:
        return "<svg xmlns='http://www.w3.org/2000/svg' width='400' height='80'><text x='12' y='40'>no nodes</text></svg>"

    layers = sorted({node.layer_index for node in nodes})
    column_x = {layer: 140 + i * 210 for i, layer in enumerate(layers)}
    rows: dict[int, list] = defaultdict(list)
    for node in sorted(nodes, key=lambda n: (n.layer_index, -abs(n.delta))):
        rows[node.layer_index].append(node)
    position: dict[tuple[int, int], tuple[float, float]] = {}
    for layer, items in rows.items():
        for i, node in enumerate(items):
            position[node.key] = (column_x[layer], 120 + i * 62)

    width = 140 + len(layers) * 210
    height = 160 + max(len(items) for items in rows.values()) * 62
    max_weight = max((abs(edge.weight) for edge in edges), default=1.0) or 1.0
    max_delta = max((abs(node.delta) for node in nodes), default=1.0) or 1.0

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' "
        f"font-family='Arial,Helvetica,sans-serif'>",
        f"<rect width='100%' height='100%' fill='#ffffff'/>",
        f"<text x='28' y='42' font-size='19' fill='#15171a'>{html.escape(title)}</text>",
        "<text x='28' y='66' font-size='12' fill='#6b7280'>"
        "nodes are transcoder features; edge width is direct-path attribution; "
        "blue edges excite, orange inhibit</text>",
    ]
    for layer in layers:
        parts.append(
            f"<text x='{column_x[layer]}' y='96' font-size='12' fill='#6b7280' "
            f"text-anchor='middle'>layer {layer}</text>"
        )

    for edge in edges:
        if edge.source.key not in position or edge.target.key not in position:
            continue
        x1, y1 = position[edge.source.key]
        x2, y2 = position[edge.target.key]
        strength = abs(edge.weight) / max_weight
        colour = "#2f5f9f" if edge.weight >= 0 else "#b05a2a"
        mid = (x1 + x2) / 2
        parts.append(
            f"<path d='M {x1 + 52} {y1} C {mid} {y1}, {mid} {y2}, {x2 - 52} {y2}' fill='none' "
            f"stroke='{colour}' stroke-width='{0.6 + 5.0 * strength:.2f}' "
            f"stroke-opacity='{0.25 + 0.6 * strength:.2f}'><title>{html.escape(edge.name)}: "
            f"{edge.weight:+.4g}</title></path>"
        )

    for node in nodes:
        if node.key not in position:
            continue
        x, y = position[node.key]
        intensity = abs(node.delta) / max_delta
        fill = "#dbeafe" if node.delta >= 0 else "#ffedd5"
        stroke = "#2f5f9f" if node.delta >= 0 else "#b05a2a"
        parts.append(
            f"<g><title>{html.escape(node.display())}  delta={node.delta:+.4g}</title>"
            f"<rect x='{x - 52}' y='{y - 20}' rx='7' width='104' height='40' fill='{fill}' "
            f"stroke='{stroke}' stroke-width='{1 + 2 * intensity:.2f}'/>"
            f"<text x='{x}' y='{y - 3}' font-size='12' text-anchor='middle' fill='#15171a'>"
            f"{html.escape(node.name)}</text>"
            f"<text x='{x}' y='{y + 12}' font-size='9' text-anchor='middle' fill='#6b7280'>"
            f"{html.escape((node.label or '')[:22])}</text></g>"
        )
    parts.append("</svg>")
    return "\n".join(parts)


def render_html(graph, svg: str, title: str, meta: dict[str, Any]) -> str:
    edge_rows = "".join(
        f"<tr><td>{html.escape(e.source.display())}</td><td>{html.escape(e.target.display())}</td>"
        f"<td>{e.weight:+.5g}</td><td>{e.wiring:+.5g}</td></tr>"
        for e in graph.edges[:120]
    )
    node_rows = "".join(
        f"<tr><td>{html.escape(n.name)}</td><td>{html.escape(n.label or '')}</td>"
        f"<td>{n.layer_index}</td><td>{n.feature}</td><td>{n.delta:+.5g}</td></tr>"
        for n in sorted(graph.nodes, key=lambda n: (n.layer_index, -abs(n.delta)))
    )
    return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>{html.escape(title)}</title>
<style>
body{{font-family:Arial,Helvetica,sans-serif;margin:24px;color:#15171a}}
table{{border-collapse:collapse;margin:12px 0;font-size:13px}}
th,td{{border:1px solid #d7dde7;padding:5px 9px;text-align:left}}
th{{background:#f4f7fb}} .note{{color:#6b7280;max-width:900px;line-height:1.5}}
svg{{border:1px solid #e5e7eb;border-radius:8px;max-width:100%}}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p class='note'>Each node is one transcoder feature, named
<code>L&lt;layer&gt;/F&lt;feature&gt;</code>. An edge is the first-order direct-path
attribution <code>{html.escape(graph.metadata.get('edge_definition', ''))}</code>:
how much of the target feature's pre-activation is explained by the source
feature writing into the residual stream.</p>
<p class='note'><b>Caveat.</b> This omits {html.escape(graph.metadata.get('omits', ''))}.
An edge here is a hypothesis to confirm by patching, not a demonstrated causal link.</p>
{svg}
<h2>Nodes ({len(graph.nodes)})</h2>
<table><tr><th>name</th><th>label</th><th>layer</th><th>feature</th><th>delta</th></tr>{node_rows}</table>
<h2>Edges (top {min(120, len(graph.edges))} of {len(graph.edges)})</h2>
<table><tr><th>source</th><th>target</th><th>attribution</th><th>wiring</th></tr>{edge_rows}</table>
<h2>Provenance</h2><pre>{html.escape(json.dumps(meta, indent=2))}</pre>
</body></html>"""


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.probe_run / "circuit")
    output_dir.mkdir(parents=True, exist_ok=True)

    deltas, peak_timestep = load_deltas(args.probe_run, args.condition)
    selectivity = load_selectivity(args.probe_run)
    print(f"loaded deltas for {len(deltas)} layers from {args.probe_run}", flush=True)

    decoders, encoders, scales = load_transcoder_weights(args.checkpoint, args.timestep)
    print(f"loaded {len(encoders)} transcoders from {args.checkpoint}", flush=True)

    nodes = select_nodes(deltas, top_per_layer=args.nodes_per_layer)
    labels = json.loads(args.labels.read_text()) if args.labels and args.labels.exists() else None
    nodes = name_nodes(nodes, peak_timestep=peak_timestep, selectivity=selectivity, labels=labels)
    print(f"selected {len(nodes)} feature nodes", flush=True)

    graph = build_graph(
        nodes,
        decoders=decoders,
        encoders=encoders,
        input_scales=scales,
        max_layer_gap=args.max_layer_gap,
        top_edges=args.top_edges,
    )
    print(f"built {len(graph.edges)} edges across {len(graph.nodes)} connected nodes", flush=True)

    meta = {
        "probe_run": str(args.probe_run),
        "checkpoint": str(args.checkpoint),
        "condition": args.condition,
        "timestep": args.timestep,
        "nodes_per_layer": args.nodes_per_layer,
        "max_layer_gap": args.max_layer_gap,
        **graph.metadata,
    }
    graph.metadata.update(meta)

    svg = render_svg(graph, args.title)
    (output_dir / "circuit.svg").write_text(svg, encoding="utf-8")
    (output_dir / "circuit.html").write_text(render_html(graph, svg, args.title, meta), encoding="utf-8")
    (output_dir / "circuit.json").write_text(json.dumps(graph.to_dict(), indent=2), encoding="utf-8")
    with (output_dir / "circuit_edges.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "target", "attribution", "wiring", "source_delta", "target_delta"])
        for edge in graph.edges:
            writer.writerow(
                [edge.source.name, edge.target.name, edge.weight, edge.wiring,
                 edge.source.delta, edge.target.delta]
            )

    print("\nstrongest edges")
    for edge in graph.edges[:15]:
        print(f"  {edge.source.display():<34} -> {edge.target.display():<34} {edge.weight:+.5g}")
    print(f"\nArtifacts in {output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Render an aggregate layer-flow report for Pi0.5 transcoder activations."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from pi05_mi.langfuse_tracing import make_langfuse_tracer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional trained transcoder checkpoint. Defaults to config.json checkpoint if present.",
    )
    parser.add_argument("--output-html", type=Path, default=None)
    parser.add_argument("--output-svg", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--top-features-per-layer", type=int, default=6)
    parser.add_argument("--min-frequency", type=float, default=0.0)
    parser.add_argument("--title", default="Pi0.5 Transcoder Layer Activation Flow")
    parser.add_argument("--langfuse-trace", action="store_true", help="Enable Langfuse tracing for this report render.")
    return parser.parse_args()


def _read_config(feature_dir: Path) -> dict[str, Any]:
    path = feature_dir / "config.json"
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def _default_output(path: Path | None, feature_dir: Path, name: str) -> Path:
    return path if path is not None else feature_dir / name


def _layer_index(name: str, payload: dict[str, Any]) -> int:
    indices = payload.get("layer_indices") or {}
    if name in indices:
        return int(indices[name])
    if ".layers." in name:
        return int(name.split(".layers.", 1)[1].split(".", 1)[0])
    raise ValueError(f"Could not infer layer index for {name}")


def _load_decoder_norms(checkpoint_path: Path | None, layer_names: list[str]) -> dict[str, torch.Tensor]:
    if checkpoint_path is None or not checkpoint_path.exists():
        return {}
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dicts = checkpoint.get("state_dicts", {})
    norms: dict[str, torch.Tensor] = {}
    for name in layer_names:
        state = state_dicts.get(name)
        if not isinstance(state, dict):
            continue
        weight = state.get("decoder.weight")
        if not torch.is_tensor(weight) or weight.ndim != 2:
            continue
        norms[name] = weight.float().norm(dim=0)
    return norms


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isnan(number):
            return ""
        if abs(number - round(number)) < 1e-9:
            return str(int(round(number)))
        return f"{number:.{digits}g}"
    return str(value)


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.ndim == 0:
            return _jsonable(value.item())
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _summarize_layer(
    *,
    name: str,
    stats_payload: dict[str, Any],
    decoder_norm: torch.Tensor | None,
    top_features_per_layer: int,
    min_frequency: float,
) -> dict[str, Any]:
    timestep_stats = stats_payload["stats"][name]
    layer = _layer_index(name, stats_payload)
    feature_score: torch.Tensor | None = None
    feature_mean: torch.Tensor | None = None
    feature_frequency: torch.Tensor | None = None
    feature_top_m_frequency: torch.Tensor | None = None
    feature_peak_score: torch.Tensor | None = None
    feature_peak_tau: torch.Tensor | None = None
    timestep_rows: list[dict[str, Any]] = []
    total_records = 0

    for offset, timestep_key in enumerate(sorted(timestep_stats, key=float)):
        state = timestep_stats[timestep_key]
        mean_vec = state["mean"].float().clamp_min(0)
        frequency = state["firing_frequency"].float()
        top_m_frequency = state.get("top_m_frequency", torch.zeros_like(frequency)).float()
        if decoder_norm is not None and decoder_norm.numel() == mean_vec.numel():
            score_vec = mean_vec * decoder_norm.float()
        else:
            score_vec = mean_vec
        if min_frequency > 0:
            score_vec = torch.where(frequency >= min_frequency, score_vec, torch.zeros_like(score_vec))
        total_records += int(state.get("count", 0))
        mass = float(score_vec.sum())
        expected_active = float(frequency.sum())
        timestep_rows.append(
            {
                "timestep": float(timestep_key),
                "activation_mass": mass,
                "expected_active_features": expected_active,
                "mean_frequency": float(frequency.mean()),
                "peak_feature_score": float(score_vec.max()) if score_vec.numel() else 0.0,
            }
        )

        feature_score = score_vec.clone() if feature_score is None else feature_score + score_vec
        feature_mean = mean_vec.clone() if feature_mean is None else feature_mean + mean_vec
        feature_frequency = frequency.clone() if feature_frequency is None else feature_frequency + frequency
        feature_top_m_frequency = (
            top_m_frequency.clone()
            if feature_top_m_frequency is None
            else feature_top_m_frequency + top_m_frequency
        )
        if feature_peak_score is None:
            feature_peak_score = score_vec.clone()
            feature_peak_tau = torch.full_like(score_vec, float(timestep_key))
        else:
            mask = score_vec > feature_peak_score
            feature_peak_score = torch.where(mask, score_vec, feature_peak_score)
            assert feature_peak_tau is not None
            feature_peak_tau = torch.where(mask, torch.full_like(feature_peak_tau, float(timestep_key)), feature_peak_tau)

    if feature_score is None:
        raise ValueError(f"No timestep stats found for {name}")

    n_timesteps = max(1, len(timestep_rows))
    feature_mean = feature_mean / n_timesteps
    feature_frequency = feature_frequency / n_timesteps
    feature_top_m_frequency = feature_top_m_frequency / n_timesteps
    k = min(max(1, top_features_per_layer), int(feature_score.numel()))
    top_values, top_indices = torch.topk(feature_score, k=k)
    top_features = []
    assert feature_peak_tau is not None
    for rank, (score, feature) in enumerate(zip(top_values.tolist(), top_indices.tolist(), strict=False), start=1):
        top_features.append(
            {
                "rank": rank,
                "feature": int(feature),
                "score": float(score),
                "mean_activation": float(feature_mean[feature]),
                "frequency": float(feature_frequency[feature]),
                "top_m_frequency": float(feature_top_m_frequency[feature]),
                "peak_timestep": float(feature_peak_tau[feature]),
            }
        )

    total_mass = float(feature_score.sum())
    peak_timestep = max(timestep_rows, key=lambda row: row["activation_mass"]) if timestep_rows else {}
    return {
        "layer": layer,
        "layer_name": name,
        "timesteps": len(timestep_rows),
        "record_count_sum": total_records,
        "activation_mass": total_mass,
        "mean_activation_mass_per_timestep": total_mass / n_timesteps,
        "expected_active_features": float(feature_frequency.sum()),
        "mean_feature_frequency": float(feature_frequency.mean()),
        "peak_timestep": peak_timestep.get("timestep"),
        "peak_timestep_mass": peak_timestep.get("activation_mass"),
        "timestep_profile": timestep_rows,
        "top_features": top_features,
    }


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    config = _read_config(args.feature_dir)
    if args.checkpoint is None and config.get("checkpoint"):
        args.checkpoint = Path(config["checkpoint"])
    stats_payload = torch.load(args.feature_dir / "feature_stats.pt", map_location="cpu", weights_only=False)
    if int(stats_payload.get("format_version", 1)) != 2:
        raise ValueError("Expected feature_stats.pt format_version=2")

    layer_names = sorted(stats_payload["layer_names"], key=lambda name: _layer_index(name, stats_payload))
    decoder_norms = _load_decoder_norms(args.checkpoint, layer_names)
    layers = [
        _summarize_layer(
            name=name,
            stats_payload=stats_payload,
            decoder_norm=decoder_norms.get(name),
            top_features_per_layer=args.top_features_per_layer,
            min_frequency=args.min_frequency,
        )
        for name in layer_names
    ]
    max_mass = max((layer["activation_mass"] for layer in layers), default=1.0)
    min_mass = min((layer["activation_mass"] for layer in layers), default=0.0)
    edges = []
    for source, target in zip(layers, layers[1:], strict=False):
        source_mass = float(source["activation_mass"])
        target_mass = float(target["activation_mass"])
        edges.append(
            {
                "source_layer": source["layer"],
                "target_layer": target["layer"],
                "source_mass": source_mass,
                "target_mass": target_mass,
                "edge_mass": (source_mass + target_mass) / 2.0,
                "delta": target_mass - source_mass,
                "delta_ratio": None if source_mass == 0 else (target_mass - source_mass) / source_mass,
            }
        )
    return {
        "format_version": 1,
        "description": (
            "Aggregate Pi0.5 transcoder layer-flow summary. Edge widths are adjacent-layer "
            "activation statistics, not causal proof of connectivity."
        ),
        "feature_dir": str(args.feature_dir),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "attribution_metric": "decoder_weighted_mean_z" if decoder_norms else "mean_z",
        "config": config,
        "normalization": {
            "min_layer_mass": min_mass,
            "max_layer_mass": max_mass,
            "min_frequency": args.min_frequency,
        },
        "layers": layers,
        "edges": edges,
    }


def _html_escape(value: Any) -> str:
    return html.escape(_fmt(value))


def render_svg(summary: dict[str, Any], title: str) -> str:
    layers = summary["layers"]
    edges = summary["edges"]
    n_layers = len(layers)
    node_w = 142
    node_h = 246
    gap = 72
    margin_x = 48
    width = max(1180, margin_x * 2 + n_layers * node_w + max(0, n_layers - 1) * gap)
    height = 650
    top = 170
    node_y = 250
    max_mass = max(float(summary["normalization"]["max_layer_mass"]), 1e-9)
    min_mass = float(summary["normalization"]["min_layer_mass"])

    def x_for(index: int) -> float:
        return margin_x + index * (node_w + gap)

    def mass_scale(value: float) -> float:
        if max_mass <= min_mass:
            return 1.0
        return max(0.0, min(1.0, (value - min_mass) / (max_mass - min_mass)))

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)}'>",
        "<defs>",
        "<marker id='arrow' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' orient='auto-start-reverse'><path d='M0,0 L10,5 L0,10 Z' fill='#2f5f9f'/></marker>",
        "<linearGradient id='nodeGrad' x1='0' x2='1'><stop offset='0' stop-color='#eef6ff'/><stop offset='1' stop-color='#fff7e8'/></linearGradient>",
        "</defs>",
        "<rect x='0' y='0' width='100%' height='100%' fill='#ffffff'/>",
        f"<text x='48' y='44' font-family='Arial,Helvetica,sans-serif' font-size='24' font-weight='700' fill='#15171a'>{html.escape(title)}</text>",
        f"<text x='48' y='72' font-family='Arial,Helvetica,sans-serif' font-size='13' fill='#4f5d6b'>metric: {html.escape(summary['attribution_metric'])}; layers are separated left to right; width and fill intensity scale with aggregate attribution mass</text>",
        "<text x='48' y='108' font-family='Arial,Helvetica,sans-serif' font-size='12' fill='#6b7280'>Edge labels show relative change in aggregate mass between adjacent action-expert MLP transcoders.</text>",
        "<line x1='48' y1='132' x2='" + str(width - 48) + "' y2='132' stroke='#d7dde7'/>",
    ]

    for idx, edge in enumerate(edges):
        sx = x_for(idx) + node_w
        tx = x_for(idx + 1)
        y = top + 18
        edge_mass = float(edge["edge_mass"])
        stroke_w = 2.0 + 11.0 * mass_scale(edge_mass)
        color = "#b05a2a" if float(edge["delta"]) < 0 else "#2f5f9f"
        c1 = sx + gap * 0.45
        c2 = tx - gap * 0.45
        parts.append(
            f"<path d='M {sx:.1f} {y:.1f} C {c1:.1f} {y-44:.1f}, {c2:.1f} {y-44:.1f}, {tx:.1f} {y:.1f}' "
            f"fill='none' stroke='{color}' stroke-width='{stroke_w:.2f}' stroke-linecap='round' opacity='0.64' marker-end='url(#arrow)'/>"
        )
        ratio = edge.get("delta_ratio")
        label = "n/a" if ratio is None else f"{ratio:+.0%}"
        parts.append(
            f"<text x='{(sx+tx)/2:.1f}' y='{y-54:.1f}' text-anchor='middle' font-family='Arial,Helvetica,sans-serif' font-size='11' fill='#374151'>{label}</text>"
        )

    for idx, layer in enumerate(layers):
        x = x_for(idx)
        mass = float(layer["activation_mass"])
        scaled = mass_scale(mass)
        fill_opacity = 0.55 + 0.35 * scaled
        stroke_w = 1.2 + 2.8 * scaled
        parts.append(
            f"<g transform='translate({x:.1f},{node_y:.1f})'>"
            f"<rect width='{node_w}' height='{node_h}' rx='8' fill='url(#nodeGrad)' fill-opacity='{fill_opacity:.3f}' stroke='#2f5f9f' stroke-width='{stroke_w:.2f}'/>"
            f"<text x='12' y='26' font-family='Arial,Helvetica,sans-serif' font-size='16' font-weight='700' fill='#111827'>Layer {int(layer['layer']):02d}</text>"
            f"<text x='12' y='48' font-family='Menlo,Consolas,monospace' font-size='11' fill='#374151'>mass {_html_escape(layer['activation_mass'])}</text>"
            f"<text x='12' y='66' font-family='Menlo,Consolas,monospace' font-size='11' fill='#374151'>active {_html_escape(layer['expected_active_features'])}</text>"
            f"<text x='12' y='84' font-family='Menlo,Consolas,monospace' font-size='11' fill='#374151'>peak tau {_html_escape(layer['peak_timestep'])}</text>"
        )
        profile = layer["timestep_profile"]
        if profile:
            chart_x = 12
            chart_y = 104
            chart_w = node_w - 24
            chart_h = 48
            max_profile = max(float(row["activation_mass"]) for row in profile) or 1.0
            bar_gap = 2
            bar_w = max(2.0, (chart_w - bar_gap * (len(profile) - 1)) / len(profile))
            parts.append(f"<line x1='{chart_x}' y1='{chart_y+chart_h}' x2='{chart_x+chart_w}' y2='{chart_y+chart_h}' stroke='#cfd6e1'/>")
            for bar_idx, row in enumerate(profile):
                h = chart_h * float(row["activation_mass"]) / max_profile
                bx = chart_x + bar_idx * (bar_w + bar_gap)
                by = chart_y + chart_h - h
                parts.append(f"<rect x='{bx:.1f}' y='{by:.1f}' width='{bar_w:.1f}' height='{h:.1f}' fill='#3f7fbd' opacity='0.8'/>")
        parts.append("<text x='12' y='174' font-family='Arial,Helvetica,sans-serif' font-size='11' font-weight='700' fill='#111827'>top features</text>")
        for feature_idx, feature in enumerate(layer["top_features"][:4]):
            y = 194 + 16 * feature_idx
            parts.append(
                f"<text x='12' y='{y}' font-family='Menlo,Consolas,monospace' font-size='10.5' fill='#374151'>"
                f"F{int(feature['feature'])}: {_html_escape(feature['score'])}</text>"
            )
        parts.append("</g>")

    parts.append("</svg>")
    return "\n".join(parts)


def write_csv(path: Path, summary: dict[str, Any]) -> None:
    fieldnames = [
        "layer",
        "activation_mass",
        "mean_activation_mass_per_timestep",
        "expected_active_features",
        "mean_feature_frequency",
        "peak_timestep",
        "peak_timestep_mass",
        "timesteps",
        "record_count_sum",
        "top_features",
        "layer_name",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for layer in summary["layers"]:
            row = {key: layer.get(key) for key in fieldnames}
            row["top_features"] = " ".join(f"F{item['feature']}:{item['score']:.4g}" for item in layer["top_features"])
            writer.writerow(row)


def render_html(summary: dict[str, Any], svg: str, title: str) -> str:
    layer_rows = []
    for layer in summary["layers"]:
        top_features = ", ".join(
            f"F{item['feature']} ({item['score']:.4g}, tau {item['peak_timestep']:.4g})"
            for item in layer["top_features"]
        )
        layer_rows.append(
            "<tr>"
            f"<td>{int(layer['layer']):02d}</td>"
            f"<td>{_html_escape(layer['activation_mass'])}</td>"
            f"<td>{_html_escape(layer['expected_active_features'])}</td>"
            f"<td>{_html_escape(layer['mean_feature_frequency'])}</td>"
            f"<td>{_html_escape(layer['peak_timestep'])}</td>"
            f"<td>{html.escape(top_features)}</td>"
            "</tr>"
        )
    return "\n".join(
        [
            "<!doctype html>",
            "<html>",
            "<head>",
            "<meta charset='utf-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1'>",
            f"<title>{html.escape(title)}</title>",
            "<style>",
            ":root{color-scheme:light;background:#ffffff;color:#15171a;font-family:Arial,Helvetica,sans-serif}",
            "body{margin:0;background:#ffffff;color:#15171a}",
            "main{max-width:1600px;margin:0 auto;padding:24px}",
            "h1{font-size:28px;margin:0 0 6px}p{margin:6px 0;color:#4f5d6b;max-width:980px;line-height:1.45}",
            ".chart{overflow-x:auto;border:1px solid #d8dee8;margin-top:18px;background:#fff}",
            ".chart svg{display:block;min-width:1180px}",
            "table{width:100%;border-collapse:collapse;margin-top:22px;font-size:13px}",
            "th,td{padding:8px 10px;border-bottom:1px solid #e8edf3;text-align:left;vertical-align:top}",
            "th{background:#f5f7fa;font-size:12px;color:#374151}",
            "td:nth-child(1),td:nth-child(2),td:nth-child(3),td:nth-child(4),td:nth-child(5){font-family:Menlo,Consolas,monospace}",
            ".note{font-size:13px;color:#6b7280}",
            "</style>",
            "</head>",
            "<body>",
            "<main>",
            f"<h1>{html.escape(title)}</h1>",
            "<p>Each node is one Pi0.5 action-expert MLP transcoder layer. Node intensity and edge width summarize sparse latent activity accumulated across collected LIBERO observations and denoising passes.</p>",
            f"<p class='note'>Attribution metric: <b>{html.escape(summary['attribution_metric'])}</b>. This is a transcoder attribution proxy, not proof of a causal edge between layers.</p>",
            f"<p class='note'>Feature dir: {html.escape(summary['feature_dir'])}</p>",
            "<div class='chart'>",
            svg,
            "</div>",
            "<table>",
            "<thead><tr><th>Layer</th><th>Mass</th><th>Expected active features</th><th>Mean firing freq</th><th>Peak tau</th><th>Top features</th></tr></thead>",
            "<tbody>",
            *layer_rows,
            "</tbody>",
            "</table>",
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def main() -> None:
    args = parse_args()
    if args.langfuse_trace:
        os.environ["PI05_LANGFUSE_TRACE"] = "1"
    args.output_html = _default_output(args.output_html, args.feature_dir, "transcoder_flow_report.html")
    args.output_svg = _default_output(args.output_svg, args.feature_dir, "transcoder_flow_chart.svg")
    args.output_json = _default_output(args.output_json, args.feature_dir, "transcoder_flow_summary.json")
    args.output_csv = _default_output(args.output_csv, args.feature_dir, "transcoder_flow_layers.csv")

    tracer = make_langfuse_tracer(feature="pi05-transcoder-flow-report", output_root=args.feature_dir)
    with tracer.trace(
        "render-transcoder-flow-report",
        input={
            "feature_dir": str(args.feature_dir),
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "top_features_per_layer": args.top_features_per_layer,
            "min_frequency": args.min_frequency,
        },
        metadata={"feature_dir": str(args.feature_dir)},
        tags=["flow-report"],
    ) as trace:
        summary = build_summary(args)
        svg = render_svg(summary, args.title)
        args.output_svg.parent.mkdir(parents=True, exist_ok=True)
        args.output_svg.write_text(svg, encoding="utf-8")
        args.output_json.write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True), encoding="utf-8")
        write_csv(args.output_csv, summary)
        args.output_html.write_text(render_html(summary, svg, args.title), encoding="utf-8")

        strongest_layers = sorted(
            (
                {
                    "layer": layer["layer"],
                    "activation_mass": layer["activation_mass"],
                    "peak_timestep": layer["peak_timestep"],
                    "top_features": layer["top_features"][:3],
                }
                for layer in summary["layers"]
            ),
            key=lambda item: float(item["activation_mass"]),
            reverse=True,
        )[:8]
        tracer.update(
            trace,
            output={
                "feature_dir": str(args.feature_dir),
                "layer_count": len(summary["layers"]),
                "edge_count": len(summary["edges"]),
                "attribution_metric": summary["attribution_metric"],
                "strongest_layers": strongest_layers,
                "files": {
                    "html": str(args.output_html),
                    "svg": str(args.output_svg),
                    "json": str(args.output_json),
                    "csv": str(args.output_csv),
                },
                "flow_chart_svg": tracer.media_from_path(args.output_svg, content_type="image/svg+xml"),
            },
        )
    tracer.flush()

    print(f"wrote {args.output_html}", flush=True)
    print(f"wrote {args.output_svg}", flush=True)
    print(f"wrote {args.output_json}", flush=True)
    print(f"wrote {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Sparsity report for Pi0.5 transcoder latents.

Reads ``transcoder_capture/events.jsonl`` and writes a summary JSON plus
two figures that compare observed L0 against the 16384-wide dictionary.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def load_latent_rows(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "transcoder_capture" / "events.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("type") != "transcoder_latent":
            continue
        timesteps = event.get("timestep") or []
        shape = event.get("shape") or []
        width = int(shape[-1]) if shape else 16384
        token_l0 = event.get("token_l0")
        if isinstance(token_l0, list) and token_l0:
            counts = [float(item) for item in np.asarray(token_l0, dtype=np.float64).reshape(-1)]
        else:
            counts = [float(event.get("l0_mean") or 0.0)]
        rows.append(
            {
                "layer": int(event["layer"]),
                "chunk": int(event.get("chunk") or 0),
                "t": round(float(timesteps[0]), 1) if timesteps else None,
                "width": width,
                "l0_mean": float(event.get("l0_mean") or 0.0),
                "token_l0": counts,
            }
        )
    if not rows:
        raise SystemExit(f"no transcoder_latent events in {path}")
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    widths = [row["width"] for row in rows]
    width = int(np.median(widths))
    token_counts = np.asarray([count for row in rows for count in row["token_l0"]], dtype=np.float64)
    token_pct = 100.0 * token_counts / max(width, 1)
    by_layer_l0: dict[int, list[float]] = defaultdict(list)
    by_layer_t: dict[tuple[int, float], list[float]] = defaultdict(list)
    for row in rows:
        mean_l0 = float(np.mean(row["token_l0"]))
        by_layer_l0[row["layer"]].append(mean_l0)
        if row["t"] is not None:
            by_layer_t[(row["layer"], float(row["t"]))].append(mean_l0)
    layers = sorted(by_layer_l0)
    layer_mean_l0 = {str(layer): float(np.mean(by_layer_l0[layer])) for layer in layers}
    return {
        "n_calls": len(rows),
        "n_tokens": int(token_counts.size),
        "dictionary_size": width,
        "mean_l0": float(token_counts.mean()),
        "median_l0": float(np.median(token_counts)),
        "p90_l0": float(np.quantile(token_counts, 0.9)),
        "max_l0": float(token_counts.max()),
        "mean_l0_pct": float(token_pct.mean()),
        "median_l0_pct": float(np.median(token_pct)),
        "sparsity_pct": float(100.0 - token_pct.mean()),
        "layers": layers,
        "layer_mean_l0": layer_mean_l0,
        "layer_mean_pct": {
            str(layer): 100.0 * layer_mean_l0[str(layer)] / max(width, 1) for layer in layers
        },
        "layer_std_pct": {
            str(layer): 100.0 * float(np.std(by_layer_l0[layer])) / max(width, 1) for layer in layers
        },
        "by_layer_t": {
            f"{layer}:{t:.1f}": float(np.mean(values))
            for (layer, t), values in by_layer_t.items()
        },
    }


def _layer_t_means(rows: list[dict[str, Any]], layer: int, times: list[float]) -> tuple[list[float], list[float]]:
    means, stds = [], []
    for t in times:
        values = [float(np.mean(row["token_l0"])) for row in rows if row["layer"] == layer and row["t"] == t]
        means.append(float(np.mean(values)) if values else float("nan"))
        stds.append(float(np.std(values)) if values else float("nan"))
    return means, stds


def plot_report(rows: list[dict[str, Any]], summary: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt

    width = int(summary["dictionary_size"])
    layers = list(summary["layers"])
    token_counts = np.asarray([count for row in rows for count in row["token_l0"]], dtype=np.float64)
    layer_active = [summary["layer_mean_pct"][str(layer)] for layer in layers]
    layer_zero = [100.0 - value for value in layer_active]
    times = sorted({row["t"] for row in rows if row["t"] is not None})

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6), dpi=140)
    fig.suptitle(
        f"Is the transcoder sparse?   mean L0 = {summary['mean_l0']:.0f} / {width} "
        f"({summary['mean_l0_pct']:.2f}% active, {summary['sparsity_pct']:.2f}% zero)   "
        f"n={summary['n_tokens']:,} tokens",
        fontsize=12,
    )

    ax = axes[0, 0]
    ax.barh([1], [width], color="#d5dbe3", height=0.55, label=f"dense dictionary = {width}")
    ax.barh([0], [summary["mean_l0"]], color="#1f4e79", height=0.55, label=f"mean active = {summary['mean_l0']:.0f}")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["observed L0", "dense (all-on)"])
    ax.set_xlabel("Number of features")
    ax.set_xlim(0, width * 1.02)
    ax.set_title("Active features vs dictionary width")
    ax.legend(frameon=False, loc="lower right")
    ax.text(summary["mean_l0"], 0, f"  {summary['mean_l0']:.0f}", va="center", ha="left", fontsize=10)

    ax = axes[0, 1]
    ax.bar(layers, layer_active, color="#1f4e79", label="active")
    ax.bar(layers, layer_zero, bottom=layer_active, color="#d5dbe3", label="zero")
    ax.set_title("Per-layer active vs zero")
    ax.set_xlabel("Action-expert layer")
    ax.set_ylabel("Share of 16384 features (%)")
    ax.set_xticks(layers)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1, 0]
    bins = min(40, max(10, int(np.sqrt(token_counts.size))))
    ax.hist(token_counts, bins=bins, color="#4c6f9c", edgecolor="white")
    ax.axvline(summary["mean_l0"], color="#b42318", linewidth=1.6, label=f"mean {summary['mean_l0']:.0f}")
    ax.axvline(summary["p90_l0"], color="#b42318", linestyle="--", linewidth=1, label=f"p90 {summary['p90_l0']:.0f}")
    ax.set_title("Per-token L0 histogram")
    ax.set_xlabel("Active features on one token")
    ax.set_ylabel("Token count")
    ax.legend(frameon=False)
    ax.text(
        0.98,
        0.95,
        f"dense would be {width}\nmax observed {summary['max_l0']:.0f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )

    ax = axes[1, 1]
    for layer in layers:
        means, _stds = _layer_t_means(rows, layer, times)
        ax.plot(times, means, color="#9aa5b1", linewidth=1)
    if times:
        mean_curve = []
        for t in times:
            values = [float(np.mean(row["token_l0"])) for row in rows if row["t"] == t]
            mean_curve.append(float(np.mean(values)) if values else float("nan"))
        ax.plot(times, mean_curve, color="#1f4e79", linewidth=2.2, label="mean across layers")
    ax.set_title("L0 vs denoise t")
    ax.set_xlabel("t (1.0 = start, 0.1 = end)")
    ax.set_ylabel("Active features")
    if times:
        ax.set_xticks(times)
    ax.legend(frameon=False)

    fig.tight_layout()
    overview = out_dir / "sparsity_overview.png"
    fig.savefig(overview, bbox_inches="tight")
    plt.close(fig)

    n_layers = len(layers)
    nrows = int(np.ceil(n_layers / 3)) if n_layers else 1
    fig, axes = plt.subplots(max(nrows, 1), 3, figsize=(13, 2.35 * max(nrows, 1)), sharex=True, sharey=True, dpi=140)
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, layer in zip(axes_flat, layers):
        means, stds = _layer_t_means(rows, layer, times)
        ax.errorbar(times, means, yerr=stds, marker="o", markersize=3.5, linewidth=1.2, capsize=2, color="#1f4e79")
        ax.set_title(f"L{layer:02d}  mean={summary['layer_mean_l0'][str(layer)]:.0f}/{width}")
        ax.set_xlabel("t")
        ax.set_ylabel("L0")
        ax.set_xticks([0.1, 0.3, 0.5, 0.7, 1.0])
        ax.grid(True, alpha=0.25)
    for ax in axes_flat[n_layers:]:
        ax.set_visible(False)
    fig.suptitle(f"Per-layer token-mean L0 vs denoise t   (dense = {width}; error bar = std over calls)")
    fig.tight_layout()
    by_layer = out_dir / "l0_percent_vs_t_by_layer.png"
    fig.savefig(by_layer, bbox_inches="tight")
    plt.close(fig)
    return overview, by_layer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run
    out_dir = run_dir / "transcoder_capture"
    rows = load_latent_rows(run_dir)
    summary = summarize(rows)
    (out_dir / "sparsity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    overview, by_layer = plot_report(rows, summary, out_dir)
    print(
        f"mean L0={summary['mean_l0']:.1f}/{summary['dictionary_size']} "
        f"({summary['mean_l0_pct']:.2f}%)  zeros={100.0 - summary['mean_l0_pct']:.2f}%"
    )
    print(f"saved {overview}")
    print(f"saved {by_layer}")
    print(f"saved {out_dir / 'sparsity_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

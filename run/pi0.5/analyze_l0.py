#!/usr/bin/env python3
"""Plot token-mean L0% vs denoise t from a local infer.py run.

Unlike the Colab cell, this can filter by the language stored on each
``chunk_start`` event, so mixed-task runs do not get averaged together.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np


def load_events(run_dir: Path) -> list[dict]:
    events_path = run_dir / "transcoder_capture" / "events.jsonl"
    if not events_path.exists():
        raise FileNotFoundError(f"missing {events_path}")
    return [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def chunk_prompts(events: list[dict]) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for ev in events:
        if ev.get("type") == "chunk_start":
            mapping[int(ev["chunk"])] = str(ev.get("task") or "")
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="outputs/pi05_infer/<run_id>")
    parser.add_argument("--task-contains", default="", help="Keep chunks whose prompt contains this substring")
    parser.add_argument("--width", type=int, default=16384)
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    events = load_events(args.run)
    prompts = chunk_prompts(events)
    want = args.task_contains.strip()
    by_layer_t: dict[tuple[int, float], list[float]] = collections.defaultdict(list)
    kept_chunks: set[int] = set()

    for ev in events:
        if ev.get("type") != "transcoder_latent":
            continue
        chunk = int(ev["chunk"])
        prompt = prompts.get(chunk, "")
        if want and want not in prompt:
            continue
        ts = ev.get("timestep") or []
        if not ts:
            continue
        layer = int(ev["layer"])
        t = round(float(ts[0]), 1)
        shape = ev.get("shape") or []
        width = int(shape[-1]) if shape else args.width
        by_layer_t[(layer, t)].append(100.0 * float(ev["l0_mean"]) / width)
        kept_chunks.add(chunk)

    if not by_layer_t:
        raise SystemExit(f"no transcoder_latent events matched task-contains={want!r}")

    layers = sorted({layer for layer, _ in by_layer_t})
    fig, axes = plt.subplots(6, 3, figsize=(14, 16), sharex=True, sharey=True, dpi=120)
    for ax, layer in zip(axes.ravel(), layers):
        times = sorted({t for item_layer, t in by_layer_t if item_layer == layer})
        values = [float(np.mean(by_layer_t[(layer, t)])) for t in times]
        ax.plot(times, values, marker="o")
        ax.set_title(f"TC_{layer}")
        ax.set_xlabel("t")
        ax.set_ylabel("mean token L0 (%)")
        ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.grid(True, alpha=0.3)
    for ax in axes.ravel()[len(layers) :]:
        ax.set_visible(False)
    title = "token-mean L0% vs denoise t"
    if want:
        title += f"  filter={want!r}"
    fig.suptitle(f"{title}  chunks={len(kept_chunks)}")
    fig.tight_layout()
    out = args.run / "transcoder_capture" / "l0_percent_vs_t_by_layer.png"
    if want:
        slug = "".join(ch if ch.isalnum() else "_" for ch in want)[:40]
        out = args.run / "transcoder_capture" / f"l0_percent_vs_t_by_layer_{slug}.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"chunks={sorted(kept_chunks)}")
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

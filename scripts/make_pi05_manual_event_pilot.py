#!/usr/bin/env python
"""Build a speed timeline and image contact sheet for manual event selection."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera", default="observation.images.image")
    parser.add_argument("--sample-every", type=int, default=15)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--event-window", default=None, help="Optional start:end range to highlight.")
    parser.add_argument("--slow-window", default=None, help="Optional start:end range for slow references.")
    parser.add_argument("--fast-window", default=None, help="Optional start:end range for fast references.")
    return parser.parse_args()


def _read_observations(path: Path, episode: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if int(row.get("episode_index", -1)) == episode:
                rows.append(row)
    rows.sort(key=lambda row: int(row["frame_index"]))
    if not rows:
        raise ValueError(f"No observations found for episode {episode}")
    return rows


def _read_images(path: Path, episode: int, camera: str) -> dict[int, Image.Image]:
    table = pq.read_table(path, columns=["episode_index", "frame_index", camera])
    table = table.filter(pc.equal(table["episode_index"], episode))
    images: dict[int, Image.Image] = {}
    for frame, encoded in zip(table["frame_index"].to_pylist(), table[camera].to_pylist(), strict=True):
        if not encoded or not encoded.get("bytes"):
            continue
        images[int(frame)] = Image.open(io.BytesIO(encoded["bytes"])).convert("RGB")
    if not images:
        raise ValueError(f"No {camera!r} images found for episode {episode}")
    return images


def _parse_window(raw: str | None) -> tuple[int, int] | None:
    if raw is None:
        return None
    start, end = raw.split(":", maxsplit=1)
    return int(start), int(end)


def _representative_frames(
    rows: list[dict[str, Any]], sample_every: int, *, frame_start: int, frame_end: int | None
) -> list[int]:
    last = int(rows[-1]["frame_index"]) if frame_end is None else min(frame_end, int(rows[-1]["frame_index"]))
    frames = list(range(frame_start, last + 1, sample_every))
    if frames[-1] != last:
        frames.append(last)
    return frames


def _save_contact_sheet(
    images: dict[int, Image.Image], frames: list[int], output: Path, *, episode: int
) -> None:
    columns = 4
    rows = (len(frames) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(12, rows * 3.0), squeeze=False)
    for axis, frame in zip(axes.flat, frames, strict=False):
        axis.imshow(images[frame])
        axis.set_title(f"frame {frame}", fontsize=11)
        axis.axis("off")
    for axis in axes.flat[len(frames) :]:
        axis.axis("off")
    fig.suptitle(f"Episode {episode}: sampled observations for manual event labeling", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)


def _save_speed_timeline(
    rows: list[dict[str, Any]],
    output: Path,
    *,
    episode: int,
    event_window: tuple[int, int] | None,
    slow_window: tuple[int, int] | None,
    fast_window: tuple[int, int] | None,
) -> None:
    frames = np.asarray([int(row["frame_index"]) for row in rows])
    speeds = np.asarray([float(row["speed_mean"]) for row in rows])
    phases = [("early", 0, 61), ("middle", 62, 147), ("late", 148, int(frames[-1]))]

    fig, axis = plt.subplots(figsize=(13, 4.8))
    for name, start, end in phases:
        axis.axvspan(start, end, alpha=0.07, label=name)
    if event_window is not None:
        axis.axvspan(*event_window, color="#7e57c2", alpha=0.10, label="manual event")
    if slow_window is not None:
        axis.axvspan(*slow_window, color="#ef6c00", alpha=0.16, label="slow references")
    if fast_window is not None:
        axis.axvspan(*fast_window, color="#2e7d32", alpha=0.16, label="fast references")
    axis.plot(frames, speeds, color="#1769aa", linewidth=2.0, label="predicted action speed")
    axis.scatter(frames, speeds, color="#1769aa", s=8, alpha=0.45)
    axis.set_title(f"Episode {episode} predicted translational action speed")
    axis.set_xlabel("Observation frame")
    axis.set_ylabel("Mean L2 speed over 50 action positions")
    axis.grid(axis="y", alpha=0.25)
    axis.set_xlim(int(frames[0]), int(frames[-1]))
    axis.legend(frameon=False, ncol=4, loc="upper left")
    fig.tight_layout()
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.sample_every <= 0:
        raise ValueError("--sample-every must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_observations(args.observations, args.episode)
    images = _read_images(args.parquet, args.episode, args.camera)
    sampled = [
        frame
        for frame in _representative_frames(
            rows,
            args.sample_every,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
        )
        if frame in images
    ]
    suffix = f"episode{args.episode}_{sampled[0]}_{sampled[-1]}"
    _save_contact_sheet(images, sampled, args.output_dir / f"{suffix}_contact_sheet.jpg", episode=args.episode)
    _save_speed_timeline(
        rows,
        args.output_dir / f"episode{args.episode}_speed_timeline.png",
        episode=args.episode,
        event_window=_parse_window(args.event_window),
        slow_window=_parse_window(args.slow_window),
        fast_window=_parse_window(args.fast_window),
    )
    print(json.dumps({"episode": args.episode, "observations": len(rows), "sampled_frames": sampled}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Rank behavior-linked transcoder features consistently across event bins."""

from __future__ import annotations

import argparse
import csv
import heapq
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--event-name")
    parser.add_argument("--top-candidates", type=int, default=100)
    parser.add_argument("--min-frequency", type=float, default=0.05)
    parser.add_argument("--max-frequency", type=float, default=0.99)
    return parser.parse_args()


def _push_candidate(
    heap: list[tuple[float, int, dict[str, Any]]],
    row: dict[str, Any],
    *,
    score: float,
    serial: int,
    limit: int,
) -> None:
    item = (score, serial, row)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif score > heap[0][0]:
        heapq.heapreplace(heap, item)


def _write_ranked(path: Path, heap: list[tuple[float, int, dict[str, Any]]], direction: str) -> None:
    rows = [item[2] for item in sorted(heap, key=lambda item: item[0], reverse=True)]
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row["direction"] = direction
    fieldnames = [
        "rank",
        "direction",
        "feature_key",
        "consistent_score",
        "mean_corr_speed",
        "event_labels",
        "corr_speed_by_event",
        "frequency_by_event",
        "mean_activation_by_event",
        "layer",
        "timestep",
        "feature",
        "layer_name",
        "timestep_key",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.top_candidates <= 0:
        raise ValueError("--top-candidates must be positive")
    if not 0.0 <= args.min_frequency <= args.max_frequency <= 1.0:
        raise ValueError("frequency bounds must satisfy 0 <= min <= max <= 1")

    behavior = torch.load(
        args.input_dir / "feature_behavior_event_stats.pt",
        map_location="cpu",
        weights_only=False,
    )
    activation = torch.load(
        args.input_dir / "feature_event_stats.pt",
        map_location="cpu",
        weights_only=False,
    )
    event_name = args.event_name or behavior.get("manual_event_name")
    if not event_name:
        raise ValueError("event name is missing; pass --event-name")
    event_labels = [f"{event_name}:q{i}" for i in range(1, int(behavior["manual_event_bins"]) + 1)]

    fast_heap: list[tuple[float, int, dict[str, Any]]] = []
    slow_heap: list[tuple[float, int, dict[str, Any]]] = []
    serial = 0
    for layer_name in behavior["layer_names"]:
        layer = int(behavior["layer_indices"][layer_name])
        for timestep_key, event_stores in sorted(behavior["stats"][layer_name].items()):
            missing = [label for label in event_labels if label not in event_stores]
            if missing:
                raise ValueError(f"missing event bins for {layer_name} at {timestep_key}: {missing}")
            correlations = torch.stack(
                [event_stores[label]["corr_behavior"].float() for label in event_labels]
            )
            activation_stores = activation["stats"][layer_name][timestep_key]
            frequencies = torch.stack(
                [activation_stores[label]["firing_frequency"].float() for label in event_labels]
            )
            means = torch.stack([activation_stores[label]["mean"].float() for label in event_labels])
            valid = (frequencies.min(dim=0).values >= args.min_frequency) & (
                frequencies.max(dim=0).values <= args.max_frequency
            )

            fast_scores = correlations.min(dim=0).values
            slow_scores = -correlations.max(dim=0).values
            for feature in torch.nonzero(valid, as_tuple=False).flatten().tolist():
                corr_values = correlations[:, feature].tolist()
                frequency_values = frequencies[:, feature].tolist()
                mean_values = means[:, feature].tolist()
                row = {
                    "feature_key": f"L{layer:02d}:tau{float(timestep_key):.4g}:F{feature}",
                    "mean_corr_speed": sum(corr_values) / len(corr_values),
                    "event_labels": ";".join(event_labels),
                    "corr_speed_by_event": ";".join(f"{value:.8g}" for value in corr_values),
                    "frequency_by_event": ";".join(f"{value:.8g}" for value in frequency_values),
                    "mean_activation_by_event": ";".join(f"{value:.8g}" for value in mean_values),
                    "layer": layer,
                    "timestep": float(timestep_key),
                    "feature": feature,
                    "layer_name": layer_name,
                    "timestep_key": timestep_key,
                }
                fast_score = float(fast_scores[feature])
                slow_score = float(slow_scores[feature])
                fast_row = dict(row, consistent_score=fast_score)
                slow_row = dict(row, consistent_score=slow_score)
                _push_candidate(
                    fast_heap,
                    fast_row,
                    score=fast_score,
                    serial=serial,
                    limit=args.top_candidates,
                )
                serial += 1
                _push_candidate(
                    slow_heap,
                    slow_row,
                    score=slow_score,
                    serial=serial,
                    limit=args.top_candidates,
                )
                serial += 1

    output_dir = args.output_dir or args.input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_ranked(output_dir / "fast_feature_candidates_consistent_by_event.csv", fast_heap, "fast")
    _write_ranked(output_dir / "slow_feature_candidates_consistent_by_event.csv", slow_heap, "slow")
    print(f"saved consistent event rankings to {output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Rank transcoder features by how selectively they respond to one object.

Reads ``latent_deltas.csv`` from a counterfactual probe run and answers the
question the probe was built for: which features move when the *task* object
changes appearance, and stay put when a visually matched *placebo* object
changes the same way.

Selectivity, not raw magnitude, is what matters. A feature that responds to
both objects is tracking generic pixel change. The candidates worth patching
are the ones that respond to the target and not the placebo.

One limitation is explicit in the output. The probe stores only the top-K
features per (layer, denoise step), so a feature missing from the placebo's
list is not known to be zero -- it is known to be below that cell's K-th
largest delta. That bound is reported as ``placebo_upper_bound`` rather than
being silently treated as zero.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_CSV_NAME = "latent_deltas.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="A counterfactual probe output directory.")
    parser.add_argument("--top", type=int, default=30, help="Candidate features to print.")
    parser.add_argument(
        "--min-cells",
        type=int,
        default=2,
        help="Require a feature to appear in at least this many (state, dose) cells, so one-off hits are dropped.",
    )
    return parser.parse_args()


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(result) else result


def load_rows(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("top_feature_ids", "top_feature_deltas"):
            raw = row.get(key) or "[]"
            try:
                row[key] = json.loads(raw)
            except json.JSONDecodeError:
                row[key] = []
        for key in ("l2_delta", "relative_l2", "dose", "max_abs_delta"):
            row[key] = _float(row.get(key))
        for key in ("denoise_step", "state_index", "features"):
            try:
                row[key] = int(float(row.get(key)))
            except (TypeError, ValueError):
                row[key] = None
    return rows


def peak_timestep_by_feature(
    rows: list[dict[str, Any]], *, condition: str, num_inference_steps: int
) -> dict[tuple[int, int], float]:
    """Flow time at which each feature responded most strongly.

    ``euler_integrate`` walks ``time = 1.0 + step * (-1/num_steps)``, so denoise
    step 0 is tau=1.0 and the last step is tau=1/num_steps. Recording tau lets a
    candidate be named in the same key format the circuit tracer consumes.
    """
    best: dict[tuple[int, int], tuple[float, float]] = {}
    for row in rows:
        if row.get("condition") != condition:
            continue
        step = row.get("denoise_step")
        if step is None:
            continue
        tau = 1.0 - (int(step) / max(1, num_inference_steps))
        for feature, delta in zip(row.get("top_feature_ids") or [], row.get("top_feature_deltas") or []):
            key = (_layer_index(row["layer"]), int(feature))
            magnitude = abs(float(delta))
            if key not in best or magnitude > best[key][1]:
                best[key] = (tau, magnitude)
    return {key: value[0] for key, value in best.items()}


def feature_key(layer_index: int, tau: float, feature: int) -> str:
    """The key format the circuit tracer expects, e.g. ``L11:tau0.7:F9970``."""
    return f"L{layer_index}:tau{tau:g}:F{feature}"


def layer_selectivity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean response per layer, target versus placebo."""
    totals: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = row.get("l2_delta")
        if value is None:
            continue
        totals[(row["layer"], row["condition"])].append(value)

    layers = sorted({layer for layer, _ in totals}, key=_layer_sort_key)
    table = []
    for layer in layers:
        target = totals.get((layer, "target"), [])
        placebo = totals.get((layer, "placebo"), [])
        null = totals.get((layer, "null"), [])
        target_mean = sum(target) / len(target) if target else 0.0
        placebo_mean = sum(placebo) / len(placebo) if placebo else 0.0
        table.append(
            {
                "layer": layer,
                "layer_index": _layer_index(layer),
                "target_mean_l2": target_mean,
                "placebo_mean_l2": placebo_mean,
                "null_mean_l2": sum(null) / len(null) if null else 0.0,
                "selectivity": (target_mean / placebo_mean) if placebo_mean > 0 else None,
            }
        )
    return table


def _layer_index(layer: str) -> int:
    parts = [part for part in layer.split(".") if part.isdigit()]
    return int(parts[-1]) if parts else -1


def _layer_sort_key(layer: str) -> tuple[int, str]:
    return (_layer_index(layer), layer)


def candidate_features(rows: list[dict[str, Any]], *, min_cells: int) -> list[dict[str, Any]]:
    """Per (layer, feature) response to the target, contrasted with the placebo."""
    target_hits: dict[tuple[str, int], list[float]] = defaultdict(list)
    placebo_hits: dict[tuple[str, int], list[float]] = defaultdict(list)
    # Smallest delta the probe recorded in each cell: the detection floor for
    # anything that did not make that cell's top-K list.
    placebo_floor: dict[str, list[float]] = defaultdict(list)
    target_cells: set[tuple[str, int, float]] = set()

    for row in rows:
        ids = row.get("top_feature_ids") or []
        deltas = row.get("top_feature_deltas") or []
        if not ids:
            continue
        magnitudes = [abs(float(value)) for value in deltas]
        if row["condition"] == "target":
            target_cells.add((row["layer"], row["state_index"], row["dose"]))
            for feature, magnitude in zip(ids, magnitudes):
                target_hits[(row["layer"], int(feature))].append(magnitude)
        elif row["condition"] == "placebo":
            placebo_floor[row["layer"]].append(min(magnitudes) if magnitudes else 0.0)
            for feature, magnitude in zip(ids, magnitudes):
                placebo_hits[(row["layer"], int(feature))].append(magnitude)

    # A top-K list is padded with zero-delta entries when fewer than K features
    # actually moved. Those carry no signal and would otherwise score as
    # infinitely selective, so the smallest positive response seen anywhere sets
    # the resolution floor used for unbounded cases.
    positives = [value for values in placebo_hits.values() for value in values if value > 0]
    positives += [value for values in target_hits.values() for value in values if value > 0]
    resolution = min(positives) if positives else 1e-9

    results = []
    for (layer, feature), magnitudes in target_hits.items():
        if len(magnitudes) < min_cells:
            continue
        target_mean = sum(magnitudes) / len(magnitudes)
        if target_mean <= 0.0:
            continue  # padding, not a response
        placebo = placebo_hits.get((layer, feature), [])
        floors = [value for value in placebo_floor.get(layer, []) if value > 0]
        bound = min(floors) if floors else 0.0
        if placebo:
            placebo_mean = sum(placebo) / len(placebo)
            measured = True
        else:
            # Absent from the placebo's top-K, so its response is at most that
            # cell's smallest recorded delta.
            placebo_mean = bound
            measured = False

        # Every feature gets a finite, comparable score. When the placebo
        # response is unmeasurably small the score is a lower bound, so a
        # strongly-responding selective feature still outranks a weak one --
        # which ranking purely by "unbounded first" would get backwards.
        denominator = max(placebo_mean, resolution)
        results.append(
            {
                "layer": layer,
                "layer_index": _layer_index(layer),
                "feature": feature,
                "target_mean_abs_delta": target_mean,
                "target_cells": len(magnitudes),
                "placebo_response": placebo_mean,
                "placebo_measured": measured,
                "placebo_upper_bound": None if measured else bound,
                "selectivity": target_mean / denominator,
                "tau": None,
                "feature_key": None,
                "selectivity_is_lower_bound": not measured or placebo_mean <= 0.0,
            }
        )

    results.sort(key=lambda row: (-row["selectivity"], -row["target_mean_abs_delta"]))
    return results


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    csv_path = args.run_dir if args.run_dir.is_file() else args.run_dir / DEFAULT_CSV_NAME
    if not csv_path.exists():
        raise SystemExit(f"No {DEFAULT_CSV_NAME} at {csv_path}")
    rows = load_rows(csv_path)
    if not rows:
        raise SystemExit(f"{csv_path} is empty")

    steps = 10
    summary_path = csv_path.parent / "counterfactual_summary.json"
    if summary_path.exists():
        try:
            steps = int(json.loads(summary_path.read_text())["config"]["num_inference_steps"])
        except (KeyError, ValueError, TypeError):
            pass
    peak_tau = peak_timestep_by_feature(rows, condition="target", num_inference_steps=steps)

    conditions = sorted({row["condition"] for row in rows})
    print(f"loaded {len(rows)} rows from {csv_path} (conditions: {', '.join(conditions)})\n")

    layers = layer_selectivity(rows)
    print("per-layer response (mean L2 over denoise steps, states and doses)")
    print(f"  {'layer':>5} {'target':>10} {'placebo':>10} {'null':>8} {'selectivity':>12}")
    for row in layers:
        sel = "n/a" if row["selectivity"] is None else f"{row['selectivity']:.2f}x"
        print(
            f"  {row['layer_index']:>5} {row['target_mean_l2']:>10.4f} "
            f"{row['placebo_mean_l2']:>10.4f} {row['null_mean_l2']:>8.4f} {sel:>12}"
        )

    ranked = max(layers, key=lambda r: r["selectivity"] or 0.0) if layers else None
    if ranked and ranked["selectivity"]:
        print(f"\nmost selective layer: {ranked['layer_index']} at {ranked['selectivity']:.2f}x")

    features = candidate_features(rows, min_cells=args.min_cells)
    for row in features:
        tau = peak_tau.get((row["layer_index"], row["feature"]))
        row["tau"] = tau
        row["feature_key"] = None if tau is None else feature_key(row["layer_index"], tau, row["feature"])
    print(
        f"\ncandidate features (>= {args.min_cells} cells, ranked by selectivity) "
        f"- {len(features)} found, showing {min(args.top, len(features))}"
    )
    print(f"  {'layer':>5} {'feature':>8} {'target':>10} {'placebo':>10} {'sel':>8} {'cells':>6}  note")
    for row in features[: args.top]:
        sel = f"{row['selectivity']:.2f}x"
        if row["selectivity_is_lower_bound"]:
            sel = ">=" + sel
        note = "" if row["placebo_measured"] else "placebo below top-K"
        print(
            f"  {row['layer_index']:>5} {row['feature']:>8} {row['target_mean_abs_delta']:>10.4f} "
            f"{row['placebo_response']:>10.4f} {sel:>8} {row['target_cells']:>6}  {note}"
        )

    out_dir = csv_path.parent
    write_csv(out_dir / "layer_selectivity.csv", layers)
    write_csv(out_dir / "candidate_features.csv", features)
    print(f"\nWrote {out_dir / 'layer_selectivity.csv'} and {out_dir / 'candidate_features.csv'}")

    nominated = [row for row in features if row.get("feature_key")][:5]
    if nominated:
        print(
            "\nNominated targets for circuit tracing. These are selected by controlled "
            "counterfactual selectivity rather than by activation statistics, so they are a "
            "different nomination route into the same tracer:"
        )
        for row in nominated:
            print(f"  {row['feature_key']:<22} selectivity {row['selectivity']:.1f}x  "
                  f"cells {row['target_cells']}")
        print("\n  python scripts/trace_pi05_transcoder_circuit.py \\")
        print(f"    --target {nominated[0]['feature_key']} \\")
        print("    --checkpoint <transcoder.pt> --feature-dir <feature discovery dir>")
    print(
        "\nA feature marked 'placebo below top-K' was not recorded for the placebo, so its "
        "placebo response is an upper bound, not zero. Its true selectivity is at least the "
        "figure shown."
    )


if __name__ == "__main__":
    main()

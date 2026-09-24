#!/usr/bin/env python
"""Rank transcoder features by how selectively they respond to one object.

Reads ``latent_deltas.csv`` from a counterfactual probe run and answers the
question the probe was built for: which features move when the *task* object
changes appearance, and stay put when a visually matched *placebo* object
changes the same way.

Selectivity, not raw magnitude, is what matters. A feature that responds to
both objects is tracking generic pixel change. The candidates worth patching
are the ones that respond to the target and not the placebo.

When the probe wrote its full delta store (``latents/``), the placebo response
of every candidate is read from it exactly, at the candidate's own flow time.
Without the store only the top-K per (layer, denoise step) is available, and a
feature missing from the placebo's list is then known only to be below that
cell's K-th largest delta; that bound is reported as ``placebo_upper_bound``
rather than being silently treated as zero.

Nomination for circuit tracing is decided here and written to
``nominated_targets.json`` so that every downstream consumer applies the same
rule: reproducible across cells, standing out within its layer, not responsive
to the placebo, and deep enough to have parents to trace.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys

import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pi05_mi.counterfactual_store import DeltaStore  # noqa: E402

DEFAULT_CSV_NAME = "latent_deltas.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="A counterfactual probe output directory.")
    parser.add_argument("--top", type=int, default=30, help="Candidate features to print.")
    parser.add_argument(
        "--min-consistency",
        type=float,
        default=0.25,
        help=(
            "Fraction of a layer's measurement cells a feature must appear in to be nominated "
            "for tracing. Selectivity alone rewards features that fired once with a large delta."
        ),
    )
    parser.add_argument(
        "--max-placebo-z",
        type=float,
        default=1.0,
        help="Reject a candidate whose placebo response is this many layer SDs above its layer mean.",
    )
    parser.add_argument(
        "--min-cells",
        type=int,
        default=2,
        help="Require a feature to appear in at least this many (state, dose) cells, so one-off hits are dropped.",
    )
    parser.add_argument(
        "--nomination-prompt",
        default="task",
        help=(
            "Prompt condition whose cells decide nomination (default: the task prompt, the H1 condition). "
            "Pass 'all' to pool every prompt."
        ),
    )
    parser.add_argument(
        "--min-trace-layer",
        type=int,
        default=4,
        help=(
            "Do not nominate features below this action-expert layer. A layer-2 target has almost "
            "nothing upstream, so its trace fans out sideways instead of finding a circuit."
        ),
    )
    parser.add_argument(
        "--min-selectivity",
        type=float,
        default=2.0,
        help=(
            "With the full delta store, require the exact target/placebo response ratio at the "
            "candidate's flow time to be at least this. Without the store --max-placebo-z applies."
        ),
    )
    return parser.parse_args()


def _prompt_matches(row: dict[str, Any], prompt: str | None) -> bool:
    """Rows from runs without a prompt column always match."""
    if prompt is None:
        return True
    value = row.get("prompt")
    return value in (None, "", prompt)


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
        for key in ("denoise_step", "state_index", "noise_index", "features"):
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


def layer_selectivity(rows: list[dict[str, Any]], *, prompt: str | None = None) -> list[dict[str, Any]]:
    """Mean layer response D_l (Eq. layer-response), target versus placebo, for one prompt."""
    totals: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = row.get("l2_delta")
        if value is None or not _prompt_matches(row, prompt):
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
                "prompt": prompt or "all",
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


def candidate_features(
    rows: list[dict[str, Any]], *, min_cells: int, prompt: str | None = None
) -> list[dict[str, Any]]:
    """Per (layer, feature) response to the target, contrasted with the placebo."""
    rows = [row for row in rows if _prompt_matches(row, prompt)]
    target_hits: dict[tuple[str, int], list[float]] = defaultdict(list)
    placebo_hits: dict[tuple[str, int], list[float]] = defaultdict(list)
    # Smallest delta the probe recorded in each cell: the detection floor for
    # anything that did not make that cell's top-K list.
    placebo_floor: dict[str, list[float]] = defaultdict(list)
    # How many measurement cells each layer actually had, so "appeared in 2" can
    # be read against "out of 80" rather than taken at face value.
    cells_per_layer: dict[str, set[tuple[Any, ...]]] = defaultdict(set)

    for row in rows:
        ids = row.get("top_feature_ids") or []
        deltas = row.get("top_feature_deltas") or []
        if not ids:
            continue
        magnitudes = [abs(float(value)) for value in deltas]
        if row["condition"] == "target":
            cells_per_layer[row["layer"]].add(
                (row["state_index"], row.get("noise_index"), row["dose"], row.get("prompt"), row["denoise_step"])
            )
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

    # Per-layer response statistics, used to put features from different layers
    # on one scale. Raw selectivity cannot do this: its denominator is the
    # layer's own placebo floor, and those floors differ by an order of
    # magnitude across depth, so an identical absolute response scores several
    # times higher in a layer that happens to have a smaller floor.
    layer_target_stats: dict[str, tuple[float, float]] = {}
    layer_placebo_stats: dict[str, tuple[float, float]] = {}
    for store, stats in ((target_hits, layer_target_stats), (placebo_hits, layer_placebo_stats)):
        by_layer: dict[str, list[float]] = defaultdict(list)
        for (layer, _feature), values in store.items():
            by_layer[layer].append(float(np.mean(values)))
        for layer, values in by_layer.items():
            array = np.asarray(values, dtype=np.float64)
            spread = float(array.std(ddof=1)) if array.size > 1 else 0.0
            stats[layer] = (float(array.mean()), spread)

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

        # Layer-standardised response: how far above this layer's own typical
        # feature the response sits, in that layer's standard deviations. Free
        # of the floor artefact, so it is comparable across depth.
        t_mean, t_std = layer_target_stats.get(layer, (0.0, 0.0))
        target_z = (target_mean - t_mean) / t_std if t_std > 0 else 0.0
        p_mean, p_std = layer_placebo_stats.get(layer, (0.0, 0.0))
        placebo_z = (placebo_mean - p_mean) / p_std if (measured and p_std > 0) else None

        results.append(
            {
                "layer": layer,
                "layer_index": _layer_index(layer),
                "feature": feature,
                "target_mean_abs_delta": target_mean,
                "target_z": target_z,
                "placebo_z": placebo_z,
                "target_cells": len(magnitudes),
                "layer_cells": len(cells_per_layer.get(layer, ())),
                "consistency": (
                    len(magnitudes) / len(cells_per_layer[layer]) if cells_per_layer.get(layer) else 0.0
                ),
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
    nomination_prompt = None if args.nomination_prompt == "all" else args.nomination_prompt
    prompts_present = {row.get("prompt") for row in rows} - {None, ""}
    if nomination_prompt is not None and prompts_present and nomination_prompt not in prompts_present:
        print(
            f"no rows carry prompt {nomination_prompt!r} (found {sorted(prompts_present)}); pooling all prompts",
            flush=True,
        )
        nomination_prompt = None
    peak_tau = peak_timestep_by_feature(
        [row for row in rows if _prompt_matches(row, nomination_prompt)],
        condition="target",
        num_inference_steps=steps,
    )

    conditions = sorted({row["condition"] for row in rows})
    print(f"loaded {len(rows)} rows from {csv_path} (conditions: {', '.join(conditions)})\n")

    layers = layer_selectivity(rows, prompt=nomination_prompt)
    print(f"per-layer response D_l (mean L2 over denoise steps, states, noise draws and doses; prompt={nomination_prompt or 'all'})")
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

    features = candidate_features(rows, min_cells=args.min_cells, prompt=nomination_prompt)
    for row in features:
        tau = peak_tau.get((row["layer_index"], row["feature"]))
        row["tau"] = tau
        row["feature_key"] = None if tau is None else feature_key(row["layer_index"], tau, row["feature"])

    # Exact per-feature responses from the full delta store, at each candidate's
    # own flow time. This replaces the top-K bound on the placebo with a number.
    store_root = csv_path.parent / "latents"
    exact_available = DeltaStore.exists(store_root)
    if exact_available:
        store = DeltaStore.open(store_root)
        mean_target, n_target_cells = store.mean_abs_delta(condition="target", prompt=nomination_prompt)
        has_placebo = "placebo" in store.conditions()
        mean_placebo, n_placebo_cells = (
            store.mean_abs_delta(condition="placebo", prompt=nomination_prompt) if has_placebo else (None, 0)
        )
        print(
            f"\nexact responses from {store_root} ({n_target_cells} target cells, {n_placebo_cells} placebo cells)"
        )
        for row in features:
            position = store.layer_position(row["layer_index"])
            if position is None or row["tau"] is None:
                continue
            step = store.step_for_tau(row["tau"])
            t_exact = float(mean_target[position, step, row["feature"]])
            row["target_exact_mean_abs_delta"] = t_exact
            if mean_placebo is not None:
                p_exact = float(mean_placebo[position, step, row["feature"]])
                row["placebo_exact_mean_abs_delta"] = p_exact
                row["placebo_exact_zero"] = p_exact == 0.0
                row["selectivity_exact"] = (t_exact / p_exact) if p_exact > 0 else None
    print(
        f"\ncandidate features (>= {args.min_cells} cells, ranked by selectivity) "
        f"- {len(features)} found, showing {min(args.top, len(features))}"
    )
    print(f"  {'layer':>5} {'feature':>8} {'target':>10} {'placebo':>10} {'sel':>8} {'cells':>9}  note")
    for row in features[: args.top]:
        sel = f"{row['selectivity']:.2f}x"
        if row["selectivity_is_lower_bound"]:
            sel = ">=" + sel
        note = "" if row["placebo_measured"] else "placebo below top-K"
        cells = f"{row['target_cells']}/{row['layer_cells']}"
        print(
            f"  {row['layer_index']:>5} {row['feature']:>8} {row['target_mean_abs_delta']:>10.4f} "
            f"{row['placebo_response']:>10.4f} {sel:>8} {cells:>9}  {note}"
        )

    out_dir = csv_path.parent
    all_layers = [
        entry
        for prompt in (sorted(prompts_present) if prompts_present else [None])
        for entry in layer_selectivity(rows, prompt=prompt)
    ]
    write_csv(out_dir / "layer_selectivity.csv", all_layers)
    write_csv(out_dir / "candidate_features.csv", features)
    print(f"\nWrote {out_dir / 'layer_selectivity.csv'} and {out_dir / 'candidate_features.csv'}")

    # Nomination. Each criterion rules out a specific way a nominee could be a
    # bad trace target:
    #   consistency  -- a feature seen in 2 of 80 cells can top the ranking on
    #                   one large delta; tracing it chases a fluke.
    #   placebo      -- with the store, the exact placebo response must be at
    #                   most 1/min_selectivity of the target's; otherwise the
    #                   layer-standardised placebo response must not be elevated.
    #   depth        -- a shallow target has nothing upstream to find.
    # Ranking is by the layer-standardised response so depths compete fairly.
    def passes_placebo(row: dict[str, Any]) -> bool:
        if exact_available and "placebo_exact_mean_abs_delta" in row:
            t_exact = row.get("target_exact_mean_abs_delta") or 0.0
            return row["placebo_exact_mean_abs_delta"] * args.min_selectivity <= t_exact and t_exact > 0
        return row["placebo_z"] is None or row["placebo_z"] < args.max_placebo_z

    rejected: dict[str, int] = defaultdict(int)
    eligible = []
    for row in features:
        if not row.get("feature_key"):
            rejected["no flow time"] += 1
        elif row["consistency"] < args.min_consistency:
            rejected["inconsistent"] += 1
        elif row["layer_index"] < args.min_trace_layer:
            rejected["too shallow"] += 1
        elif not passes_placebo(row):
            rejected["placebo-responsive"] += 1
        else:
            eligible.append(row)
    eligible.sort(key=lambda row: -row["target_z"])
    criteria = {
        "prompt": nomination_prompt or "all",
        "min_consistency": args.min_consistency,
        "min_trace_layer": args.min_trace_layer,
        "placebo_rule": (
            f"exact placebo response <= target / {args.min_selectivity:g} at the candidate's flow time"
            if exact_available
            else f"layer-standardised placebo z < {args.max_placebo_z:g} or below top-K"
        ),
        "ranking": "layer-standardised target response z, descending",
        "rejected": dict(rejected),
        "candidates_considered": len(features),
    }
    if not eligible:
        best = max((r["consistency"] for r in features), default=0.0)
        print(
            f"\nNo feature met every nomination criterion (rejections: {dict(rejected)}; best "
            f"consistency {best:.0%}). Widen the probe (--states, --noise-samples, --dose) or relax a "
            "criterion deliberately and say so."
        )
    nominated = eligible[:5]
    if nominated:
        print(
            "\nNominated targets for circuit tracing. Criteria: fired in >= "
            f"{args.min_consistency:.0%} of the layer's cells; layer >= {args.min_trace_layer}; "
            f"{criteria['placebo_rule']}; ranked by layer-standardised response z. Rejected: {dict(rejected)}."
        )
        for row in nominated:
            exact = row.get("selectivity_exact")
            exact_note = (
                ""
                if not exact_available
                else (f"  exact sel {exact:.1f}x" if exact is not None else "  exact placebo 0 (unbounded)")
            )
            print(f"  {row['feature_key']:<22} z={row['target_z']:+.2f}  "
                  f"top-K sel {row['selectivity']:.1f}x{exact_note}  "
                  f"cells {row['target_cells']}/{row['layer_cells']} ({row['consistency']:.0%})")
        print("\n  python scripts/trace_pi05_transcoder_circuit.py \\")
        print(f"    --target {nominated[0]['feature_key']} \\")
        print("    --checkpoint <transcoder.pt> --feature-dir <feature discovery dir>")
    (out_dir / "nominated_targets.json").write_text(
        json.dumps({"criteria": criteria, "nominated": nominated}, indent=2, default=str), encoding="utf-8"
    )
    print(f"Wrote {out_dir / 'nominated_targets.json'}")
    if not exact_available:
        print(
            "\nA feature marked 'placebo below top-K' was not recorded for the placebo, so its "
            "placebo response is an upper bound, not zero. Its true selectivity is at least the "
            "figure shown. Rerun the probe with the full delta store to make it exact."
        )


if __name__ == "__main__":
    main()

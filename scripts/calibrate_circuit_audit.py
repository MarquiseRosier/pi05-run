#!/usr/bin/env python
"""Establish what the circuit audit can detect, before trusting what it did not.

The audit compares a traced circuit against matched random features and reports
a selectivity enrichment. On the reported run that enrichment was flat, and H3
was recorded as falsified. That reading has a hole in it: the audit has a
negative control and no positive one, so a flat result is ambiguous between
"this circuit carries nothing" and "this test cannot see what it is looking
for".

This script closes the hole by auditing circuits whose content is known by
construction. From the probe's own delta store it builds sets of the requested
size at the target's layers and flow time, mixing a fraction ``c`` of features
with the highest measured object selectivity into a remainder drawn at random
from the features the scene exercises. At ``c = 1`` the set is maximally
content-carrying and the audit must rate it highly; at ``c = 0`` it is a random
set and the audit must not. Sweeping ``c`` in between gives the detection curve
and, from it, the smallest contamination the audit reliably calls.

That number is what makes a negative interpretable. Instead of "the traced
circuit was not enriched" the claim becomes "the traced circuit is less than
``c*`` content-carrying, where the audit detects ``c*`` at 80 per cent". It also
catches the failure that would otherwise be invisible: if even ``c = 1`` is not
detected, the audit is broken and every verdict it has produced is void.

Selection uses the target condition's own measurements, so this is a
calibration of the instrument and not a hypothesis test. It says what the audit
can see, not what the model contains.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pi05_mi.counterfactual_store import DeltaStore  # noqa: E402
from validate_circuit_with_counterfactual import monte_carlo_p, selectivity_stats  # noqa: E402

DEFAULT_LEVELS = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)
DETECTION_RATE = 0.8


def feature_pool(
    store: DeltaStore, *, max_layer: int, step: int, prompt: str | None
) -> dict[str, Any]:
    """Exercised features below ``max_layer`` at one flow time, with their selectivity.

    Returns flat arrays indexed alike: the (layer position, feature) of each
    candidate, its mean target response, its mean placebo response and their
    ratio. Features the scene does not exercise are excluded, exactly as the
    audit excludes them.
    """
    mean_target, n_target = store.mean_abs_delta(condition="target", prompt=prompt)
    mean_placebo, n_placebo = store.mean_abs_delta(condition="placebo", prompt=prompt)
    exercised = store.exercised_mask(condition="target", prompt=prompt)

    positions, features, t_vals, p_vals = [], [], [], []
    for position, layer_index in enumerate(store.layer_indices()):
        if layer_index >= max_layer:
            continue
        live = np.flatnonzero(exercised[position, step])
        if live.size == 0:
            continue
        positions.append(np.full(live.size, position, dtype=np.int64))
        features.append(live)
        t_vals.append(mean_target[position, step, live])
        p_vals.append(mean_placebo[position, step, live])
    if not positions:
        return {"n": 0}
    t = np.concatenate(t_vals).astype(np.float64)
    p = np.concatenate(p_vals).astype(np.float64)
    # A placebo response of exactly zero is unbounded selectivity; rank it by
    # target magnitude instead of dividing, using the smallest positive placebo
    # response as the resolution floor.
    floor = float(p[p > 0].min()) if (p > 0).any() else 1e-9
    selectivity = t / np.maximum(p, floor)
    return {
        "n": int(t.size),
        "position": np.concatenate(positions),
        "feature": np.concatenate(features),
        "target": t,
        "placebo": p,
        "selectivity": selectivity,
        "cells": {"target": n_target, "placebo": n_placebo},
    }


def audit_set(
    pool: dict[str, Any], chosen: np.ndarray, *, step: int, store: DeltaStore,
    responses: dict[str, np.ndarray], exercised: np.ndarray, draws: int, rng: np.random.Generator,
) -> dict[str, Any] | None:
    """Run the audit's own statistic on one feature set."""
    positions = pool["position"][chosen]
    features = pool["feature"][chosen]

    circuit = {c: float(np.mean(responses[c][positions, step, features])) for c in responses}
    # Matched null: per member, a random exercised feature at the same layer and
    # flow time, the same draw evaluated under every condition.
    control: dict[str, np.ndarray] = {c: np.zeros(draws) for c in responses}
    for position in np.unique(positions):
        count = int((positions == position).sum())
        live = np.flatnonzero(exercised[position, step])
        if live.size == 0:
            return None
        picks = live[rng.integers(0, live.size, size=(draws, count))]
        for c, array in responses.items():
            control[c] += array[position, step, picks].sum(axis=1)
    for c in control:
        control[c] /= len(chosen)

    if circuit["placebo"] <= 0.0 and circuit["target"] > 0.0:
        # A set with no placebo response at all is infinitely selective. The
        # ratio is undefined, not absent: dropping it would silently discard the
        # most extreme positive control, which is exactly the level that decides
        # whether the audit can see anything.
        return {
            "selectivity": float("inf"),
            "random_selectivity": float(control["target"].mean() / control["placebo"].mean())
            if control["placebo"].mean() > 0
            else float("nan"),
            "enrichment": float("inf"),
            "monte_carlo_p": 1.0 / (draws + 1),
            "unbounded": True,
            "response_enrichment": (
                circuit["target"] / float(control["target"].mean()) if control["target"].mean() > 0 else float("nan")
            ),
        }
    stats = selectivity_stats(circuit["target"], circuit["placebo"], control["target"], control["placebo"])
    if stats is None:
        return None
    return {
        "selectivity": stats["circuit"],
        "random_selectivity": stats["random_mean"],
        "enrichment": stats["enrichment"],
        "monte_carlo_p": stats["monte_carlo_p"],
        "unbounded": False,
        "response_enrichment": (
            circuit["target"] / float(control["target"].mean()) if control["target"].mean() > 0 else float("nan")
        ),
    }


def calibrate(
    store: DeltaStore, *, max_layer: int, step: int, size: int, levels, repeats: int,
    draws: int, prompt: str | None, alpha: float, seed: int,
) -> dict[str, Any]:
    pool = feature_pool(store, max_layer=max_layer, step=step, prompt=prompt)
    if pool["n"] < size * 2:
        raise SystemExit(
            f"Only {pool.get('n', 0)} exercised features below layer {max_layer} at step {step}; "
            f"need at least {size * 2} to build sets of {size}."
        )
    responses = {c: store.mean_abs_delta(condition=c, prompt=prompt)[0] for c in ("target", "placebo")}
    exercised = store.exercised_mask(condition="target", prompt=prompt)
    order = np.argsort(-pool["selectivity"])
    rng = np.random.default_rng(seed)

    rows = []
    for level in levels:
        n_selective = int(round(level * size))
        detected, enrichments, unbounded = 0, [], 0
        for _ in range(repeats):
            # Draw the selective part from the top of the selectivity ranking,
            # widened so repeats differ rather than returning the same set.
            head = order[: max(n_selective * 4, n_selective + 1)]
            picked_selective = rng.choice(head, size=n_selective, replace=False) if n_selective else np.array([], int)
            remainder = np.setdiff1d(np.arange(pool["n"]), picked_selective, assume_unique=False)
            picked_random = rng.choice(remainder, size=size - n_selective, replace=False)
            chosen = np.concatenate([picked_selective, picked_random]).astype(int)
            result = audit_set(pool, chosen, step=step, store=store, responses=responses,
                               exercised=exercised, draws=draws, rng=rng)
            if result is None:
                continue
            enrichments.append(result["enrichment"])
            unbounded += int(bool(result.get("unbounded")))
            if result["enrichment"] > 1.0 and result["monte_carlo_p"] is not None and result["monte_carlo_p"] < alpha:
                detected += 1
        if not enrichments:
            continue
        arr = np.asarray(enrichments, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        rows.append(
            {
                "contamination": float(level),
                "selective_members": n_selective,
                "size": size,
                "repeats": len(enrichments),
                # Medians over the finite repeats, with the unbounded ones
                # counted rather than folded in as a number they are not.
                "enrichment_median": float(np.median(finite)) if finite.size else None,
                "enrichment_min": float(finite.min()) if finite.size else None,
                "enrichment_max": float(finite.max()) if finite.size else None,
                "unbounded_repeats": unbounded,
                "detection_rate": detected / len(enrichments),
            }
        )

    detectable = [r for r in rows if r["detection_rate"] >= DETECTION_RATE]
    smallest = min((r["contamination"] for r in detectable), default=None)
    at_zero = next((r for r in rows if r["contamination"] == 0.0), None)
    at_one = next((r for r in rows if r["contamination"] == 1.0), None)

    if at_one is not None and at_one["detection_rate"] < DETECTION_RATE:
        verdict = (
            f"the audit is blind: even a fully content-carrying set of {size} is detected only "
            f"{at_one['detection_rate']:.0%} of the time, so no negative verdict it has produced is "
            "interpretable"
        )
    elif at_zero is not None and at_zero["detection_rate"] > 1 - DETECTION_RATE:
        verdict = (
            f"the audit over-calls: a purely random set is detected {at_zero['detection_rate']:.0%} of "
            "the time, so its positives are not trustworthy"
        )
    elif smallest is None:
        verdict = "not calibrated: no contamination level reached the detection rate"
    else:
        verdict = (
            f"the audit detects a set that is at least {smallest:.0%} content-carrying, at "
            f"{DETECTION_RATE:.0%} of repeats; a circuit it reads as flat is below that"
        )

    return {
        "summary": {
            "max_layer": max_layer,
            "step": step,
            "flow_time": store.tau_for_step(step),
            "size": size,
            "prompt": prompt or "all",
            "pool_features": pool["n"],
            "cells": pool["cells"],
            "draws": draws,
            "repeats": repeats,
            "alpha": alpha,
            "detection_rate_threshold": DETECTION_RATE,
            "smallest_detectable_contamination": smallest,
            "decision_rule": (
                f"a contamination level counts as detectable when at least {DETECTION_RATE:.0%} of "
                f"repeats give selectivity enrichment above 1 at p < {alpha}"
            ),
            "verdict": verdict,
        },
        "levels": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("probe_run", type=Path, help="A probe run directory holding latents/.")
    parser.add_argument("--target-layer", type=int, required=True, help="Members are drawn below this layer.")
    parser.add_argument("--tau", type=float, default=None, help="Flow time; defaults to the store's last step.")
    parser.add_argument("--size", type=int, default=16, help="Set size. Match the circuit being audited.")
    parser.add_argument("--levels", default=",".join(str(v) for v in DEFAULT_LEVELS))
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--prompt", default="task")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    root = args.probe_run / "latents"
    if not DeltaStore.exists(root):
        raise SystemExit(f"No delta store at {root}; rerun the probe without --no-full-deltas.")
    store = DeltaStore.open(root)
    step = store.num_steps - 1 if args.tau is None else store.step_for_tau(args.tau)
    levels = [float(v) for v in args.levels.split(",") if v.strip()]
    prompt = None if args.prompt == "all" else args.prompt

    report = calibrate(
        store, max_layer=args.target_layer, step=step, size=args.size, levels=levels,
        repeats=args.repeats, draws=args.draws, prompt=prompt, alpha=args.alpha, seed=args.seed,
    )
    s = report["summary"]
    print(
        f"audit calibration: sets of {s['size']} below layer {s['max_layer']} at tau={s['flow_time']:.2g}, "
        f"drawn from {s['pool_features']} exercised features\n"
    )
    print(f"{'content':>8} {'members':>8} {'sel enrich (median)':>20} {'min':>7} {'max':>7} {'detected':>9}")
    for row in report["levels"]:
        def g(value, spec=".2f"):
            return "  inf" if value is None else format(value, spec)

        note = f"  ({row['unbounded_repeats']} unbounded)" if row.get("unbounded_repeats") else ""
        print(
            f"{row['contamination']:>7.0%} {row['selective_members']:>8} "
            f"{g(row['enrichment_median'], '.3f'):>20} {g(row['enrichment_min']):>7} "
            f"{g(row['enrichment_max']):>7} {row['detection_rate']:>8.0%}{note}"
        )
    print(f"\nrule: {s['decision_rule']}")
    print(f"verdict: {s['verdict']}")

    out_dir = args.output_dir or (args.probe_run / "audit_calibration")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audit_calibration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["levels"]:
        with (out_dir / "audit_calibration.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(report["levels"][0].keys()))
            writer.writeheader()
            writer.writerows(report["levels"])
    print(f"\nWrote {out_dir / 'audit_calibration.json'} and audit_calibration.csv")


if __name__ == "__main__":
    main()

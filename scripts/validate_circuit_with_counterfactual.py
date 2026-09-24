#!/usr/bin/env python
"""Test a traced circuit against a known, controlled cause.

Circuit tracing produces a hypothesis: these upstream features feed this
target. Ablation shows the circuit matters *more than random*, but it cannot
say the circuit is the right circuit for any particular thing, because nothing
in that setup fixes what the circuit is supposed to be carrying.

The counterfactual probe does fix it. It changes exactly one property of one
object in an otherwise bit-identical scene, with a pixel-matched placebo object
as control, so the cause is known by construction. That makes it an independent
yardstick for the trace:

    if the traced parents genuinely carry the target's information about this
    object, they should respond to perturbing *that* object and not to
    perturbing the matched control.

So this script scores the circuit's nodes on the probe's measurements and
compares them against matched random features drawn from the same layers. Two
things can come out of it. The circuit is enriched, which corroborates the
attribution on a cause we controlled rather than inferred. Or it is not, which
says the traced edges are not carrying this particular signal -- and that is
worth knowing before spending a patching run on them.

Read the coverage figure first. The probe records only the top-K features per
layer and denoise step, so a circuit node it never recorded has an unknown
response, not a zero one. Those nodes are reported separately rather than
folded in as zeros.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace_dir", type=Path, help="Circuit trace output directory (holds graph.json).")
    parser.add_argument("probe_run", type=Path, help="Counterfactual probe run directory.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--random-draws", type=int, default=200, help="Matched random control samples.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_circuit(trace_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = trace_dir / "graph.json"
    if not path.exists():
        raise SystemExit(f"No graph.json in {trace_dir}; run trace_pi05_transcoder_circuit.py first.")
    graph = json.loads(path.read_text())
    return graph.get("nodes", []), graph.get("config", {})


def load_probe_responses(run_dir: Path) -> dict[str, dict[tuple[int, int], float]]:
    """Per-condition mean |delta| for every (layer, feature) the probe recorded."""
    path = run_dir / "latent_deltas.csv"
    if not path.exists():
        raise SystemExit(f"No latent_deltas.csv in {run_dir}")

    collected: dict[str, dict[tuple[int, int], list[float]]] = defaultdict(lambda: defaultdict(list))
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            condition = row.get("condition")
            if not condition:
                continue
            layer = _layer_index(row.get("layer", ""))
            try:
                ids = json.loads(row.get("top_feature_ids") or "[]")
                deltas = json.loads(row.get("top_feature_deltas") or "[]")
            except json.JSONDecodeError:
                continue
            for feature, delta in zip(ids, deltas):
                collected[condition][(layer, int(feature))].append(abs(float(delta)))
    return {
        condition: {key: float(np.mean(values)) for key, values in features.items()}
        for condition, features in collected.items()
    }


def _layer_index(name: str) -> int:
    digits = [part for part in str(name).split(".") if part.isdigit()]
    return int(digits[-1]) if digits else -1


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def score_nodes(
    nodes: list[dict[str, Any]], responses: dict[tuple[int, int], float]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split the circuit's nodes into those the probe measured and those it did not."""
    measured, unmeasured = [], []
    for node in nodes:
        try:
            key = (int(node["layer"]), int(node["feature"]))
        except (KeyError, TypeError, ValueError):
            continue
        entry = {
            "node_key": node.get("node_key"),
            "layer": key[0],
            "feature": key[1],
            "depth": node.get("depth"),
            "kind": node.get("kind"),
            "influence": node.get("influence"),
        }
        if key in responses:
            entry["response"] = responses[key]
            measured.append(entry)
        else:
            unmeasured.append(entry)
    return measured, unmeasured


def random_control(
    responses: dict[tuple[int, int], float],
    layer_counts: dict[int, int],
    *,
    draws: int,
    rng: random.Random,
) -> list[float]:
    """Draw feature sets with the circuit's per-layer shape, from what the probe saw.

    Matching the layer composition matters: response magnitude varies by layer
    by more than an order of magnitude, so an unmatched control would mostly
    measure which layers the circuit happens to live in.
    """
    by_layer: dict[int, list[float]] = defaultdict(list)
    for (layer, _feature), value in responses.items():
        by_layer[layer].append(value)

    samples: list[float] = []
    for _ in range(draws):
        picked: list[float] = []
        for layer, count in layer_counts.items():
            pool = by_layer.get(layer, [])
            if not pool:
                continue
            picked.extend(rng.choice(pool) for _ in range(count))
        if picked:
            samples.append(float(np.mean(picked)))
    return samples


def permutation_p(observed: float, control: list[float]) -> float | None:
    """Fraction of matched random circuits scoring at least as high."""
    if not control or math.isnan(observed):
        return None
    at_least = sum(1 for value in control if value >= observed)
    return (at_least + 1) / (len(control) + 1)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.trace_dir / "counterfactual_validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    nodes, trace_config = load_circuit(args.trace_dir)
    responses = load_probe_responses(args.probe_run)
    target_condition = "target"
    if target_condition not in responses:
        raise SystemExit(f"Probe run has no '{target_condition}' condition; found {sorted(responses)}")

    print(f"circuit: {len(nodes)} nodes from {args.trace_dir}")
    print(f"probe:   {args.probe_run}  conditions={sorted(responses)}")
    print(f"traced target: {trace_config.get('target')}\n")

    report: dict[str, Any] = {
        "trace_dir": str(args.trace_dir),
        "probe_run": str(args.probe_run),
        "traced_target": trace_config.get("target"),
        "circuit_nodes": len(nodes),
        "conditions": {},
    }

    measured_target, unmeasured = score_nodes(nodes, responses[target_condition])
    coverage = len(measured_target) / max(1, len(nodes))
    print(f"coverage: the probe recorded {len(measured_target)}/{len(nodes)} circuit nodes "
          f"({coverage:.0%})")
    if unmeasured:
        print(f"  {len(unmeasured)} nodes were never in the probe's top-K, so their response is "
              "unknown rather than zero; they are excluded, not counted as silent.")
    report["coverage"] = coverage
    report["measured_nodes"] = len(measured_target)
    report["unmeasured_nodes"] = len(unmeasured)

    if not measured_target:
        print("\nNo circuit node was measured by the probe. Either the trace and the probe are "
              "about different layers, or the probe's --top-features is too small to reach them. "
              "Nothing can be concluded.")
        (output_dir / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return

    layer_counts: dict[int, int] = defaultdict(int)
    for entry in measured_target:
        layer_counts[entry["layer"]] += 1

    print(f"\n{'condition':<10} {'circuit mean':>13} {'random mean':>12} {'ratio':>7} {'p':>7}")
    for condition in sorted(responses):
        measured, _ = score_nodes(nodes, responses[condition])
        values = [entry["response"] for entry in measured]
        circuit_mean = _mean(values)
        control = random_control(responses[condition], layer_counts, draws=args.random_draws, rng=rng)
        control_mean = _mean(control)
        ratio = circuit_mean / control_mean if control_mean else float("nan")
        p_value = permutation_p(circuit_mean, control)
        report["conditions"][condition] = {
            "circuit_mean": circuit_mean,
            "random_mean": control_mean,
            "ratio": ratio,
            "permutation_p": p_value,
            "measured_nodes": len(values),
        }
        print(f"{condition:<10} {circuit_mean:>13.5g} {control_mean:>12.5g} "
              f"{ratio:>7.2f} {('n/a' if p_value is None else f'{p_value:.3f}'):>7}")

    target_stats = report["conditions"].get("target", {})
    placebo_stats = report["conditions"].get("placebo", {})
    verdict: list[str] = []

    ratio = target_stats.get("ratio")
    p_value = target_stats.get("permutation_p")
    # `is not None`, not truthiness: a ratio of exactly 0.0 is the strongest
    # possible negative result and must not be silently skipped.
    if ratio is not None and not math.isnan(ratio):
        if ratio > 1.0 and p_value is not None and p_value < 0.05:
            verdict.append(
                f"The circuit's features respond {ratio:.2f}x more than matched random features "
                f"to the controlled perturbation (permutation p={p_value:.3f}). The trace is "
                "corroborated on a cause we set, not one we inferred from activations."
            )
        else:
            verdict.append(
                f"The circuit responds {ratio:.2f}x random (p="
                f"{'n/a' if p_value is None else f'{p_value:.3f}'}). That is not enrichment, so "
                "these edges are not carrying the perturbed property, whatever else they carry."
            )

    placebo_mean = placebo_stats.get("circuit_mean")
    target_mean = target_stats.get("circuit_mean")
    if placebo_mean and target_mean is not None and not math.isnan(target_mean):
        selectivity = target_mean / placebo_mean
        report["circuit_selectivity_target_over_placebo"] = selectivity
        verdict.append(
            f"On the circuit's own features, perturbing the task object moves them "
            f"{selectivity:.2f}x more than perturbing the pixel-matched control object."
        )
        if selectivity < 1.5:
            verdict.append(
                "  That is weak: the circuit reacts almost as much to the control object, so it "
                "looks tuned to generic change rather than to this object."
            )

    if coverage < 0.5:
        verdict.append(
            f"Caveat: only {coverage:.0%} of the circuit was measured. Raise the probe's "
            "--top-features before treating this as conclusive."
        )

    print("\n--- verdict ---")
    for line in verdict:
        print(" ", line)
    report["verdict"] = verdict

    with (output_dir / "circuit_node_responses.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["node_key", "layer", "feature", "depth", "kind", "influence", "response"],
        )
        writer.writeheader()
        writer.writerows(sorted(measured_target, key=lambda row: -row["response"]))
    (output_dir / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nArtifacts in {output_dir}")


if __name__ == "__main__":
    main()

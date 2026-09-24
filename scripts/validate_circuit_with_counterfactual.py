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

Two design points keep the test honest.

The target node is excluded. It was nominated *because* it responded to the
probe, so scoring it inside the circuit would make enrichment circular. Only
the parents the tracer found are tested; the target's own response is printed
for reference.

The null is matched on layer, flow time and "exercised by this scene". A
feature the scene never activates has a genuine zero response here, and a
random draw over all 16384 features would be mostly such zeros, making any
active circuit look enriched. So the random sets are drawn, per parent, from
features at the same (layer, flow time) that are active in some baseline or
move under the perturbation in some cell. Parents that are not exercised are
reported separately; the fraction that is exercised bounds what the test can
say.

With the probe's full delta store every parent has an exact response at its own
flow time. Without it the script falls back to the top-K CSV, where a node the
probe never recorded has an unknown response and coverage must be read first.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pi05_mi.counterfactual_store import DeltaStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace_dir", type=Path, help="Circuit trace output directory (holds graph.json).")
    parser.add_argument("probe_run", type=Path, help="Counterfactual probe run directory.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--random-draws", type=int, default=2000, help="Matched random control sets.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prompt",
        default="task",
        help="Prompt condition of the probe cells to score on (default: the task prompt, the H1 condition).",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--min-exercised-fraction",
        type=float,
        default=0.5,
        help="Below this fraction of exercised parents the verdict is inconclusive rather than decided.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------- shared


def load_circuit(trace_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = trace_dir / "graph.json"
    if not path.exists():
        raise SystemExit(f"No graph.json in {trace_dir}; run trace_pi05_transcoder_circuit.py first.")
    graph = json.loads(path.read_text())
    return graph.get("nodes", []), graph.get("config", {})


def split_target_and_parents(
    nodes: list[dict[str, Any]], target_key: str | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The target was selected for responding; it must not be scored as evidence."""
    targets, parents = [], []
    for node in nodes:
        if node.get("kind") == "target" or (target_key and node.get("node_key") == target_key):
            targets.append(node)
        else:
            parents.append(node)
    return targets, parents


def monte_carlo_p(observed: float, control: list[float] | np.ndarray) -> float | None:
    """(1 + #{draws >= observed}) / (M + 1): conservative, and never exactly zero."""
    control = np.asarray(control, dtype=np.float64)
    if control.size == 0 or observed is None or math.isnan(observed):
        return None
    return float((1 + int((control >= observed).sum())) / (control.size + 1))


# Kept under its old name for callers written against the first version.
permutation_p = monte_carlo_p


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def h3_verdict(
    *, enrichment: float | None, p_value: float | None, exercised_fraction: float | None,
    alpha: float, min_exercised: float,
) -> str:
    if enrichment is None or p_value is None or exercised_fraction is None or math.isnan(enrichment):
        return "not measured"
    if exercised_fraction < min_exercised:
        return f"inconclusive: only {exercised_fraction:.0%} of parents are exercised by this scene"
    if enrichment > 1.0 and p_value < alpha:
        return "supported"
    return "falsified"


# ---------------------------------------------------------------- full-delta path


def validate_with_store(
    store: DeltaStore,
    *,
    parents: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    prompt: str | None,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    conditions = store.conditions()
    if "target" not in conditions:
        raise SystemExit(f"Store has no 'target' condition; found {conditions}")
    if prompt is not None and prompt not in store.prompts():
        print(f"store has no prompt {prompt!r} (found {store.prompts()}); pooling all prompts", flush=True)
        prompt = None

    responses: dict[str, np.ndarray] = {}
    cells: dict[str, int] = {}
    for condition in conditions:
        responses[condition], cells[condition] = store.mean_abs_delta(condition=condition, prompt=prompt)
    exercised = store.exercised_mask(condition="target", prompt=prompt)

    def locate(node: dict[str, Any]) -> dict[str, Any] | None:
        try:
            layer, feature, tau = int(node["layer"]), int(node["feature"]), float(node["timestep"])
        except (KeyError, TypeError, ValueError):
            return None
        position = store.layer_position(layer)
        if position is None or not 0 <= feature < store.num_features:
            return None
        step = store.step_for_tau(tau)
        entry = {
            "node_key": node.get("node_key"),
            "layer": layer,
            "timestep": tau,
            "feature": feature,
            "depth": node.get("depth"),
            "kind": node.get("kind"),
            "influence": node.get("influence"),
            "step": step,
            "tau_exact": abs(store.tau_for_step(step) - tau) < 1e-6,
            "exercised": bool(exercised[position, step, feature]),
            "_position": position,
        }
        for condition, array in responses.items():
            entry[f"response_{condition}"] = float(array[position, step, feature])
        return entry

    entries = [e for e in (locate(node) for node in parents) if e is not None]
    unlocatable = len(parents) - len(entries)
    exercised_entries = [e for e in entries if e["exercised"]]
    target_entries = [e for e in (locate(node) for node in targets) if e is not None]

    report: dict[str, Any] = {
        "mode": "full-delta",
        "prompt": prompt or "all",
        "cells_per_condition": cells,
        "parent_nodes": len(parents),
        "unlocatable_parents": unlocatable,
        "excluded_target_nodes": len(targets),
        "exercised_parents": len(exercised_entries),
        "exercised_fraction": (len(exercised_entries) / len(entries)) if entries else None,
        "tau_mismatch_nodes": sum(1 for e in entries if not e["tau_exact"]),
        "target_node_response": {
            e["node_key"]: {c: e.get(f"response_{c}") for c in conditions} for e in target_entries
        },
        "conditions": {},
    }
    if not exercised_entries:
        report["nodes"] = [_public(e) for e in entries]
        return report

    # Matched null: per parent, a random exercised feature at the same (layer, step).
    rng = np.random.default_rng(seed)
    pools = {}
    for e in exercised_entries:
        key = (e["_position"], e["step"])
        if key not in pools:
            pools[key] = np.flatnonzero(exercised[key[0], key[1]])
    picks = np.empty((draws, len(exercised_entries)), dtype=np.int64)
    for column, e in enumerate(exercised_entries):
        pool = pools[(e["_position"], e["step"])]
        picks[:, column] = pool[rng.integers(0, pool.size, size=draws)]

    for condition, array in responses.items():
        circuit_values = [e[f"response_{condition}"] for e in exercised_entries]
        circuit_mean = _mean(circuit_values)
        control = np.empty(draws, dtype=np.float64)
        for column, e in enumerate(exercised_entries):
            control_col = array[e["_position"], e["step"], picks[:, column]]
            control = control + control_col if column else control_col.astype(np.float64)
        control = control / len(exercised_entries)
        control_mean = float(control.mean())
        report["conditions"][condition] = {
            "circuit_mean": circuit_mean,
            "random_mean": control_mean,
            "ratio": (circuit_mean / control_mean) if control_mean > 0 else float("nan"),
            "monte_carlo_p": monte_carlo_p(circuit_mean, control),
            "draws": int(draws),
            "scored_nodes": len(circuit_values),
        }
    report["nodes"] = [_public(e) for e in entries]
    return report


def _public(entry: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in entry.items() if not key.startswith("_")}


# ---------------------------------------------------------------- legacy top-K path


def load_probe_responses(run_dir: Path) -> dict[str, dict[tuple[int, int], float]]:
    """Per-condition mean |delta| for every (layer, feature) the probe's top-K recorded."""
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


def score_nodes(
    nodes: list[dict[str, Any]], responses: dict[tuple[int, int], float]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split nodes into those the probe's top-K measured and those it did not."""
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
    """Feature sets with the circuit's per-layer shape, drawn from what the top-K recorded."""
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


def validate_with_csv(
    run_dir: Path, *, parents: list[dict[str, Any]], targets: list[dict[str, Any]], draws: int, seed: int
) -> dict[str, Any]:
    responses = load_probe_responses(run_dir)
    if "target" not in responses:
        raise SystemExit(f"Probe run has no 'target' condition; found {sorted(responses)}")
    rng = random.Random(seed)
    measured, unmeasured = score_nodes(parents, responses["target"])
    report: dict[str, Any] = {
        "mode": "top-k-csv",
        "parent_nodes": len(parents),
        "excluded_target_nodes": len(targets),
        "measured_nodes": len(measured),
        "unmeasured_nodes": len(unmeasured),
        "coverage": (len(measured) / len(parents)) if parents else None,
        "exercised_fraction": (len(measured) / len(parents)) if parents else None,
        "conditions": {},
        "nodes": measured,
    }
    if not measured:
        return report
    layer_counts: dict[int, int] = defaultdict(int)
    for entry in measured:
        layer_counts[entry["layer"]] += 1
    for condition in sorted(responses):
        scored, _ = score_nodes(parents, responses[condition])
        values = [entry["response"] for entry in scored]
        circuit_mean = _mean(values)
        control = random_control(responses[condition], layer_counts, draws=draws, rng=rng)
        control_mean = _mean(control)
        report["conditions"][condition] = {
            "circuit_mean": circuit_mean,
            "random_mean": control_mean,
            "ratio": circuit_mean / control_mean if control_mean else float("nan"),
            "monte_carlo_p": monte_carlo_p(circuit_mean, control),
            "draws": len(control),
            "scored_nodes": len(values),
        }
    return report


# ---------------------------------------------------------------- main


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.trace_dir / "counterfactual_validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    nodes, trace_config = load_circuit(args.trace_dir)
    target_key = trace_config.get("target")
    targets, parents = split_target_and_parents(nodes, target_key)
    print(f"circuit: {len(nodes)} nodes from {args.trace_dir} ({len(parents)} parents, "
          f"{len(targets)} target node excluded from scoring)")
    print(f"probe:   {args.probe_run}")
    print(f"traced target: {target_key}\n")

    store_root = args.probe_run / "latents"
    prompt = None if args.prompt == "all" else args.prompt
    if DeltaStore.exists(store_root):
        store = DeltaStore.open(store_root)
        report = validate_with_store(
            store, parents=parents, targets=targets, prompt=prompt, draws=args.random_draws, seed=args.seed
        )
        print(f"mode: full delta store ({report['cells_per_condition']} cells per condition, prompt={report['prompt']})")
        print(f"exercised parents: {report['exercised_parents']}/{report['parent_nodes']} "
              f"({'n/a' if report['exercised_fraction'] is None else f'{report['exercised_fraction']:.0%}'})")
        if report["unlocatable_parents"]:
            print(f"  {report['unlocatable_parents']} parents sit in layers the probe did not instrument")
        if report["tau_mismatch_nodes"]:
            print(f"  {report['tau_mismatch_nodes']} parents have a flow time the probe did not sample exactly; "
                  "the nearest step was used")
        for key, values in report["target_node_response"].items():
            print(f"target {key} own response (reference only): "
                  + ", ".join(f"{c}={v:.4g}" for c, v in values.items()))
    else:
        report = validate_with_csv(args.probe_run, parents=parents, targets=targets, draws=args.random_draws, seed=args.seed)
        print("mode: top-K CSV (no full delta store found; rerun the probe to get exact coverage)")
        print(f"coverage: the probe recorded {report['measured_nodes']}/{report['parent_nodes']} parents "
              f"({'n/a' if report['coverage'] is None else f'{report['coverage']:.0%}'})")
        if report["unmeasured_nodes"]:
            print(f"  {report['unmeasured_nodes']} parents were never in the probe's top-K, so their response is "
                  "unknown rather than zero; they are excluded, not counted as silent.")

    report.update({
        "trace_dir": str(args.trace_dir),
        "probe_run": str(args.probe_run),
        "traced_target": target_key,
        "circuit_nodes": len(nodes),
        "alpha": args.alpha,
        "min_exercised_fraction": args.min_exercised_fraction,
    })

    verdict: list[str] = []
    if not report["conditions"]:
        if not parents:
            verdict.append("The circuit has no parent nodes; only the target, which is excluded by design. "
                           "Nothing to test.")
        else:
            verdict.append("No parent node is exercised by this scene, so nothing can be concluded about "
                           "whether the traced edges carry the perturbed property.")
        report["h3_verdict"] = "not measured"
    else:
        print(f"\n{'condition':<10} {'circuit mean':>13} {'random mean':>12} {'enrich':>8} {'p':>8} {'nodes':>6}")
        for condition, stats in report["conditions"].items():
            p_value = stats["monte_carlo_p"]
            print(f"{condition:<10} {stats['circuit_mean']:>13.5g} {stats['random_mean']:>12.5g} "
                  f"{stats['ratio']:>8.2f} {('n/a' if p_value is None else f'{p_value:.4f}'):>8} {stats['scored_nodes']:>6}")

        target_stats = report["conditions"].get("target", {})
        placebo_stats = report["conditions"].get("placebo", {})
        ratio = target_stats.get("ratio")
        p_value = target_stats.get("monte_carlo_p")
        decision = h3_verdict(
            enrichment=ratio, p_value=p_value, exercised_fraction=report.get("exercised_fraction"),
            alpha=args.alpha, min_exercised=args.min_exercised_fraction,
        )
        report["h3_verdict"] = decision
        # `is not None`, not truthiness: a ratio of exactly 0.0 is the strongest
        # possible negative result and must not be silently skipped.
        if ratio is not None and not math.isnan(ratio):
            if decision == "supported":
                verdict.append(
                    f"The traced parents respond {ratio:.2f}x more than matched random features "
                    f"to the controlled perturbation (Monte-Carlo p={p_value:.4f}). The trace is "
                    "corroborated on a cause we set, not one we inferred from activations."
                )
            elif decision.startswith("inconclusive"):
                verdict.append(
                    f"Enrichment is {ratio:.2f}x (p={'n/a' if p_value is None else f'{p_value:.4f}'}), but "
                    f"{decision}. The scene does not exercise enough of the circuit to decide."
                )
            else:
                verdict.append(
                    f"The traced parents respond {ratio:.2f}x random (p="
                    f"{'n/a' if p_value is None else f'{p_value:.4f}'}). That is not enrichment, so "
                    "these edges are not carrying the perturbed property, whatever else they carry."
                )
        placebo_mean = placebo_stats.get("circuit_mean")
        target_mean = target_stats.get("circuit_mean")
        if placebo_mean and target_mean is not None and not math.isnan(target_mean):
            selectivity = target_mean / placebo_mean
            report["circuit_selectivity_target_over_placebo"] = selectivity
            verdict.append(
                f"On the traced parents themselves, perturbing the task object moves them "
                f"{selectivity:.2f}x more than perturbing the pixel-matched control object."
            )
            if selectivity < 1.5:
                verdict.append(
                    "  That is weak: the circuit reacts almost as much to the control object, so it "
                    "looks tuned to generic change rather than to this object."
                )
        elif placebo_mean == 0.0 and target_mean:
            report["circuit_selectivity_target_over_placebo"] = None
            verdict.append("The traced parents did not move at all under the placebo perturbation.")

    print("\n--- verdict ---")
    for line in verdict:
        print(" ", line)
    print(f"H3 verdict: {report['h3_verdict']}")
    report["verdict"] = verdict

    node_rows = report.get("nodes", [])
    if node_rows:
        fieldnames = sorted({key for row in node_rows for key in row})
        with (output_dir / "circuit_node_responses.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(node_rows)
    report_public = {k: v for k, v in report.items() if k != "nodes"}
    (output_dir / "validation.json").write_text(json.dumps(report_public, indent=2), encoding="utf-8")
    print(f"\nArtifacts in {output_dir}")


if __name__ == "__main__":
    main()

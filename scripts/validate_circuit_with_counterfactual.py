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

What the test is on matters. Asking whether the parents *respond* more than
random cannot separate a set that carries the perturbed property from a set of
generically responsive features: attribution favours features that are active
and high-variance, and those respond more to any manipulation. So the primary
statistic is *selectivity*, the circuit's target-over-placebo response ratio
against the same ratio on matched random sets, and it is accompanied by a
specificity check against every other manipulation the probe measured (a
prompt swap, say). A set enriched as much for a language change as for a
recolour is not carrying the recolour.

Results are also reported as a function of how much of the circuit is
included, ordered by traced influence. A tracer whose ranking means something
shows strong enrichment among its top parents that decays down the list; a
flat profile says the ranking carries no information about the property, and a
large flat circuit is a pruning setting, not a finding.

With the probe's full delta store every parent has an exact response at its own
flow time. Without it the script falls back to the top-K CSV, where a node the
probe never recorded has an unknown response and coverage must be read first,
and where no paired selectivity statistic is available.
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


def _ok(stats: dict[str, Any] | None) -> bool:
    return bool(stats) and stats.get("enrichment") is not None and not math.isnan(stats["enrichment"])


def h3_verdict(
    *,
    head: dict[str, Any] | None,
    full: dict[str, Any] | None,
    specificity: float | None,
    response_enrichment: float | None,
    response_p: float | None,
    exercised_fraction: float | None,
    alpha: float,
    min_exercised: float,
    head_nodes: int | None = None,
) -> str:
    """Decide H3 on selectivity enrichment at the head of the influence ranking.

    Three facts shape the rule. Bare response enrichment cannot separate a
    property-carrying circuit from a generically responsive one, since
    attribution selects active, high-variance features and those respond more
    to any manipulation; selectivity divides that common factor out, because
    generic responsiveness scales the target and placebo responses alike and
    leaves their ratio at the random set's. At large node counts every
    Monte-Carlo p saturates at 1/(M+1) whatever the effect size, so the
    decision is taken where p is informative: the top parents by traced
    influence, which is also the tracer's strongest claim. And a set enriched
    as much for a manipulation of a different kind is not thereby falsified,
    because a circuit that integrates the referent from vision and language
    would look exactly like that; specificity qualifies a supported verdict,
    it does not overturn one.

    ``head`` and ``full`` are selectivity statistics for the decision stratum
    and for the whole circuit; the full circuit must agree in direction.
    """
    if exercised_fraction is None:
        return "not measured"
    if exercised_fraction < min_exercised:
        return f"inconclusive: only {exercised_fraction:.1%} of parents are exercised by this scene"

    if _ok(head) or _ok(full):
        decide = head if _ok(head) else full
        where = f"the top {head_nodes} parents by influence" if (head_nodes and _ok(head)) else "the circuit"
        if decide["enrichment"] <= 1.0 or decide.get("monte_carlo_p") is None or decide["monte_carlo_p"] >= alpha:
            return (
                f"falsified: {where} are no more selective for the perturbed object than matched "
                f"random features (selectivity enrichment {decide['enrichment']:.3f}, p="
                f"{'n/a' if decide.get('monte_carlo_p') is None else f'{decide['monte_carlo_p']:.4f}'})"
            )
        if _ok(full) and full is not decide and full["enrichment"] <= 1.0:
            return (
                f"falsified: {where} are enriched but the circuit as a whole is less selective than "
                f"random ({full['enrichment']:.3f}); the head does not represent the set"
            )
        if specificity is None or math.isnan(specificity):
            return "supported (no manipulation of a different kind was measured, so specificity is untested)"
        if specificity > 1.0:
            return "supported: object-specific"
        return (
            "supported, not specific to the recolour: as enriched for a manipulation of a different "
            "kind, which is consistent with a circuit that integrates the referent from vision and "
            "language and cannot be separated from that here"
        )

    if response_enrichment is None or response_p is None or math.isnan(response_enrichment):
        return "not measured"
    if response_enrichment > 1.0 and response_p < alpha:
        return "supported (response enrichment only: no placebo condition for the selectivity test)"
    return "falsified"

    if response_enrichment is None or response_p is None or math.isnan(response_enrichment):
        return "not measured"
    if response_enrichment > 1.0 and response_p < alpha:
        return "supported (response enrichment only: no placebo condition for the selectivity test)"
    return "falsified"


def selectivity_stats(
    circuit_target: float, circuit_placebo: float, control_target: np.ndarray, control_placebo: np.ndarray
) -> dict[str, Any] | None:
    """Circuit target/placebo ratio against the same ratio on each random draw.

    The random draws are paired across conditions (the same features are drawn
    for target and placebo), so a per-draw ratio is well defined.
    """
    if circuit_placebo <= 0 or control_target.size == 0:
        return None
    usable = control_placebo > 0
    if not usable.any():
        return None
    random_sel = control_target[usable] / control_placebo[usable]
    circuit_sel = circuit_target / circuit_placebo
    random_mean = float(random_sel.mean())
    return {
        "circuit": float(circuit_sel),
        "random_mean": random_mean,
        "enrichment": float(circuit_sel / random_mean) if random_mean > 0 else float("nan"),
        "monte_carlo_p": monte_carlo_p(circuit_sel, random_sel),
        "draws": int(usable.sum()),
    }


def specificity_stats(conditions: dict[str, dict[str, Any]]) -> tuple[float | None, dict[str, float]]:
    """Target enrichment against enrichment on manipulations of a different kind.

    The placebo is excluded: it is the matched control for the *same*
    manipulation and is already the denominator of the selectivity statistic.
    What is wanted here is a different kind of intervention entirely, such as
    a prompt swap.
    """
    target = conditions.get("target", {}).get("ratio")
    others = {
        name: stats["ratio"]
        for name, stats in conditions.items()
        if name not in ("target", "placebo", "null")
        and stats.get("ratio") is not None
        and not math.isnan(stats["ratio"])
        and stats["ratio"] > 0
    }
    if target is None or math.isnan(target) or not others:
        return None, others
    return float(target / max(others.values())), others


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

    # Matched null: per parent, a random exercised feature at the same (layer,
    # step). The same draws serve every condition, so per-draw ratios pair.
    rng = np.random.default_rng(seed)
    pools: dict[tuple[int, int], np.ndarray] = {}
    for e in exercised_entries:
        key = (e["_position"], e["step"])
        if key not in pools:
            pools[key] = np.flatnonzero(exercised[key[0], key[1]])
    picks = np.empty((draws, len(exercised_entries)), dtype=np.int64)
    for column, e in enumerate(exercised_entries):
        pool = pools[(e["_position"], e["step"])]
        picks[:, column] = pool[rng.integers(0, pool.size, size=draws)]

    # Order by traced influence, so enrichment can be read as a function of how
    # much of the circuit is included.
    order = sorted(
        range(len(exercised_entries)), key=lambda i: -(exercised_entries[i].get("influence") or 0.0)
    )
    ordered = [exercised_entries[i] for i in order]
    picks = picks[:, order]
    n_parents = len(ordered)
    sizes = [k for k in (10, 50, 200) if k < n_parents] + [n_parents]
    report["influence_ranked"] = sum(1 for e in ordered if e.get("influence"))

    circuit_cumsum = {
        condition: np.cumsum([float(e[f"response_{condition}"]) for e in ordered])
        for condition in responses
    }
    control_at: dict[str, dict[int, np.ndarray]] = {c: {} for c in responses}
    for condition, array in responses.items():
        running = np.zeros(draws, dtype=np.float64)
        for column, e in enumerate(ordered):
            running = running + array[e["_position"], e["step"], picks[:, column]]
            if column + 1 in sizes:
                control_at[condition][column + 1] = running.copy()

    strata = []
    for k in sizes:
        conditions: dict[str, dict[str, Any]] = {}
        control_means: dict[str, np.ndarray] = {}
        for condition in responses:
            circuit_mean = float(circuit_cumsum[condition][k - 1] / k)
            control = control_at[condition][k] / k
            control_means[condition] = control
            random_mean = float(control.mean())
            conditions[condition] = {
                "circuit_mean": circuit_mean,
                "random_mean": random_mean,
                "ratio": (circuit_mean / random_mean) if random_mean > 0 else float("nan"),
                "monte_carlo_p": monte_carlo_p(circuit_mean, control),
                "draws": int(draws),
                "scored_nodes": int(k),
            }
        selectivity = None
        if "target" in conditions and "placebo" in conditions:
            selectivity = selectivity_stats(
                conditions["target"]["circuit_mean"],
                conditions["placebo"]["circuit_mean"],
                control_means["target"],
                control_means["placebo"],
            )
        specificity, others = specificity_stats(conditions)
        strata.append(
            {
                "nodes": int(k),
                "is_full_circuit": k == n_parents,
                "conditions": conditions,
                "selectivity": selectivity,
                "specificity": specificity,
                "specificity_against": others,
            }
        )

    full = strata[-1]
    report["strata"] = strata
    report["conditions"] = full["conditions"]
    report["selectivity"] = full["selectivity"]
    report["specificity"] = full["specificity"]
    report["specificity_against"] = full["specificity_against"]

    # A flat enrichment profile means the influence ranking carries no
    # information about the property, and a large flat circuit is a pruning
    # setting rather than a finding.
    top = strata[0]
    if len(strata) > 1 and top["selectivity"] and full["selectivity"]:
        report["selectivity_enrichment_top_vs_full"] = (
            top["selectivity"]["enrichment"] / full["selectivity"]["enrichment"]
            if full["selectivity"]["enrichment"]
            else None
        )
    report["nodes"] = [_public(e) for e in ordered] + [
        _public(e) for e in entries if not e["exercised"]
    ]
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

    # Draw once and evaluate every condition on the same features, so that a
    # per-draw selectivity ratio is defined here too and both paths can be
    # decided by the same rule. The pool is the features the probe recorded
    # under every condition, per layer.
    rng = np.random.default_rng(seed)
    common = set(responses["target"])
    for condition in responses:
        common &= set(responses[condition])
    by_layer: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for key in sorted(common):
        by_layer[key[0]].append(key)

    columns = [entry["layer"] for entry in measured if by_layer.get(entry["layer"])]
    pool_index: dict[int, np.ndarray] = {}
    for column, layer in enumerate(columns):
        pool_index[column] = rng.integers(0, len(by_layer[layer]), size=draws)

    control_means: dict[str, np.ndarray] = {}
    for condition in sorted(responses):
        scored, _ = score_nodes(parents, responses[condition])
        values = [entry["response"] for entry in scored]
        circuit_mean = _mean(values)
        if columns:
            control = np.zeros(draws, dtype=np.float64)
            for column, layer in enumerate(columns):
                pool = np.asarray([responses[condition][key] for key in by_layer[layer]], dtype=np.float64)
                control = control + pool[pool_index[column]]
            control = control / len(columns)
        else:
            control = np.asarray(
                random_control(responses[condition], _layer_counts(measured), draws=draws, rng=random.Random(seed)),
                dtype=np.float64,
            )
        control_means[condition] = control
        control_mean = float(control.mean()) if control.size else float("nan")
        report["conditions"][condition] = {
            "circuit_mean": circuit_mean,
            "random_mean": control_mean,
            "ratio": circuit_mean / control_mean if control_mean else float("nan"),
            "monte_carlo_p": monte_carlo_p(circuit_mean, control),
            "draws": int(control.size),
            "scored_nodes": len(values),
        }

    if "placebo" in report["conditions"]:
        report["selectivity"] = selectivity_stats(
            report["conditions"]["target"]["circuit_mean"],
            report["conditions"]["placebo"]["circuit_mean"],
            control_means["target"],
            control_means["placebo"],
        )
    report["specificity"], report["specificity_against"] = specificity_stats(report["conditions"])
    return report


def _layer_counts(measured: list[dict[str, Any]]) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    for entry in measured:
        counts[entry["layer"]] += 1
    return counts


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
              f"({'n/a' if report['exercised_fraction'] is None else f'{report['exercised_fraction']:.1%}'})")
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

        strata = report.get("strata") or []
        if strata:
            print(f"\n{'influence stratum':<18} {'nodes':>7} {'target enr':>11} {'selectivity':>12} "
                  f"{'sel enrich':>11} {'sel p':>8} {'specificity':>12}")
            for stratum in strata:
                sel = stratum.get("selectivity") or {}
                label = "all (full circuit)" if stratum["is_full_circuit"] else f"top {stratum['nodes']}"
                spec = stratum.get("specificity")
                sel_p = sel.get("monte_carlo_p")
                sel_p_text = "n/a" if sel_p is None else f"{sel_p:.4f}"
                spec_text = "n/a" if spec is None else f"{spec:.3f}"
                target_ratio = stratum["conditions"].get("target", {}).get("ratio", float("nan"))
                print(
                    f"{label:<18} {stratum['nodes']:>7} {target_ratio:>11.3f} "
                    f"{sel.get('circuit', float('nan')):>12.3f} "
                    f"{sel.get('enrichment', float('nan')):>11.3f} "
                    f"{sel_p_text:>8} {spec_text:>12}"
                )
            print("  selectivity = the circuit's target/placebo response ratio; sel enrich = that ratio "
                  "over a matched random set's;")
            print("  specificity = target enrichment over enrichment on a manipulation of a different "
                  "kind. Both must exceed 1.")

        target_stats = report["conditions"].get("target", {})
        placebo_stats = report["conditions"].get("placebo", {})
        ratio = target_stats.get("ratio")
        p_value = target_stats.get("monte_carlo_p")
        selectivity = report.get("selectivity") or {}
        head_stratum = strata[0] if strata else None
        head_sel = (head_stratum or {}).get("selectivity")
        head_spec = (head_stratum or {}).get("specificity")
        report["decision_stratum_nodes"] = head_stratum["nodes"] if head_stratum else None
        decision = h3_verdict(
            head=head_sel,
            full=selectivity or None,
            specificity=head_spec if head_stratum else report.get("specificity"),
            response_enrichment=ratio,
            response_p=p_value,
            exercised_fraction=report.get("exercised_fraction"),
            alpha=args.alpha,
            min_exercised=args.min_exercised_fraction,
            head_nodes=head_stratum["nodes"] if head_stratum else None,
        )
        report["h3_verdict"] = decision
        # `is not None`, not truthiness: a ratio of exactly 0.0 is the strongest
        # possible negative result and must not be silently skipped.
        sel_enrich = selectivity.get("enrichment")
        sel_p = selectivity.get("monte_carlo_p")
        specificity = report.get("specificity")
        head_enrich = (head_sel or {}).get("enrichment")
        head_p = (head_sel or {}).get("monte_carlo_p")
        # The deciding statistic: the head stratum when there is one, else the
        # circuit's own selectivity (legacy path, or a circuit smaller than the head).
        deciding = head_sel if head_enrich is not None else (selectivity or None)
        deciding_enrich = (deciding or {}).get("enrichment")
        deciding_p = (deciding or {}).get("monte_carlo_p")
        deciding_where = (
            f"The top {head_stratum['nodes']} parents by influence"
            if head_enrich is not None and head_stratum and not head_stratum["is_full_circuit"]
            else "The traced parents"
        )
        if decision.startswith("supported") and deciding_enrich is not None and not math.isnan(deciding_enrich):
            agree = (
                f", and the full circuit agrees in direction ({sel_enrich:.3f}x)"
                if deciding is head_sel and sel_enrich is not None and head_stratum and not head_stratum["is_full_circuit"]
                else ""
            )
            verdict.append(
                f"{deciding_where} are {deciding_enrich:.3f}x more selective for the perturbed object than "
                f"matched random features (Monte-Carlo p={deciding_p:.4f}){agree}. The trace is "
                "corroborated on a cause we set, not one we inferred from activations."
            )
            if decision.startswith("supported, not specific"):
                verdict.append(
                    "  It is as enriched for the prompt swap as for the recolour. That does not undo the "
                    "result: a circuit that computes the referent from both vision and language would "
                    "look like this. It does mean the parents are not a colour-only handle."
                )
        elif decision.startswith("supported"):
            verdict.append(
                f"The traced parents respond {ratio:.2f}x more than matched random features to the "
                f"controlled perturbation (Monte-Carlo p={p_value:.4f}), which corroborates the trace. "
                "No placebo condition was recorded, so the stronger selectivity test could not be run "
                "and the result cannot separate this property from generic responsiveness."
            )
        elif decision.startswith("inconclusive"):
            verdict.append(f"{decision}. The scene does not exercise enough of the circuit to decide.")
        elif decision.startswith("not measured"):
            verdict.append("Nothing measurable: no selectivity statistic and no response enrichment.")
        else:
            if sel_enrich is None or math.isnan(sel_enrich):
                verdict.append(
                    f"The traced parents respond {ratio:.2f}x random (p="
                    f"{'n/a' if p_value is None else f'{p_value:.4f}'}). That is not enrichment, so "
                    "these edges are not carrying the perturbed property, whatever else they carry."
                )
            if sel_enrich is not None and not math.isnan(sel_enrich):
                head_note = (
                    f" At the top {head_stratum['nodes']} by influence it is {head_enrich:.3f}x (p="
                    f"{'n/a' if head_p is None else f'{head_p:.4f}'})."
                    if head_enrich is not None and head_stratum and not head_stratum["is_full_circuit"]
                    else ""
                )
                verdict.append(
                    f"The traced parents respond {ratio:.2f}x more than random, but they are only "
                    f"{sel_enrich:.3f}x more SELECTIVE than random "
                    f"({selectivity.get('circuit', float('nan')):.2f}x against "
                    f"{selectivity.get('random_mean', float('nan')):.2f}x).{head_note} Response enrichment "
                    "alone does not separate a circuit that carries this property from a set of "
                    "generically responsive features; selectivity does, and here it is at the random level."
                )
            if specificity is not None and not math.isnan(specificity) and specificity <= 1.0:
                against = ", ".join(
                    f"{name} {value:.2f}x" for name, value in (report.get("specificity_against") or {}).items()
                )
                verdict.append(
                    f"  Consistent with that: the set is enriched {ratio:.2f}x for the recolour and {against} "
                    "for a manipulation of a different kind. With selectivity at the random level, equal "
                    "enrichment across unrelated interventions reads as responsiveness in general."
                )
            if p_value is not None and p_value <= 1.5 / (target_stats.get("draws") or 1):
                verdict.append(
                    f"  Note the p-value is at its floor of 1/(M+1); with {target_stats.get('scored_nodes')} "
                    "nodes the test detects arbitrarily small differences, so read the effect size."
                )
        placebo_mean = placebo_stats.get("circuit_mean")
        target_mean = target_stats.get("circuit_mean")
        if placebo_mean and target_mean is not None and not math.isnan(target_mean):
            circuit_sel = target_mean / placebo_mean
            report["circuit_selectivity_target_over_placebo"] = circuit_sel
            verdict.append(
                f"On the traced parents themselves, perturbing the task object moves them "
                f"{circuit_sel:.2f}x more than perturbing the pixel-matched control object"
                + (
                    f"; a matched random set of features gives {selectivity['random_mean']:.2f}x, "
                    "which is what that number has to be read against."
                    if selectivity.get("random_mean")
                    else ", but no matched random selectivity is available to read that against."
                )
            )
        elif placebo_mean == 0.0 and target_mean:
            report["circuit_selectivity_target_over_placebo"] = None
            verdict.append("The traced parents did not move at all under the placebo perturbation.")

        ratio_top_full = report.get("selectivity_enrichment_top_vs_full")
        if ratio_top_full is not None and ratio_top_full > 1.25 and len(strata) > 1:
            verdict.append(
                f"  The top {strata[0]['nodes']} parents by influence are {ratio_top_full:.2f}x more "
                "enriched than the circuit as a whole, so the attribution ranking does carry signal and "
                "the graph is simply pruned too loosely. Tighten --node-cumulative-threshold and retrace."
            )
        elif ratio_top_full is not None and len(strata) > 1:
            verdict.append(
                "  Enrichment is flat across the influence ranking, so the ranking carries no "
                "information about this property; a smaller circuit would not help."
            )

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

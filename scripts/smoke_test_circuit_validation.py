#!/usr/bin/env python
"""Smoke test for validating a traced circuit against the counterfactual probe.

Built around a planted ground truth: a set of features is made to respond to
the controlled perturbation, and the validator must rate a circuit built from
those features far above one built from non-responders -- while refusing to
count the target node itself, and refusing to decide when the scene does not
exercise the circuit.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import probe_pi05_transcoder_counterfactual as probe  # noqa: E402
from pi05_mi.counterfactual_store import DeltaStore  # noqa: E402
from validate_circuit_with_counterfactual import (  # noqa: E402
    load_probe_responses,
    monte_carlo_p,
    random_control,
    score_nodes,
    split_target_and_parents,
)

N_FEATURES = 64
STEPS = 4
LAYERS = [f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp" for i in (7, 8, 9)]
RESPONDERS = {7: [5, 6, 7], 8: [5, 6, 7], 9: [5, 6, 7]}
ACTIVE = list(range(32))  # features 0..31 are active at baseline; 32..63 never fire
TAU = 1.0 - 1 / STEPS  # step 1
SCRIPT = Path(__file__).resolve().parent / "validate_circuit_with_counterfactual.py"


def _accumulator(responders: dict[int, list[int]], amplitude: float, *, tilt: float = 0.0):
    """Non-responders sit at 0.02 (+ tilt * index, so top-K membership is not a tie-break)."""
    acc = probe.LatentAccumulator()
    for name in LAYERS:
        layer = int(name.split(".")[-2])
        for step in range(STEPS):
            vector = np.full(N_FEATURES, 0.02) + tilt * np.arange(N_FEATURES)
            for feature in responders.get(layer, []):
                vector[feature] = amplitude
            acc.max[(name, step)] = vector
            acc.mean[(name, step)] = vector
    return acc


def _csv_probe_run() -> Path:
    """Legacy fixture: only the top-K CSV, no store."""
    # Deltas for non-responders are 1e-4 * index, so the top-20 is the 3
    # responders plus the 17 highest indices (47..63), deterministically.
    baseline = _accumulator({}, 0.0)
    target = _accumulator(RESPONDERS, 3.0, tilt=1e-4)
    placebo = _accumulator(RESPONDERS, 0.25, tilt=1e-4)
    rows = []
    for state in (0, 1):
        for condition, other in (("target", target), ("placebo", placebo)):
            for row in probe.latent_delta_rows(baseline, other, condition=condition, top_features=20):
                row.update({"state_index": state, "dose": 1.0, "target": condition, "prompt": "task"})
                rows.append(row)
    run = Path(tempfile.mkdtemp())
    probe.write_csv(run / "latent_deltas.csv", rows)
    return run


def _store_probe_run() -> Path:
    """Full-delta fixture. Active features 0..31; responders 5,6,7 at every layer.

    Target delta: responders 3.0, other active features 0.02, inactive 0.
    Placebo delta: responders 0.25, other active 0.02.
    """
    run = Path(tempfile.mkdtemp())
    store = DeltaStore.create(run / "latents", layer_names=LAYERS, num_steps=STEPS, num_features=N_FEATURES)
    for state in (0, 1):
        for prompt in ("task", "alt"):
            baseline, target, placebo, null = {}, {}, {}, {}
            for name in LAYERS:
                layer = int(name.split(".")[-2])
                for step in range(STEPS):
                    base = np.zeros(N_FEATURES, dtype=np.float32)
                    base[ACTIVE] = 1.0
                    d_t = np.zeros(N_FEATURES, dtype=np.float32)
                    d_t[ACTIVE] = 0.02
                    d_p = d_t.copy()
                    for f in RESPONDERS[layer]:
                        d_t[f] = 3.0 if prompt == "task" else 1.0
                        d_p[f] = 0.25
                    baseline[(name, step)] = base
                    target[(name, step)] = d_t
                    placebo[(name, step)] = d_p
                    null[(name, step)] = np.zeros(N_FEATURES, dtype=np.float32)
            store.write_block(
                state=state, noise=0, prompt=prompt, baseline_max=baseline,
                deltas={("null", 0.0): null, ("target", 1.0): target, ("placebo", 1.0): placebo},
            )
    # Also a CSV, as the probe always writes one; the validator must prefer the store.
    probe.write_csv(run / "latent_deltas.csv", [])
    return run


def _trace_dir(parents_by_layer: dict[int, list[int]], *, target=(9, 5), tau: float = TAU) -> Path:
    trace = Path(tempfile.mkdtemp())
    target_key = f"L{target[0]:02d}:tau{tau:.4g}:F{target[1]}"
    nodes = [{"node_key": target_key, "layer": target[0], "timestep": tau, "feature": target[1],
              "depth": 0, "kind": "target", "influence": 1.0}]
    nodes += [
        {"node_key": f"L{layer:02d}:tau{tau:.4g}:F{feature}", "layer": layer, "timestep": tau,
         "feature": feature, "depth": 1, "kind": "parent", "influence": 0.1}
        for layer, features in parents_by_layer.items()
        for feature in features
    ]
    (trace / "graph.json").write_text(json.dumps({"config": {"target": target_key}, "nodes": nodes, "edges": []}))
    return trace


def _run(trace: Path, run: Path, *extra: str) -> tuple[str, dict]:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(trace), str(run), *extra], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((trace / "counterfactual_validation" / "validation.json").read_text())
    return result.stdout, report


# ------------------------------------------------------------------ full-delta path


def test_store_true_circuit_is_enriched_selective_and_supported() -> None:
    stdout, report = _run(_trace_dir({7: [5, 6], 8: [5, 6, 7]}), _store_probe_run())
    assert report["mode"] == "full-delta"
    assert report["excluded_target_nodes"] == 1 and report["parent_nodes"] == 5
    assert report["exercised_fraction"] == 1.0 and report["tau_mismatch_nodes"] == 0
    target = report["conditions"]["target"]
    # Null pool = 32 exercised features of which 3 respond: E[random] ~ (3*3 + 29*0.02)/32 ~ 0.30
    assert target["ratio"] > 5.0, target
    assert target["monte_carlo_p"] < 0.05, target
    assert report["h3_verdict"] == "supported"
    assert report["circuit_selectivity_target_over_placebo"] > 5.0
    assert report["cells_per_condition"]["target"] == 2, "task prompt only: two states"
    assert "corroborated" in stdout


def test_store_target_node_is_excluded_from_the_evidence() -> None:
    """A circuit of only the target must not be enriched; the target was picked for responding."""
    stdout, report = _run(_trace_dir({}), _store_probe_run())
    assert report["parent_nodes"] == 0 and report["excluded_target_nodes"] == 1
    assert report["conditions"] == {} and report["h3_verdict"] == "not measured"
    assert "excluded by design" in stdout
    # Its own response is still shown for reference, not scored.
    assert abs(list(report["target_node_response"].values())[0]["target"] - 3.0) < 1e-6


def test_store_wrong_circuit_is_not_enriched_and_says_so() -> None:
    stdout, report = _run(_trace_dir({7: [20, 21], 8: [22, 23, 24]}), _store_probe_run())
    target = report["conditions"]["target"]
    assert target["ratio"] < 0.5, target
    assert report["h3_verdict"] == "falsified"
    assert "not enrichment" in stdout


def test_store_unexercised_parents_make_the_verdict_inconclusive() -> None:
    """Features 40+ never fire on this scene: a genuine zero, but not evidence either way."""
    stdout, report = _run(_trace_dir({7: [40, 41], 8: [5, 42]}), _store_probe_run())
    assert report["exercised_parents"] == 1 and abs(report["exercised_fraction"] - 0.25) < 1e-9
    assert report["h3_verdict"].startswith("inconclusive")
    assert "does not exercise enough" in stdout


def test_store_scores_at_the_nodes_own_flow_time() -> None:
    """A node's tau picks the probe step; a tau the probe never sampled is flagged."""
    _, exact = _run(_trace_dir({7: [5]}, tau=TAU), _store_probe_run())
    assert exact["tau_mismatch_nodes"] == 0
    _, off = _run(_trace_dir({7: [5]}, tau=0.9), _store_probe_run())
    assert off["tau_mismatch_nodes"] == 1


def test_store_prompt_selects_cells() -> None:
    _, task = _run(_trace_dir({7: [5, 6]}), _store_probe_run())
    _, pooled = _run(_trace_dir({7: [5, 6]}), _store_probe_run(), "--prompt", "all")
    assert abs(task["conditions"]["target"]["circuit_mean"] - 3.0) < 1e-6
    assert abs(pooled["conditions"]["target"]["circuit_mean"] - 2.0) < 1e-6, "alt cells respond 1.0"
    assert pooled["cells_per_condition"]["target"] == 4


# ------------------------------------------------------------------ legacy top-K path


def test_csv_true_circuit_is_enriched_and_selective() -> None:
    stdout, report = _run(_trace_dir(RESPONDERS), _csv_probe_run())
    assert report["mode"] == "top-k-csv"
    target = report["conditions"]["target"]
    assert report["coverage"] == 1.0
    assert target["ratio"] > 2.0, target
    assert target["monte_carlo_p"] < 0.05, target
    assert report["circuit_selectivity_target_over_placebo"] > 5.0
    assert "corroborated" in stdout


def test_csv_wrong_circuit_reports_a_near_zero_ratio() -> None:
    """A near-zero ratio is the strongest negative and must still be reported, not skipped."""
    stdout, report = _run(_trace_dir({7: [60, 61, 62], 8: [58, 59, 63]}), _csv_probe_run())
    assert report["coverage"] == 1.0, "these features are in the deterministic top-K"
    assert report["conditions"]["target"]["ratio"] < 0.05, report["conditions"]["target"]
    assert report["h3_verdict"] == "falsified"
    assert "not enrichment" in stdout


def test_csv_unmeasured_nodes_are_not_counted_as_zero() -> None:
    mixed = {7: [5, 6], 8: [30, 31]}  # two responders, two the top-K never saw
    stdout, report = _run(_trace_dir(mixed), _csv_probe_run())
    assert report["measured_nodes"] + report["unmeasured_nodes"] == report["parent_nodes"]
    assert 0.0 < report["coverage"] < 1.0
    assert "unknown rather than zero" in stdout


# ------------------------------------------------------------------ units


def test_split_excludes_target_by_kind_or_key() -> None:
    nodes = [
        {"node_key": "L9:tau0.75:F5", "kind": "target"},
        {"node_key": "L8:tau0.75:F1", "kind": "parent"},
        {"node_key": "L7:tau0.75:F2"},  # legacy graph without kind: matched by key
    ]
    targets, parents = split_target_and_parents(nodes, "L7:tau0.75:F2")
    assert {t["node_key"] for t in targets} == {"L9:tau0.75:F5", "L7:tau0.75:F2"}
    assert [p["node_key"] for p in parents] == ["L8:tau0.75:F1"]


def test_random_control_matches_the_circuit_layer_composition() -> None:
    import random

    responses = load_probe_responses(_csv_probe_run())["target"]
    samples = random_control(responses, {7: 3, 8: 1}, draws=50, rng=random.Random(0))
    assert len(samples) == 50
    assert all(value >= 0 and np.isfinite(value) for value in samples)
    assert len(random_control(responses, {9: 2}, draws=20, rng=random.Random(1))) == 20


def test_monte_carlo_p_endpoints() -> None:
    assert monte_carlo_p(10.0, [1.0, 2.0, 3.0]) == 0.25, "a clear winner gets (1+0)/(3+1)"
    assert monte_carlo_p(0.0, [1.0, 2.0, 3.0]) == 1.0, "losing to every control is p=1"
    assert monte_carlo_p(1.0, []) is None


def test_score_nodes_splits_measured_from_unmeasured() -> None:
    responses = {(7, 5): 1.0}
    nodes = [
        {"node_key": "L7:tau0.5:F5", "layer": 7, "feature": 5},
        {"node_key": "L7:tau0.5:F99", "layer": 7, "feature": 99},
    ]
    measured, unmeasured = score_nodes(nodes, responses)
    assert [row["feature"] for row in measured] == [5]
    assert [row["feature"] for row in unmeasured] == [99]


def test_store_round_trip_and_tau_mapping() -> None:
    run = _store_probe_run()
    store = DeltaStore.open(run / "latents")
    assert store.layer_indices() == [7, 8, 9] and store.num_steps == STEPS
    assert store.step_for_tau(1.0) == 0 and store.step_for_tau(TAU) == 1 and store.step_for_tau(0.25) == 3
    assert store.prompts() == ["alt", "task"] and store.conditions() == ["null", "placebo", "target"]
    mean_t, n = store.mean_abs_delta(condition="target", prompt="task")
    assert n == 2 and mean_t.shape == (3, STEPS, N_FEATURES)
    assert abs(mean_t[2, 1, 5] - 3.0) < 1e-6 and abs(mean_t[2, 1, 20] - 0.02) < 1e-6 and mean_t[2, 1, 40] == 0.0
    mask = store.exercised_mask(condition="target", prompt="task")
    assert mask[0, 0].sum() == 32, "exactly the 32 active features are exercised"


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok {test.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    main()

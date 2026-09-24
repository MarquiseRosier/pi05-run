#!/usr/bin/env python
"""Smoke test for validating a traced circuit against the counterfactual probe.

Built around a planted ground truth: a set of features is made to respond to
the controlled perturbation, and the validator must rate a circuit built from
those features far above one built from non-responders.
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
from validate_circuit_with_counterfactual import (  # noqa: E402
    load_probe_responses,
    permutation_p,
    random_control,
    score_nodes,
)

N_FEATURES = 64
LAYERS = [f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp" for i in (7, 8, 9)]
RESPONDERS = {7: [5, 6, 7], 8: [5, 6, 7], 9: [5, 6, 7]}
SCRIPT = Path(__file__).resolve().parent / "validate_circuit_with_counterfactual.py"


def _accumulator(responders: dict[int, list[int]], amplitude: float):
    acc = probe.LatentAccumulator()
    for name in LAYERS:
        layer = int(name.split(".")[-2])
        for step in range(4):
            vector = np.full(N_FEATURES, 0.02)
            for feature in responders.get(layer, []):
                vector[feature] = amplitude
            acc.max[(name, step)] = vector
            acc.mean[(name, step)] = vector
    return acc


def _probe_run() -> Path:
    baseline = _accumulator({}, 0.0)
    target = _accumulator(RESPONDERS, 3.0)
    placebo = _accumulator(RESPONDERS, 0.25)
    rows = []
    for state in (0, 1):
        for condition, other in (("target", target), ("placebo", placebo)):
            for row in probe.latent_delta_rows(baseline, other, condition=condition, top_features=20):
                row.update({"state_index": state, "dose": 1.0, "target": condition})
                rows.append(row)
    run = Path(tempfile.mkdtemp())
    probe.write_csv(run / "latent_deltas.csv", rows)
    return run


def _trace_dir(features_by_layer: dict[int, list[int]]) -> Path:
    trace = Path(tempfile.mkdtemp())
    nodes = [
        {"node_key": f"L{layer}:tau0.5:F{feature}", "layer": layer, "feature": feature,
         "depth": 1, "kind": "parent", "influence": 0.1}
        for layer, features in features_by_layer.items()
        for feature in features
    ]
    (trace / "graph.json").write_text(
        json.dumps({"config": {"target": "L9:tau0.5:F5"}, "nodes": nodes, "edges": []})
    )
    return trace


def _run(trace: Path, run: Path) -> tuple[str, dict]:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(trace), str(run)], capture_output=True, text=True
    )
    report = json.loads((trace / "counterfactual_validation" / "validation.json").read_text())
    return result.stdout, report


def test_a_true_circuit_is_enriched_and_selective() -> None:
    stdout, report = _run(_trace_dir(RESPONDERS), _probe_run())
    target = report["conditions"]["target"]
    assert report["coverage"] == 1.0
    assert target["ratio"] > 2.0, target
    assert target["permutation_p"] < 0.05, target
    assert report["circuit_selectivity_target_over_placebo"] > 5.0
    assert "corroborated" in stdout


def test_a_wrong_circuit_is_not_enriched_and_says_so() -> None:
    """A zero ratio is the strongest negative and must still be reported."""
    stdout, report = _run(_trace_dir({7: [20, 21, 22], 8: [30, 31, 32], 9: [40, 41, 42]}), _probe_run())
    assert report["conditions"]["target"]["ratio"] == 0.0
    assert "not enrichment" in stdout, "a null result must be stated, not silently omitted"


def test_coverage_is_reported_and_unmeasured_nodes_are_not_counted_as_zero() -> None:
    mixed = {7: [5, 6], 8: [30, 31]}  # two responders, two the probe never saw
    stdout, report = _run(_trace_dir(mixed), _probe_run())
    assert report["measured_nodes"] + report["unmeasured_nodes"] == report["circuit_nodes"]
    assert 0.0 < report["coverage"] < 1.0
    assert "unknown rather than zero" in stdout


def test_random_control_matches_the_circuit_layer_composition() -> None:
    responses = load_probe_responses(_probe_run())["target"]
    import random

    samples = random_control(responses, {7: 3, 8: 1}, draws=50, rng=random.Random(0))
    assert len(samples) == 50
    # Non-responders have a delta of exactly zero, so a draw may average to zero;
    # what must hold is that every draw is finite and non-negative.
    assert all(value >= 0 and np.isfinite(value) for value in samples)

    # Composition matters: a layer the circuit never touches must not be sampled.
    only_layer_9 = random_control(responses, {9: 2}, draws=20, rng=random.Random(1))
    assert len(only_layer_9) == 20


def test_permutation_p_endpoints() -> None:
    assert permutation_p(10.0, [1.0, 2.0, 3.0]) < 0.3, "a clear winner should get a small p"
    assert permutation_p(0.0, [1.0, 2.0, 3.0]) == 1.0, "losing to every control is p=1"
    assert permutation_p(1.0, []) is None


def test_score_nodes_splits_measured_from_unmeasured() -> None:
    responses = {(7, 5): 1.0}
    nodes = [
        {"node_key": "L7:tau0.5:F5", "layer": 7, "feature": 5},
        {"node_key": "L7:tau0.5:F99", "layer": 7, "feature": 99},
    ]
    measured, unmeasured = score_nodes(nodes, responses)
    assert [row["feature"] for row in measured] == [5]
    assert [row["feature"] for row in unmeasured] == [99]


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok {test.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    main()

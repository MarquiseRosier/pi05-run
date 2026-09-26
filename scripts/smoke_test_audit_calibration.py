#!/usr/bin/env python
"""Smoke test for the circuit-audit calibration.

The calibration exists to make a negative verdict interpretable, so the tests
plant stores whose answer is known: one where selectivity is real and gradeable,
one where no feature is selective at all so nothing should be detectable, and
one where the audit would have to be blind for the negative to be excused.
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

from calibrate_circuit_audit import DETECTION_RATE, calibrate, feature_pool  # noqa: E402
from pi05_mi.counterfactual_store import DeltaStore  # noqa: E402

LAYERS = [f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp" for i in range(6)]
N_FEATURES = 256
STEPS = 4
SCRIPT = Path(__file__).resolve().parent / "calibrate_circuit_audit.py"


def _store(*, n_selective: int, selectivity: float = 40.0) -> Path:
    """A store where the first ``n_selective`` features per layer are object-selective.

    Every feature is exercised and responds to the target; only the selective
    ones respond far less to the placebo. So the pool is uniform in
    responsiveness and graded only in selectivity, which is what the audit has
    to pick out.
    """
    run = Path(tempfile.mkdtemp())
    store = DeltaStore.create(run / "latents", layer_names=LAYERS, num_steps=STEPS, num_features=N_FEATURES)
    rng = np.random.default_rng(0)
    for state in (0, 1):
        baseline, target, placebo = {}, {}, {}
        for name in LAYERS:
            for step in range(STEPS):
                base = np.full(N_FEATURES, 1.0, dtype=np.float32)
                d_t = np.abs(rng.normal(1.0, 0.05, N_FEATURES)).astype(np.float32)
                d_p = d_t.copy()
                d_p[:n_selective] = d_t[:n_selective] / selectivity
                baseline[(name, step)] = base
                target[(name, step)] = d_t
                placebo[(name, step)] = d_p
        store.write_block(state=state, noise=0, prompt="task", baseline_max=baseline,
                          deltas={("target", 1.0): target, ("placebo", 1.0): placebo})
    return run


def _calibrate(run: Path, **kw):
    store = DeltaStore.open(run / "latents")
    params = dict(max_layer=5, step=STEPS - 1, size=8, levels=[0.0, 0.5, 1.0],
                  repeats=8, draws=400, prompt="task", alpha=0.05, seed=0)
    params.update(kw)
    return calibrate(store, **params)


def _level(report, contamination):
    return next(r for r in report["levels"] if abs(r["contamination"] - contamination) < 1e-9)


def test_pool_excludes_the_target_layer_and_above() -> None:
    store = DeltaStore.open(_store(n_selective=32) / "latents")
    pool = feature_pool(store, max_layer=3, step=STEPS - 1, prompt="task")
    # Layers 0, 1 and 2 only.
    assert pool["n"] == 3 * N_FEATURES, pool["n"]
    assert set(np.unique(pool["position"]).tolist()) == {0, 1, 2}
    wider = feature_pool(store, max_layer=6, step=STEPS - 1, prompt="task")
    assert wider["n"] == 6 * N_FEATURES


def test_a_fully_selective_set_is_detected_and_a_random_one_is_not() -> None:
    """The two endpoints the calibration exists to establish."""
    report = _calibrate(_store(n_selective=64))
    top, bottom = _level(report, 1.0), _level(report, 0.0)
    assert top["detection_rate"] >= DETECTION_RATE, top
    assert top["enrichment_median"] > 2.0, top
    assert bottom["detection_rate"] <= 1 - DETECTION_RATE, bottom
    assert abs(bottom["enrichment_median"] - 1.0) < 0.35, bottom
    s = report["summary"]
    assert s["smallest_detectable_contamination"] is not None
    assert s["verdict"].startswith("the audit detects")


def test_detection_increases_with_content() -> None:
    report = _calibrate(_store(n_selective=64))
    levels = sorted(report["levels"], key=lambda r: r["contamination"])
    medians = [r["enrichment_median"] for r in levels]
    assert medians == sorted(medians), medians
    assert levels[-1]["enrichment_median"] > levels[0]["enrichment_median"] * 1.5


def test_a_store_with_no_selective_feature_calibrates_as_blind() -> None:
    """If nothing in the pool is selective, even 100% content is undetectable.

    This is the case that would otherwise excuse a negative verdict, and the
    calibration must name it rather than quietly reporting a low enrichment.
    """
    report = _calibrate(_store(n_selective=0))
    assert _level(report, 1.0)["detection_rate"] < DETECTION_RATE
    assert report["summary"]["verdict"].startswith("the audit is blind"), report["summary"]["verdict"]
    assert report["summary"]["smallest_detectable_contamination"] is None


def test_a_weakly_selective_pool_raises_the_detection_floor() -> None:
    """Halving the contrast must not make detection easier."""
    strong = _calibrate(_store(n_selective=64, selectivity=40.0))
    weak = _calibrate(_store(n_selective=64, selectivity=1.5))
    assert _level(strong, 1.0)["enrichment_median"] > _level(weak, 1.0)["enrichment_median"]


def test_too_small_a_pool_refuses_rather_than_guessing() -> None:
    store = DeltaStore.open(_store(n_selective=8) / "latents")
    try:
        calibrate(store, max_layer=5, step=STEPS - 1, size=10_000, levels=[1.0],
                  repeats=2, draws=100, prompt="task", alpha=0.05, seed=0)
    except SystemExit as exc:
        assert "need at least" in str(exc)
        return
    raise AssertionError("a pool smaller than twice the set size must raise")


def test_end_to_end_writes_its_artefacts() -> None:
    run = _store(n_selective=64)
    out = subprocess.run(
        [sys.executable, str(SCRIPT), str(run), "--target-layer", "5", "--size", "8",
         "--levels", "0,1.0", "--repeats", "6", "--draws", "300"],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    report = json.loads((run / "audit_calibration" / "audit_calibration.json").read_text())
    assert len(report["levels"]) == 2
    assert (run / "audit_calibration" / "audit_calibration.csv").exists()
    assert "verdict:" in out.stdout


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok {test.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    main()

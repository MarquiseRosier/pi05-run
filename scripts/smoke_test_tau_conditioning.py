#!/usr/bin/env python
"""Smoke test for the flow-time conditioning analysis.

Planted ground truth: three synthetic layers whose tau structure is known by
construction, plus a layer that is pure sampling noise. The analysis must
separate them, and must refuse to report the noisy one rather than dividing by
a near-zero reliability.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_tau_conditioning import (  # noqa: E402
    MIN_RELIABILITY,
    analyse_layer,
    corrected_correlation,
    reliability,
)

N_FEATURES = 512
N_TAU = 10
TAUS = [1.0 - i / N_TAU for i in range(N_TAU)]


def _counts(n: float) -> np.ndarray:
    return np.full(N_TAU, n, dtype=np.float64)


def _tau_independent(rng, noise: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Every flow time sees the same underlying code."""
    base = np.abs(rng.normal(1.0, 0.5, N_FEATURES))
    means = np.stack([base + rng.normal(0, noise, N_FEATURES) for _ in range(N_TAU)])
    stds = np.full_like(means, 0.5)
    return means, stds


def _smooth_drift(rng, step: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """A random walk along tau: the AR(1) null the analysis is built against."""
    base = np.abs(rng.normal(1.0, 0.5, N_FEATURES))
    rows, current = [], base
    for _ in range(N_TAU):
        current = current + rng.normal(0, step, N_FEATURES)
        rows.append(current.copy())
    means = np.stack(rows)
    return means, np.full_like(means, 0.5)


def _regime_switch(rng) -> tuple[np.ndarray, np.ndarray]:
    """Two unrelated codes, early tau and late tau: structure, not drift."""
    early = np.abs(rng.normal(1.0, 0.5, N_FEATURES))
    late = np.abs(rng.normal(1.0, 0.5, N_FEATURES))
    rows = []
    for i in range(N_TAU):
        w = i / (N_TAU - 1)
        rows.append((1 - w) * early + w * late + rng.normal(0, 0.01, N_FEATURES))
    means = np.stack(rows)
    return means, np.full_like(means, 0.5)


def test_reliability_is_one_without_noise_and_zero_when_all_noise() -> None:
    rng = np.random.default_rng(0)
    mean = rng.normal(0, 1.0, N_FEATURES)
    # No per-observation spread at all: the means are exact.
    assert reliability(mean, np.zeros(N_FEATURES), 100.0) == 1.0
    # Observed spread entirely explained by sampling noise: nothing is signal.
    # Var(mean) = 1, and std^2/count = 1 when std = 10, count = 100.
    assert reliability(mean, np.full(N_FEATURES, 10.0), 100.0) < 0.15
    # Degenerate inputs do not raise.
    assert reliability(mean, np.ones(N_FEATURES), 1.0) == 0.0
    assert reliability(np.zeros(2), np.ones(2), 100.0) == 0.0


def test_correction_divides_by_the_geometric_mean_of_reliabilities() -> None:
    assert corrected_correlation(0.5, 1.0, 1.0) == 0.5
    assert abs(corrected_correlation(0.49, 0.7, 0.7) - 0.7) < 1e-12
    assert abs(corrected_correlation(0.5, 1.0, 0.25) - 1.0) < 1e-12
    assert corrected_correlation(0.5, 0.0, 1.0) is None, "a zero ceiling must not be divided by"


def test_a_tau_independent_layer_reports_no_conditioning() -> None:
    """If every flow time carries the same code, the time MLP is doing nothing."""
    rng = np.random.default_rng(1)
    means, stds = _tau_independent(rng)
    row = analyse_layer(means, stds, _counts(1000), TAUS)
    assert row["usable"]
    assert row["adjacent_r"] > 0.99 and row["far_r"] > 0.99
    # Drift predicts ~1 and we observe ~1, so the ratio is ~1: no structure.
    assert row["structure_ratio"] > 0.9
    assert row["uses_conditioning"] is False


def test_smooth_drift_is_not_counted_as_conditioning() -> None:
    """The null the rule exists for: decorrelation that compounds is not structure."""
    rng = np.random.default_rng(2)
    means, stds = _smooth_drift(rng)
    row = analyse_layer(means, stds, _counts(1000), TAUS)
    assert row["usable"]
    assert row["far_r"] < row["adjacent_r"], "a walk decorrelates with lag"
    # The whole point: far_r is low, but only as low as compounding predicts.
    assert 0.6 < row["structure_ratio"] < 1.6, row["structure_ratio"]
    assert row["uses_conditioning"] is False, (
        "a low distant correlation alone must not count as using tau"
    )


def test_a_regime_switch_is_counted_as_conditioning() -> None:
    """Distant flow times far less correlated than their own drift predicts."""
    rng = np.random.default_rng(3)
    means, stds = _regime_switch(rng)
    row = analyse_layer(means, stds, _counts(1000), TAUS)
    assert row["usable"]
    assert row["adjacent_r"] > 0.95, "adjacent steps are still nearly identical"
    assert row["far_r"] < 0.5, "the endpoints are unrelated codes"
    assert row["structure_ratio"] < 0.5, row["structure_ratio"]
    assert row["uses_conditioning"] is True


def test_an_underpowered_layer_is_refused_not_corrected() -> None:
    """With too few observations the means are noise; say so rather than divide."""
    rng = np.random.default_rng(4)
    means, stds = _tau_independent(rng, noise=3.0)
    row = analyse_layer(means, np.full_like(means, 30.0), _counts(4), TAUS)
    assert row["reliability_min"] < MIN_RELIABILITY
    assert row["usable"] is False
    assert row["uses_conditioning"] is False, "an unusable layer must never claim a finding"


def test_a_dead_layer_is_reported_not_crashed() -> None:
    means = np.zeros((N_TAU, N_FEATURES))
    row = analyse_layer(means, np.zeros_like(means), _counts(100), TAUS)
    assert row["usable"] is False and row["features_live"] == 0
    assert "reason" in row


def test_live_feature_filter_ignores_never_firing_slots() -> None:
    """Features that never fire carry no tau information and must not dilute it."""
    rng = np.random.default_rng(5)
    means, stds = _regime_switch(rng)
    padded_means = np.concatenate([means, np.zeros((N_TAU, 4096))], axis=1)
    padded_stds = np.concatenate([stds, np.zeros((N_TAU, 4096))], axis=1)
    row = analyse_layer(padded_means, padded_stds, _counts(1000), TAUS)
    bare = analyse_layer(means, stds, _counts(1000), TAUS)
    assert row["features_live"] == N_FEATURES and row["features_total"] == N_FEATURES + 4096
    assert abs(row["far_r"] - bare["far_r"]) < 1e-9, "padding must not change the correlation"


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok {test.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    main()

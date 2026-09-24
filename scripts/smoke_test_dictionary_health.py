#!/usr/bin/env python
"""Smoke test for the dictionary-occupancy analysis.

The claim this analysis exists to police is that a never-fired count is not a
dead-feature count. The tests plant dictionaries whose true occupancy is known
and check that the script reports a dead fraction only when the sample size can
support one, and refuses with a required sample size when it cannot.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_dictionary_health import (  # noqa: E402
    DEFAULT_ALPHA,
    analyse,
    analyse_layer,
    observations_to_detect,
    zero_count_upper_bound,
)

N_FEATURES = 1000
N_TAU = 10


class _T:
    """Stands in for a torch tensor: the analysis only calls .numpy()."""

    def __init__(self, array):
        self._array = array

    def numpy(self):
        return self._array


def _payload(freqs_by_layer: dict[str, np.ndarray], observations: int) -> dict:
    names = list(freqs_by_layer)
    return {
        "layer_names": names,
        "observation_count": observations,
        "stats": {
            name: {
                f"{1.0 - i / N_TAU:.8f}": {"firing_frequency": _T(freqs_by_layer[name][i])}
                for i in range(N_TAU)
            }
            for name in names
        },
    }


def _layer(dead_fraction: float, rate: float = 0.05) -> np.ndarray:
    freqs = np.zeros((N_TAU, N_FEATURES))
    n_live = int(round(N_FEATURES * (1 - dead_fraction)))
    freqs[:, :n_live] = rate
    return freqs


def test_the_bound_matches_the_closed_form_and_the_rule_of_three() -> None:
    # (1-p)^n = alpha  =>  p = 1 - alpha**(1/n)
    for n in (10, 160, 1000, 13835):
        expected = 1.0 - DEFAULT_ALPHA ** (1.0 / n)
        assert abs(zero_count_upper_bound(n) - expected) < 1e-15
    # The familiar approximation: 3/n at alpha = 0.05.
    assert abs(zero_count_upper_bound(1000) - 3.0 / 1000) < 2e-4
    # The two functions are inverses of each other.
    for rate in (0.001, 0.00166, 0.01):
        n = observations_to_detect(rate)
        assert zero_count_upper_bound(n) <= rate
        assert zero_count_upper_bound(n - 1) > rate, "one fewer observation must not suffice"
    assert zero_count_upper_bound(0) is None
    assert observations_to_detect(0.0) is None and observations_to_detect(1.0) is None


def test_the_papers_own_feature_is_undetectable_at_160_observations() -> None:
    """The motivating case. F9970 fires at 0.00166; 160 observations cannot see it."""
    bound = zero_count_upper_bound(160)
    assert bound > 0.00166 * 10, f"bound {bound} should be an order of magnitude above the rate"
    assert observations_to_detect(0.00166) > 1500
    # The paper's own discovery run is large enough.
    assert zero_count_upper_bound(13835) < 0.00166


def test_a_small_run_refuses_to_report_a_dead_fraction() -> None:
    report = analyse(_payload({"m.layers.0.mlp": _layer(0.5)}, 160), reference_rate=0.00166, alpha=0.05)
    s = report["summary"]
    assert s["never_fired_fraction"] == 0.5, "the observed count is still reported"
    assert s["dead_is_identifiable"] is False
    assert "not identifiable" in s["verdict"]
    assert str(s["observations_to_detect_reference"]) in s["verdict"], "must name the sample size needed"


def test_a_large_run_reports_the_dead_fraction() -> None:
    report = analyse(_payload({"m.layers.0.mlp": _layer(0.5)}, 20000), reference_rate=0.00166, alpha=0.05)
    s = report["summary"]
    assert s["dead_is_identifiable"] is True
    assert s["verdict"].startswith("dead fraction 50.0%")


def test_a_fully_used_dictionary_reports_no_dead_features() -> None:
    report = analyse(_payload({"m.layers.0.mlp": _layer(0.0)}, 20000), reference_rate=0.00166, alpha=0.05)
    s = report["summary"]
    assert s["features_never_fired"] == 0 and s["never_fired_fraction"] == 0.0
    assert s["dead_is_identifiable"] is True


def test_a_feature_firing_at_any_flow_time_counts_as_used() -> None:
    """Occupancy is about the dictionary, so one live flow time makes a feature live."""
    freqs = np.zeros((N_TAU, N_FEATURES))
    freqs[3, :100] = 0.2  # only one flow time, only 100 features
    row = analyse_layer(freqs)
    assert row["features_ever_fired"] == 100
    assert row["features_never_fired"] == N_FEATURES - 100


def test_expected_active_is_the_mean_over_flow_times_of_the_summed_rates() -> None:
    freqs = np.zeros((N_TAU, N_FEATURES))
    freqs[:, :50] = 0.5  # 50 features at rate 0.5 => 25 expected active, every flow time
    row = analyse_layer(freqs)
    assert abs(row["expected_active_per_observation"] - 25.0) < 1e-9
    assert abs(row["rate_median"] - 0.5) < 1e-9 and abs(row["rate_max"] - 0.5) < 1e-9


def test_per_layer_range_is_reported_across_uneven_layers() -> None:
    report = analyse(
        _payload({"m.layers.0.mlp": _layer(0.9), "m.layers.1.mlp": _layer(0.1)}, 20000),
        reference_rate=0.00166,
        alpha=0.05,
    )
    low, high = report["summary"]["never_fired_fraction_range"]
    assert abs(low - 0.1) < 1e-9 and abs(high - 0.9) < 1e-9
    assert abs(report["summary"]["never_fired_fraction"] - 0.5) < 1e-9, "pooled over both layers"


def test_a_run_without_an_observation_count_says_not_measured() -> None:
    report = analyse(_payload({"m.layers.0.mlp": _layer(0.5)}, 0), reference_rate=0.00166, alpha=0.05)
    assert report["summary"]["verdict"].startswith("not measured")
    assert report["summary"]["zero_count_upper_bound"] is None


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok {test.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    main()

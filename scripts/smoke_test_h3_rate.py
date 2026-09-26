#!/usr/bin/env python
"""Smoke test for the H3 rate summary.

The interval is the whole point of this script: it decides how much a null
result excludes, so its endpoints are checked against published
Clopper-Pearson values rather than against the implementation. The verdict
tests plant the three outcomes that must never be confused: a method that
works, a method that does not, and an audit too blind for either reading.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from summarize_h3_rate import (  # noqa: E402
    binomial_tail,
    normalize_feature_key,
    clopper_pearson,
    load_audits,
    load_calibrations,
    required_targets_for_bound,
    summarize,
)

SCRIPT = Path(__file__).resolve().parent / "summarize_h3_rate.py"


def _audit(tmp: Path, name: str, *, supported: bool, enrichment: float = 1.0,
           p: float = 0.5, parents: int = 40) -> Path:
    d = tmp / name / "counterfactual_validation"
    d.mkdir(parents=True)
    (d / "validation.json").write_text(json.dumps({
        # The validator writes traced_target; the reader must accept it, or the
        # rate table loses every target name to a None column.
        "traced_target": name,
        "parent_nodes": parents,
        "exercised_fraction": 0.9,
        "decision_stratum_nodes": 10,
        "strata": [{"selectivity": {"enrichment": enrichment, "monte_carlo_p": p}}],
        "selectivity": {"enrichment": enrichment},
        "specificity": 1.2,
        "h3_verdict": "supported: object-specific" if supported else "falsified: not enriched",
    }))
    return tmp / name


def _calibration(tmp: Path, name: str, *, floor, blind: bool = False) -> Path:
    d = tmp / name
    d.mkdir(parents=True)
    (d / "audit_calibration.json").write_text(json.dumps({
        "levels": [{"contamination": 0.0, "detection_rate": 0.05, "enrichment_median": 1.0}],
        "summary": {
            "smallest_detectable_contamination": floor,
            "verdict": ("the audit is blind" if blind else
                        f"the audit detects a set that is at least {floor}"),
        },
    }))
    return d


def test_interval_endpoints_match_published_values() -> None:
    """Hand-checked Clopper-Pearson values, not the implementation's own output."""
    low, high = clopper_pearson(1, 10)
    assert abs(low - 0.00253) < 5e-5 and abs(high - 0.44502) < 5e-5, (low, high)
    low, high = clopper_pearson(5, 10)
    assert abs(low - 0.18709) < 5e-5 and abs(high - 0.81291) < 5e-5, (low, high)
    # Symmetric about a half, as a binomial interval must be.
    a, b = clopper_pearson(3, 10)
    c, d = clopper_pearson(7, 10)
    assert abs(a - (1 - d)) < 1e-9 and abs(b - (1 - c)) < 1e-9


def test_zero_successes_is_not_a_degenerate_interval() -> None:
    """The case that decides a null H3, and the one a Wald interval gets wrong.

    A Wald interval at k=0 has zero width and would certify a rate of exactly
    zero from five targets. The exact interval closes on the analytic value
    ``1 - alpha**(1/n)`` instead.
    """
    for n in (5, 10, 20, 50):
        low, high = clopper_pearson(0, n, sided=1)
        assert low == 0.0
        assert abs(high - (1 - 0.05 ** (1 / n))) < 1e-6, (n, high)
        assert high > 0.0
    assert abs(clopper_pearson(0, 20)[1] - 0.16843) < 5e-5
    # And the mirror image at k = n.
    assert abs(clopper_pearson(20, 20)[0] - 0.83157) < 5e-5


def test_binomial_tail_is_exact() -> None:
    assert abs(binomial_tail(0, 4, 0.5, upper=False) - 1 / 16) < 1e-12
    assert abs(binomial_tail(4, 4, 0.5, upper=True) - 1 / 16) < 1e-12
    assert abs(binomial_tail(0, 4, 0.5, upper=True) - 1.0) < 1e-12
    assert abs(binomial_tail(2, 4, 0.5, upper=False) - 11 / 16) < 1e-12


def test_required_targets_is_the_number_the_notebook_prints() -> None:
    assert required_targets_for_bound(0.50) == 5
    assert required_targets_for_bound(0.15) == 19
    n = required_targets_for_bound(0.14)
    assert clopper_pearson(0, n, sided=1)[1] <= 0.14
    assert clopper_pearson(0, n - 1, sided=1)[1] > 0.14


def test_a_uniform_null_is_falsified_and_reads_against_the_floor() -> None:
    tmp = Path(tempfile.mkdtemp())
    audits = load_audits([_audit(tmp, f"t{i}", supported=False) for i in range(20)])
    cal = load_calibrations([_calibration(tmp, "cal", floor=0.25)])
    s = summarize(audits, cal)
    assert s["corroborated"] == 0 and s["audited"] == 20
    assert s["verdict"].startswith("falsified"), s["verdict"]
    assert "25%" in s["reading"] and "14%" in s["reading"], s["reading"]


def test_five_targets_cannot_falsify_however_flat_they_are() -> None:
    """The bound at n=5 straddles a half, so the honest verdict is inconclusive.

    This is the failure the pilot's single trace had in its strongest possible
    form, and the reason TRACE_N_TARGETS is not 5.
    """
    tmp = Path(tempfile.mkdtemp())
    audits = load_audits([_audit(tmp, f"t{i}", supported=False) for i in range(5)])
    s = summarize(audits, load_calibrations([_calibration(tmp, "cal", floor=0.25)]))
    assert s["clopper_pearson_95"]["high"] > 0.5
    assert s["verdict"].startswith("inconclusive"), s["verdict"]


def test_a_working_method_is_supported() -> None:
    tmp = Path(tempfile.mkdtemp())
    audits = load_audits([_audit(tmp, f"t{i}", supported=i < 8) for i in range(10)])
    s = summarize(audits, load_calibrations([_calibration(tmp, "cal", floor=0.25)]))
    assert s["corroborated"] == 8
    assert s["verdict"].startswith("supported"), s["verdict"]
    assert s["clopper_pearson_95"]["low"] > 0.05


def test_a_blind_audit_invalidates_the_rate_rather_than_falsifying_it() -> None:
    """A flat reading from a test that cannot see is not evidence of absence."""
    tmp = Path(tempfile.mkdtemp())
    audits = load_audits([_audit(tmp, f"t{i}", supported=False) for i in range(20)])
    cal = load_calibrations([_calibration(tmp, "good", floor=0.25),
                             _calibration(tmp, "bad", floor=None, blind=True)])
    s = summarize(audits, cal)
    assert s["verdict"].startswith("invalid"), s["verdict"]
    assert len(s["blind_pools"]) == 1


def test_the_weakest_pool_sets_the_floor() -> None:
    tmp = Path(tempfile.mkdtemp())
    cal = load_calibrations([_calibration(tmp, "a", floor=0.125),
                             _calibration(tmp, "b", floor=0.50)])
    assert cal["detection_floor"] == 0.50, "a claim can only be as strong as the worst pool"


def test_an_uncalibrated_null_says_so_instead_of_claiming_a_bound() -> None:
    tmp = Path(tempfile.mkdtemp())
    audits = load_audits([_audit(tmp, f"t{i}", supported=False) for i in range(20)])
    s = summarize(audits, load_calibrations([]))
    assert s["detection_floor"] is None
    assert "never calibrated" in s["reading"], s["reading"]


def test_failed_traces_are_counted_not_silently_dropped() -> None:
    tmp = Path(tempfile.mkdtemp())
    audits = load_audits([_audit(tmp, "t0", supported=False), tmp / "never_traced"])
    s = summarize(audits, load_calibrations([]))
    assert s["audited"] == 1 and s["failed"] == 1


def test_layer_padding_is_canonicalised() -> None:
    """The nominator writes L4 and the tracer writes L04.

    Joining the two tables on the raw string dropped every single-digit layer
    from the results table, which showed as a blank task and a missing
    selectivity rather than as an error.
    """
    assert normalize_feature_key("L4:tau0.1:F735") == "L04:tau0.1:F735"
    assert normalize_feature_key("L04:tau0.1:F735") == "L04:tau0.1:F735"
    assert normalize_feature_key("L14:tau0.7:F3807") == "L14:tau0.7:F3807"
    # Already-canonical and unrecognised inputs pass through unharmed.
    assert normalize_feature_key("") == ""
    assert normalize_feature_key("weird") == "weird"


def test_the_target_name_survives_either_spelling() -> None:
    tmp = Path(tempfile.mkdtemp())
    d = tmp / "c" / "counterfactual_validation"
    d.mkdir(parents=True)
    (d / "validation.json").write_text(json.dumps(
        {"target": "L5:tau0.1:F735", "h3_verdict": "falsified"}))
    # Either spelling is accepted, and both come back canonicalised.
    assert load_audits([tmp / "c"])[0]["target"] == "L05:tau0.1:F735"
    planted = _audit(tmp, "L9:tau0.1:F42", supported=False)
    assert load_audits([planted])[0]["target"] == "L09:tau0.1:F42"


def test_no_audit_is_untestable_not_falsified() -> None:
    s = summarize(load_audits([Path("/nonexistent")]), load_calibrations([]))
    assert s["verdict"].startswith("untestable"), s["verdict"]
    assert s["rate"] is None


def test_end_to_end_writes_its_artefacts() -> None:
    tmp = Path(tempfile.mkdtemp())
    dirs = [_audit(tmp, f"t{i}", supported=False) for i in range(20)]
    cal = _calibration(tmp, "cal", floor=0.25)
    out = subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, dirs), "--calibration", str(cal),
         "--source-policy", "previous-layer", "--output-dir", str(tmp / "h3")],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    report = json.loads((tmp / "h3" / "h3_rate.json").read_text())
    assert report["summary"]["audited"] == 20
    assert report["summary"]["source_policy"] == "previous-layer"
    assert (tmp / "h3" / "h3_rate.csv").exists()
    assert "verdict: falsified" in out.stdout


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok {test.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    main()

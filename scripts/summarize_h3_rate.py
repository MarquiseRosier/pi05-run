#!/usr/bin/env python
"""Turn a set of circuit audits into a rate with an exact interval.

One traced circuit can only ever produce one anecdote. H3 asks whether the
circuits a tracer produces carry the content their targets are selective for,
and that is a property of the method, so the quantity is a proportion over
several targets rather than a verdict on one.

Two things make the proportion readable. The audit's own calibration
(``calibrate_circuit_audit.py``) fixes how much content it could have seen, so
a null result says "below this much content-carrying" instead of "not
enriched". And a Clopper-Pearson interval on the proportion says how much a
null result actually excludes: 0 of 5 leaves the rate possibly as high as 52%,
0 of 20 caps it at 17%.

Reads each target's ``counterfactual_validation/validation.json`` and, if
given, the calibration reports; writes ``h3_rate.json`` and ``h3_rate.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

# A rate at or below this cannot be called a working method.
WORKS_HALF_THE_TIME = 0.5
# The audit's own false-positive rate; a rate must clear it to mean anything.
NOMINAL_FALSE_POSITIVE = 0.05


def _log_binom_pmf(i: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 0.0 if i == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if i == n else -math.inf
    return (
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
        + i * math.log(p) + (n - i) * math.log1p(-p)
    )


def binomial_tail(k: int, n: int, p: float, *, upper: bool) -> float:
    """P(X >= k) if ``upper`` else P(X <= k), for X ~ Binomial(n, p)."""
    lo, hi = (k, n) if upper else (0, k)
    return sum(math.exp(_log_binom_pmf(i, n, p)) for i in range(lo, hi + 1))


def _bisect(fn, target: float, *, decreasing: bool) -> float:
    """Solve fn(p) = target on [0, 1] for a monotone fn."""
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        above = fn(mid) > target
        if above == decreasing:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def clopper_pearson(k: int, n: int, *, alpha: float = 0.05, sided: int = 2) -> tuple[float, float]:
    """Exact binomial interval by inverting the binomial tail.

    Bisection on the exact tail rather than a beta quantile, so the only
    dependency is the standard library and the endpoints k=0 and k=n come out
    right instead of collapsing the way a Wald interval does.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError(f"k={k} outside 0..{n}")
    tail = alpha / 2 if sided == 2 else alpha
    low = 0.0 if k == 0 else _bisect(
        lambda p: binomial_tail(k, n, p, upper=True), tail, decreasing=False)
    high = 1.0 if k == n else _bisect(
        lambda p: binomial_tail(k, n, p, upper=False), tail, decreasing=True)
    if sided == 1:
        return (0.0, high) if k == 0 else (low, 1.0)
    return low, high


def load_audits(paths: Sequence[Path]) -> list[dict]:
    """Read one audit per target. Accepts a trace dir or the json itself."""
    rows = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            candidate = path / "counterfactual_validation" / "validation.json"
            path = candidate if candidate.exists() else path / "validation.json"
        if not path.exists():
            rows.append({"path": str(path), "status": "missing"})
            continue
        v = json.loads(path.read_text())
        head = ((v.get("strata") or [{}])[0].get("selectivity") or {})
        rows.append({
            "path": str(path),
            "status": "read",
            # The validator names it traced_target; accept either spelling so a
            # rename upstream shows as a missing name, not a silent None column.
            "target": v.get("traced_target") or v.get("target"),
            "parents": v.get("parent_nodes"),
            "exercised": v.get("exercised_fraction"),
            "head_nodes": v.get("decision_stratum_nodes"),
            "head_enrichment": head.get("enrichment"),
            "head_p": head.get("monte_carlo_p"),
            "full_enrichment": (v.get("selectivity") or {}).get("enrichment"),
            "specificity": v.get("specificity"),
            "verdict": v.get("h3_verdict", ""),
            "corroborated": str(v.get("h3_verdict", "")).startswith("supported"),
        })
    return rows


def load_calibrations(paths: Iterable[Path]) -> dict:
    """Collapse the calibration reports into a detection floor."""
    floors, blind, read = [], [], []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            path = path / "audit_calibration.json"
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        summary = report.get("summary", {})
        read.append(str(path))
        if str(summary.get("verdict", "")).startswith("the audit is blind"):
            blind.append(str(path))
            continue
        floor = summary.get("smallest_detectable_contamination")
        if floor is not None:
            floors.append(float(floor))
    return {
        "reports": read,
        # The weakest pool decides what a null result can claim.
        "detection_floor": max(floors) if floors else None,
        "blind_pools": blind,
    }


def summarize(audits: Sequence[dict], calibration: dict, *, alpha: float = 0.05) -> dict:
    read = [r for r in audits if r["status"] == "read"]
    k = sum(1 for r in read if r["corroborated"])
    n = len(read)
    floor = calibration.get("detection_floor")

    if not n:
        verdict = "untestable: no circuit was audited"
        ci = one_sided = {"low": None, "high": None}
    else:
        low, high = clopper_pearson(k, n, alpha=alpha, sided=2)
        ci = {"low": low, "high": high}
        low1, high1 = clopper_pearson(k, n, alpha=alpha, sided=1)
        one_sided = {"low": low1, "high": high1}
        if calibration.get("blind_pools"):
            verdict = (f"invalid: the audit is blind on {len(calibration['blind_pools'])} pool(s), "
                       "so a flat reading carries no information")
        elif low > NOMINAL_FALSE_POSITIVE:
            verdict = f"supported: {k} of {n} circuits corroborated, above the audit's own error rate"
        elif high < WORKS_HALF_THE_TIME:
            verdict = (f"falsified: at most {high:.0%} of circuits traced this way carry the content "
                       f"their targets are selective for")
        else:
            verdict = (f"inconclusive: {k} of {n} leaves the rate anywhere in "
                       f"[{low:.0%}, {high:.0%}]; more targets are needed")

    return {
        "corroborated": k,
        "audited": n,
        "failed": len(audits) - n,
        "rate": (k / n) if n else None,
        "clopper_pearson_95": ci,
        "clopper_pearson_95_one_sided": one_sided,
        "detection_floor": floor,
        "blind_pools": calibration.get("blind_pools", []),
        "verdict": verdict,
        "reading": (
            None if not n or k else
            f"fewer than {(one_sided['high'] or 0):.0%} of traced circuits are even "
            f"{floor:.0%} content-carrying" if floor is not None else
            f"fewer than {(one_sided['high'] or 0):.0%} of traced circuits are corroborated, "
            "but the audit was never calibrated so its sensitivity is unknown"
        ),
    }


def required_targets_for_bound(bound: float, *, alpha: float = 0.05) -> int:
    """Targets needed so that a null result caps the rate at ``bound``."""
    if not 0 < bound < 1:
        raise ValueError("bound must lie in (0, 1)")
    n = 1
    while clopper_pearson(0, n, alpha=alpha, sided=1)[1] > bound:
        n += 1
        if n > 100_000:
            raise RuntimeError("bound unreachable")
    return n


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audits", nargs="+", type=Path,
                        help="Trace directories, or validation.json paths, one per target.")
    parser.add_argument("--calibration", action="append", type=Path, default=[],
                        help="audit_calibration.json (or its directory). Repeatable.")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--source-policy", default=None,
                        help="Recorded in the report; the tracer arm these audits came from.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audits = load_audits(args.audits)
    calibration = load_calibrations(args.calibration)
    summary = summarize(audits, calibration, alpha=args.alpha)
    summary["source_policy"] = args.source_policy

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "h3_rate.json").write_text(
        json.dumps({"summary": summary, "calibration": calibration, "audits": audits}, indent=2))
    fields = ["target", "parents", "exercised", "head_nodes", "head_enrichment", "head_p",
              "full_enrichment", "specificity", "corroborated", "verdict", "status", "path"]
    with (args.output_dir / "h3_rate.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(audits)

    def fmt(value, spec=".3f"):
        return "n/a" if value is None else format(float(value), spec)

    print(f"\nH3 over {summary['audited']} audited circuits"
          f"{f' (sources: {args.source_policy})' if args.source_policy else ''}\n")
    print(f"{'target':<24}{'parents':>8}{'head n':>8}{'head enrich':>13}"
          f"{'p':>9}{'spec':>8}  verdict")
    for row in audits:
        if row["status"] != "read":
            print(f"{row['path'][-24:]:<24}{'':>46}  {row['status']}")
            continue
        print(f"{str(row['target']):<24}{row['parents'] or 0:>8}{row['head_nodes'] or 0:>8}"
              f"{fmt(row['head_enrichment']):>13}{fmt(row['head_p'], '.4f'):>9}"
              f"{fmt(row['specificity'], '.2f'):>8}  {str(row['verdict'])[:44]}")
    if summary["detection_floor"] is not None:
        print(f"\ndetection floor: the audit finds a set that is at least "
              f"{summary['detection_floor']:.0%} content-carrying")
    ci = summary["clopper_pearson_95"]
    if summary["audited"]:
        print(f"rate: {summary['corroborated']}/{summary['audited']} = {summary['rate']:.0%}, "
              f"95% CI [{ci['low']:.1%}, {ci['high']:.1%}]")
    if summary["reading"]:
        print(f"reading: {summary['reading']}")
    print(f"verdict: {summary['verdict']}")
    print(f"\nWrote {args.output_dir / 'h3_rate.json'} and h3_rate.csv")


if __name__ == "__main__":
    main()

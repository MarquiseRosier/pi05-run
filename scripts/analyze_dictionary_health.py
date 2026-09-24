#!/usr/bin/env python
"""E5a: how much of the transcoder dictionary does the policy actually use?

A sixteen-times expansion buys 16384 features per layer. How many of them are
real is a question the paper does not answer, and the obvious way to answer it
is wrong.

The obvious way is to count features that never fired. That count is not the
dead-feature count, it is an upper bound on it, and at small sample sizes the
bound is so loose as to be useless. A feature that fires once in every three
hundred observations will usually be seen zero times in a hundred, and the
paper's own case-study feature fires at a rate of 0.00166. So this script
reports the occupancy, and alongside it the question of whether the run is even
large enough for the occupancy to mean anything.

With no firings in ``N`` observations, the exact one-sided upper confidence
bound on a feature's true rate solves ``(1 - p)^N = alpha``, giving
``p_max = 1 - alpha**(1/N)``, the familiar rule of three for small ``p``. A run
can separate dead from rare only when that bound sits below the rate worth
detecting. Turned around, detecting a rate ``p`` needs about
``log(alpha)/log(1 - p)`` observations. Both are reported, so a run that cannot
support the claim says so and names the sample size that would.

Two caveats are built into how the numbers are labelled. Discovery observations
are consecutive frames from episodes, so a feature's firings are correlated
across them and the effective sample is smaller than the nominal one; the bound
is therefore optimistic. And discovery collapses the code by a maximum over
action positions, so a feature counts as firing if it fired at any of the fifty
tokens: these are per-observation rates, not per-token ones. For per-token L0,
see the companion analysis over a rollout's captured latents.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

# The paper's case-study feature L11:tau0.7:F9970 fires at this rate. A run that
# cannot detect a feature this rare cannot speak about the paper's own example.
DEFAULT_REFERENCE_RATE = 0.00166
DEFAULT_ALPHA = 0.05


def zero_count_upper_bound(n_observations: int, alpha: float = DEFAULT_ALPHA) -> float | None:
    """Largest firing rate consistent with never having been observed.

    Solves ``(1 - p)**n = alpha``. For small ``p`` this is the rule of three,
    ``p ~ 3/n`` at ``alpha = 0.05``.
    """
    if n_observations <= 0:
        return None
    return float(1.0 - alpha ** (1.0 / n_observations))


def observations_to_detect(rate: float, alpha: float = DEFAULT_ALPHA) -> int | None:
    """Observations needed before never-seen rules out a feature firing at ``rate``."""
    if not 0.0 < rate < 1.0:
        return None
    return int(math.ceil(math.log(alpha) / math.log(1.0 - rate)))


def _numeric_layer_index(name: str) -> int:
    digits = [part for part in str(name).split(".") if part.isdigit()]
    return int(digits[-1]) if digits else -1


def analyse_layer(frequencies: np.ndarray) -> dict[str, Any]:
    """Occupancy at one layer from its per-(tau, feature) firing frequencies.

    ``frequencies`` is [n_tau, d_features]. A feature counts as used when it
    fires at some flow time, so the per-feature rate is the maximum over tau.
    """
    per_feature = frequencies.max(axis=0)
    live = per_feature > 0
    n_live = int(live.sum())
    row: dict[str, Any] = {
        "features_total": int(per_feature.size),
        "features_ever_fired": n_live,
        "features_never_fired": int(per_feature.size - n_live),
        "never_fired_fraction": float(1.0 - n_live / per_feature.size),
        # Expected number of features active in a given observation, averaged
        # over flow times. An upper bound on per-token L0, since discovery takes
        # a maximum over the fifty action positions.
        "expected_active_per_observation": float(frequencies.sum(axis=1).mean()),
    }
    if n_live:
        rates = per_feature[live]
        row.update(
            {
                "rate_median": float(np.median(rates)),
                "rate_p90": float(np.quantile(rates, 0.9)),
                "rate_max": float(rates.max()),
                "fraction_firing_above_1pct": float((rates > 0.01).mean()),
            }
        )
    return row


def analyse(payload: dict[str, Any], *, reference_rate: float, alpha: float) -> dict[str, Any]:
    stats = payload["stats"]
    n_obs = int(payload.get("observation_count", 0))
    layers = []
    for name in payload["layer_names"]:
        by_tau = stats[name]
        keys = sorted(by_tau, key=float)
        freqs = np.stack([by_tau[k]["firing_frequency"].numpy().astype(np.float64) for k in keys])
        row = analyse_layer(freqs)
        row["layer"] = name
        row["layer_index"] = _numeric_layer_index(name)
        row["n_tau"] = len(keys)
        layers.append(row)
    layers.sort(key=lambda r: r["layer_index"])

    bound = zero_count_upper_bound(n_obs, alpha)
    needed = observations_to_detect(reference_rate, alpha)
    identifiable = bound is not None and bound < reference_rate

    total = sum(r["features_total"] for r in layers)
    never = sum(r["features_never_fired"] for r in layers)
    never_fractions = [r["never_fired_fraction"] for r in layers]

    summary = {
        "observations": n_obs,
        "alpha": alpha,
        "reference_rate": reference_rate,
        "zero_count_upper_bound": bound,
        "observations_to_detect_reference": needed,
        "dead_is_identifiable": bool(identifiable),
        "features_total": total,
        "features_never_fired": never,
        "never_fired_fraction": float(never / total) if total else None,
        "never_fired_fraction_range": (
            [float(min(never_fractions)), float(max(never_fractions))] if never_fractions else None
        ),
        "expected_active_per_observation_median": float(
            np.median([r["expected_active_per_observation"] for r in layers])
        )
        if layers
        else None,
        "decision_rule": (
            "the never-fired fraction is an upper bound on the dead fraction; it is reported as a "
            f"dead-feature count only when the {1 - alpha:.0%} upper bound on an unobserved "
            f"feature's rate falls below the reference rate {reference_rate:g}"
        ),
    }
    if not n_obs:
        summary["verdict"] = "not measured: the run records no observation count"
    elif identifiable:
        summary["verdict"] = (
            f"dead fraction {summary['never_fired_fraction']:.1%} of the dictionary, identifiable at "
            f"this sample size (an unseen feature fires at most {bound:.2%}, below the {reference_rate:g} "
            "reference)"
        )
    else:
        summary["verdict"] = (
            f"not identifiable at {n_obs} observations: {summary['never_fired_fraction']:.1%} never "
            f"fired, but an unseen feature could still fire at up to {bound:.2%}, which is "
            f"{bound / reference_rate:.0f}x the {reference_rate:g} reference rate. "
            f"Separating dead from rare needs about {needed} observations."
        )
    return {"summary": summary, "layers": layers}


def _quantile_from_histogram(histogram: np.ndarray, q: float) -> float:
    total = float(histogram.sum())
    if total <= 0:
        return float("nan")
    return float(min(int(np.searchsorted(np.cumsum(histogram), q * total)), histogram.size - 1))


def analyse_token_sparsity(payload: dict[str, Any]) -> dict[str, Any]:
    """Per-token L0 from a discovery run that recorded it.

    This is the statistic the word "sparse" refers to. The occupancy table
    bounds it from above by up to the number of action tokens, because a
    feature counts there if it fired at any token.
    """
    d_features = int(payload["d_features"])
    pooled = np.zeros(d_features + 1, dtype=np.int64)
    layers = []
    for name in payload["layer_names"]:
        stores = payload["token_sparsity"].get(name) or {}
        if not stores:
            continue
        hist = np.zeros(d_features + 1, dtype=np.int64)
        tokens = 0
        weighted = 0.0
        for entry in stores.values():
            h = entry["histogram"].numpy().astype(np.int64)
            hist += h
            tokens += int(entry["tokens"])
            weighted += float(entry["mean"]) * int(entry["tokens"])
        pooled += hist
        layers.append(
            {
                "layer": name,
                "layer_index": _numeric_layer_index(name),
                "tokens": tokens,
                "l0_mean": weighted / tokens if tokens else float("nan"),
                "l0_median": _quantile_from_histogram(hist, 0.5),
                "l0_p90": _quantile_from_histogram(hist, 0.9),
                "l0_max": float(np.flatnonzero(hist)[-1]) if hist.any() else float("nan"),
            }
        )
    layers.sort(key=lambda r: r["layer_index"])
    if not layers:
        return {"available": False}
    tokens = sum(r["tokens"] for r in layers)
    mean = sum(r["l0_mean"] * r["tokens"] for r in layers) / tokens
    means = [r["l0_mean"] for r in layers]
    return {
        "available": True,
        "d_features": d_features,
        "tokens": tokens,
        "l0_mean": mean,
        "l0_median": _quantile_from_histogram(pooled, 0.5),
        "l0_p90": _quantile_from_histogram(pooled, 0.9),
        "l0_max": float(np.flatnonzero(pooled)[-1]) if pooled.any() else float("nan"),
        "l0_mean_pct": 100.0 * mean / d_features,
        "layer_mean_range": [float(min(means)), float(max(means))],
        "layers": layers,
    }


def load_stats(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "stats" not in payload:
        raise SystemExit(f"{path} has no 'stats' block; is this a feature_stats.pt?")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("feature_dir", type=Path, help="A feature discovery directory, or a feature_stats.pt.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--reference-rate",
        type=float,
        default=DEFAULT_REFERENCE_RATE,
        help="The firing rate the run should be able to detect. Default: the paper's case-study feature.",
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    args = parser.parse_args()

    path = args.feature_dir if args.feature_dir.is_file() else args.feature_dir / "feature_stats.pt"
    if not path.exists():
        raise SystemExit(f"No feature_stats.pt at {path}")
    out_dir = args.output_dir or path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    report = analyse(load_stats(path), reference_rate=args.reference_rate, alpha=args.alpha)
    s = report["summary"]

    sparsity_path = path.parent / "token_sparsity.pt"
    if sparsity_path.exists():
        import torch

        report["token_sparsity"] = analyse_token_sparsity(
            torch.load(sparsity_path, map_location="cpu", weights_only=False)
        )
    else:
        report["token_sparsity"] = {
            "available": False,
            "reason": (
                "this discovery run predates per-token L0 recording; rerun discovery to get it, "
                "or the occupancy figures remain an upper bound on sparsity"
            ),
        }

    print(f"dictionary health, {s['observations']} observations\n")
    print(f"{'layer':>5} {'never fired':>12} {'of total':>9} {'median rate':>12} {'max rate':>9} {'E[active]':>10}")
    for row in report["layers"]:
        print(
            f"{row['layer_index']:>5} {row['features_never_fired']:>12} "
            f"{row['never_fired_fraction']:>8.1%} {row.get('rate_median', 0):>12.4f} "
            f"{row.get('rate_max', 0):>9.3f} {row['expected_active_per_observation']:>10.1f}"
        )
    print(
        f"\nnever fired overall: {s['features_never_fired']} of {s['features_total']} "
        f"({s['never_fired_fraction']:.1%}), per-layer range "
        f"{s['never_fired_fraction_range'][0]:.0%} to {s['never_fired_fraction_range'][1]:.0%}"
    )
    print(f"an unseen feature fires at most {s['zero_count_upper_bound']:.2%} ({1 - s['alpha']:.0%} bound)")
    print(f"reference rate {s['reference_rate']:g} needs about {s['observations_to_detect_reference']} observations")
    ts = report["token_sparsity"]
    if ts.get("available"):
        print(
            f"\nper-token L0: mean {ts['l0_mean']:.1f}, median {ts['l0_median']:.0f}, "
            f"p90 {ts['l0_p90']:.0f}, max {ts['l0_max']:.0f} of {ts['d_features']} "
            f"({ts['l0_mean_pct']:.2f}% of the dictionary, {ts['tokens']:,} tokens)"
        )
        print(f"per-layer mean L0 ranges {ts['layer_mean_range'][0]:.1f} to {ts['layer_mean_range'][1]:.1f}")
    else:
        print(f"\nper-token L0: not recorded. {ts.get('reason', '')}")

    print(f"\nrule: {s['decision_rule']}")
    print(f"verdict: {s['verdict']}")

    (out_dir / "dictionary_health.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fields = [
        "layer_index", "features_total", "features_ever_fired", "features_never_fired",
        "never_fired_fraction", "rate_median", "rate_p90", "rate_max",
        "fraction_firing_above_1pct", "expected_active_per_observation",
    ]
    with (out_dir / "dictionary_health.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report["layers"]:
            writer.writerow({k: row.get(k) for k in fields})
    print(f"\nWrote {out_dir / 'dictionary_health.json'} and {out_dir / 'dictionary_health.csv'}")


if __name__ == "__main__":
    main()

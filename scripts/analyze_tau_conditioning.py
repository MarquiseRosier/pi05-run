#!/usr/bin/env python
"""E4a: does the flow timestep change the sparse code, and by how much?

The transcoders are conditioned on the flow timestep tau, and a feature is
identified by the triple (layer, tau, index), so the node space of a circuit is
ten times the dictionary. Neither the value of that conditioning nor the cost of
that multiplication has been measured. This script measures both from a feature
discovery run, with no retraining and no GPU.

The quantity is the correlation, across features at one layer, between the
per-feature mean activation at one flow time and at another. Two contrasts
matter and they answer different questions.

*Adjacent* flow times say how redundant the node space is. If the code at
tau=1.0 and tau=0.9 is nearly the same vector, then L12:tau1.0:F7584 and
L12:tau0.9:F7584 are nearly the same node, and a circuit traced over both is
counting one thing twice.

*Distant* flow times say whether conditioning earns its place. If the code at
the first denoise step and at the last is the same, the time MLP is decoration.

Two things make this an experiment rather than a plot.

**A null model, so "low" has a meaning.** If the code simply drifts smoothly
along tau, each step decorrelating a little and independently, the correlation
at lag k is the adjacent correlation raised to the k. That is the AR(1)
prediction, and it is what "conditioning does nothing but smooth interpolation"
looks like. Observing a distant correlation far *below* that prediction means
the code changes with tau in a structured way, not by drift.

**A reliability ceiling, so noise is not mistaken for signal.** These are
correlations between means estimated from a finite number of observations, so
sampling noise attenuates every one of them toward zero, and a genuinely
identical pair of flow times would still not correlate at 1.0. The ceiling is
estimated per (layer, tau) by the classical correction: with mean_i and std_i
per feature over n observations, the noise variance of the estimated means is
E[std_i^2]/n and the reliability is 1 - E[std_i^2]/n / Var(mean_i). A
correlation divided by the geometric mean of the two reliabilities is the
disattenuated estimate. Reliability near zero means the run is too small to say
anything at that layer, which is reported rather than divided by.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

# Below this reliability the means are mostly sampling noise and a corrected
# correlation would be an artefact of dividing by a small number.
MIN_RELIABILITY = 0.2
# A layer counts as using its conditioning when the distant-tau correlation
# falls this far below what smooth drift alone predicts. One half is the point
# at which the departure is larger than the prediction itself.
STRUCTURE_RATIO = 0.5


def _numeric_layer_index(name: str) -> int:
    digits = [part for part in str(name).split(".") if part.isdigit()]
    return int(digits[-1]) if digits else -1


def reliability(mean: np.ndarray, std: np.ndarray, count: float) -> float:
    """Fraction of the across-feature variance in ``mean`` that is not noise.

    ``mean`` and ``std`` are per-feature statistics over ``count`` observations.
    The estimated means carry a sampling variance of ``std**2 / count``; what is
    left of the observed spread is signal. Returns a value in [0, 1], or 0 when
    the observed spread is entirely accounted for by noise.
    """
    if count <= 1 or mean.size < 2:
        return 0.0
    observed = float(np.var(mean, ddof=1))
    if observed <= 0.0:
        return 0.0
    noise = float(np.mean(np.square(std))) / float(count)
    return float(max(0.0, min(1.0, 1.0 - noise / observed)))


def corrected_correlation(observed: float, rel_a: float, rel_b: float) -> float | None:
    """Disattenuate a correlation by the reliability of both estimates."""
    ceiling = math.sqrt(max(rel_a, 0.0) * max(rel_b, 0.0))
    if ceiling <= 0.0:
        return None
    return float(observed / ceiling)


def analyse_layer(
    means: np.ndarray, stds: np.ndarray, counts: np.ndarray, taus: list[float]
) -> dict[str, Any]:
    """Flow-time structure at one layer.

    ``means`` and ``stds`` are [n_tau, d_features]; ``counts`` is [n_tau].
    Features that never fire at any flow time carry no information about tau and
    are dropped, so the correlation is over the layer's live dictionary.
    """
    live = (means != 0).any(axis=0)
    n_live = int(live.sum())
    out: dict[str, Any] = {
        "n_tau": len(taus),
        "features_total": int(means.shape[1]),
        "features_live": n_live,
    }
    if n_live < 2:
        out["usable"] = False
        out["reason"] = "fewer than two live features"
        return out

    m = means[:, live]
    s = stds[:, live]
    rels = [reliability(m[i], s[i], float(counts[i])) for i in range(len(taus))]
    corr = np.corrcoef(m)

    adjacent = [float(corr[i, i + 1]) for i in range(len(taus) - 1)]
    adjacent_corrected = [
        corrected_correlation(adjacent[i], rels[i], rels[i + 1]) for i in range(len(taus) - 1)
    ]
    far = float(corr[0, -1])
    far_corrected = corrected_correlation(far, rels[0], rels[-1])

    r_adj = float(np.mean(adjacent))
    # Smooth-drift null: independent decorrelation per step compounds.
    lag = len(taus) - 1
    far_predicted = float(r_adj**lag) if r_adj > 0 else 0.0
    structure = (far / far_predicted) if far_predicted > 0 else None

    usable = min(rels) >= MIN_RELIABILITY
    out.update(
        {
            "usable": usable,
            "reliability_min": float(min(rels)),
            "reliability_mean": float(np.mean(rels)),
            "adjacent_r": r_adj,
            "adjacent_r_min": float(np.min(adjacent)),
            "adjacent_r_corrected": (
                float(np.mean([v for v in adjacent_corrected if v is not None]))
                if any(v is not None for v in adjacent_corrected)
                else None
            ),
            "far_r": far,
            "far_r_corrected": far_corrected,
            "far_r_predicted_by_drift": far_predicted,
            "structure_ratio": structure,
            "uses_conditioning": bool(
                usable and structure is not None and structure < STRUCTURE_RATIO
            ),
            "redundancy_pairs": lag,
        }
    )
    return out


def load_stats(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "stats" not in payload:
        raise SystemExit(f"{path} has no 'stats' block; is this a feature_stats.pt?")
    return payload


def analyse(payload: dict[str, Any]) -> dict[str, Any]:
    stats = payload["stats"]
    layers = []
    for name in payload["layer_names"]:
        by_tau = stats[name]
        taus = sorted((float(t) for t in by_tau), key=float)
        keys = sorted(by_tau, key=float)
        means = np.stack([by_tau[k]["mean"].numpy().astype(np.float64) for k in keys])
        stds = np.stack([by_tau[k]["std"].numpy().astype(np.float64) for k in keys])
        counts = np.asarray([float(by_tau[k]["count"]) for k in keys], dtype=np.float64)
        row = analyse_layer(means, stds, counts, taus)
        row["layer"] = name
        row["layer_index"] = _numeric_layer_index(name)
        layers.append(row)
    layers.sort(key=lambda r: r["layer_index"])

    usable = [r for r in layers if r.get("usable")]
    using = [r for r in usable if r.get("uses_conditioning")]
    adj = [r["adjacent_r"] for r in usable]
    far = [r["far_r"] for r in usable]
    summary = {
        "observations": int(payload.get("observation_count", 0)),
        "layers": len(layers),
        "layers_usable": len(usable),
        "layers_using_conditioning": len(using),
        "layer_indices_using_conditioning": [r["layer_index"] for r in using],
        "adjacent_r_median": float(np.median(adj)) if adj else None,
        "adjacent_r_range": [float(np.min(adj)), float(np.max(adj))] if adj else None,
        "far_r_median": float(np.median(far)) if far else None,
        "far_r_range": [float(np.min(far)), float(np.max(far))] if far else None,
        "thresholds": {"min_reliability": MIN_RELIABILITY, "structure_ratio": STRUCTURE_RATIO},
        "decision_rule": (
            "a layer uses its time conditioning when its distant-tau correlation falls below "
            f"{STRUCTURE_RATIO} of what smooth drift predicts from its own adjacent-tau "
            "correlation; a layer is usable when the reliability of every flow time's mean "
            f"estimate is at least {MIN_RELIABILITY}"
        ),
    }
    if not usable:
        summary["verdict"] = "not measured: no layer has reliable mean estimates at this sample size"
    elif len(using) == 0:
        summary["verdict"] = "conditioning unused: no layer departs from smooth drift"
    elif len(using) == len(usable):
        summary["verdict"] = "conditioning used at every usable layer"
    else:
        summary["verdict"] = (
            f"conditioning used at {len(using)} of {len(usable)} usable layers "
            f"(indices {summary['layer_indices_using_conditioning']})"
        )
    return {"summary": summary, "layers": layers}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("feature_dir", type=Path, help="A feature discovery directory, or a feature_stats.pt.")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    path = args.feature_dir if args.feature_dir.is_file() else args.feature_dir / "feature_stats.pt"
    if not path.exists():
        raise SystemExit(f"No feature_stats.pt at {path}")
    out_dir = args.output_dir or path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    report = analyse(load_stats(path))
    s = report["summary"]

    print(f"flow-time conditioning, {s['observations']} observations, {s['layers']} layers\n")
    print(f"{'layer':>5} {'live':>6} {'rel':>6} {'adj r':>7} {'far r':>7} {'drift pred':>11} {'ratio':>7}  uses tau")
    for row in report["layers"]:
        if not row.get("usable"):
            print(f"{row['layer_index']:>5} {row.get('features_live', 0):>6} "
                  f"{row.get('reliability_min', 0):>6.2f}   (not reliable at this sample size)")
            continue
        ratio = row["structure_ratio"]
        print(
            f"{row['layer_index']:>5} {row['features_live']:>6} {row['reliability_min']:>6.2f} "
            f"{row['adjacent_r']:>7.3f} {row['far_r']:>7.3f} {row['far_r_predicted_by_drift']:>11.3f} "
            f"{('n/a' if ratio is None else f'{ratio:>7.3f}')}  {'yes' if row['uses_conditioning'] else 'no'}"
        )
    print(f"\nadjacent-tau r: median {s['adjacent_r_median']:.3f}, range {s['adjacent_r_range']}")
    print(f"distant-tau r:  median {s['far_r_median']:.3f}, range {s['far_r_range']}")
    print(f"\nrule: {s['decision_rule']}")
    print(f"verdict: {s['verdict']}")

    (out_dir / "tau_conditioning.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fields = [
        "layer_index", "features_live", "reliability_min", "adjacent_r", "adjacent_r_corrected",
        "far_r", "far_r_corrected", "far_r_predicted_by_drift", "structure_ratio", "uses_conditioning",
    ]
    with (out_dir / "tau_conditioning.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report["layers"]:
            writer.writerow({k: row.get(k) for k in fields})
    print(f"\nWrote {out_dir / 'tau_conditioning.json'} and {out_dir / 'tau_conditioning.csv'}")


if __name__ == "__main__":
    main()

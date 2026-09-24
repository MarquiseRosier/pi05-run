#!/usr/bin/env python
"""Recompute the H1/H2 decision metrics of a finished probe run.

The decision functions are pure: everything they need is in the run's
``counterfactual_summary.json`` (every measurement, and the per-cell prompt
grounding). When a decision rule is corrected after a run, this re-reads the
verdicts under the new rule without repeating the forward passes.

Perturbation-size statistics are *not* recomputed here: they are measured in
the probe, so a change to how they are measured needs a rerun.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from probe_pi05_transcoder_counterfactual import (  # noqa: E402
    _json_default,
    _print_decision_metrics,
    compute_decision_metrics,
    write_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--no-write", action="store_true", help="Print only; leave the run's files untouched.")
    args = parser.parse_args()

    summary_path = args.run_dir / "counterfactual_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"No counterfactual_summary.json in {args.run_dir}")
    summary = json.loads(summary_path.read_text())
    measurements = summary.get("measurements") or []
    if not measurements:
        raise SystemExit("The summary holds no measurements")
    grounding = summary.get("prompt_grounding_rel_l2") or []

    decision = compute_decision_metrics(measurements, grounding_values=grounding)
    decision["recomputed_from"] = str(summary_path)
    _print_decision_metrics(decision)

    if not args.no_write:
        summary["decision_metrics"] = decision
        summary_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
        (args.run_dir / "decision_metrics.json").write_text(
            json.dumps(decision, indent=2, default=_json_default), encoding="utf-8"
        )
        write_csv(
            args.run_dir / "h1_cells.csv",
            [{k: v for k, v in cell.items() if not k.startswith("per_layer_")} for cell in decision["h1"]["cells"]],
        )
        print(f"\nRewrote decision_metrics.json and h1_cells.csv in {args.run_dir}")


if __name__ == "__main__":
    main()

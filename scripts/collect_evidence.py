#!/usr/bin/env python
"""Print every decision quantity of a counterfactual batch, small enough to paste.

A finished batch is gigabytes, nearly all of it delta stores and rendered
frames. The evidence is not in those: it is a few kilobytes of JSON spread over
a run directory per task plus the aggregate, calibration and H3 artefacts. This
walks a batch directory and prints them as one compact report, and writes the
same content to ``evidence.json`` so it can be moved as a single small file.

Nothing here computes a verdict. Every number is read from the artefact that
recorded it, and a missing artefact is reported as missing rather than skipped,
because a stage that did not run is itself evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _f(value, spec=".3f", missing="n/a") -> str:
    if value is None or value is False:
        return str(value) if value is False else missing
    if value is True:
        return "True"
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def find_batch(root: Path) -> Path:
    """Accept the batch directory, or a parent holding several."""
    if (root / "aggregate").exists() or any(root.glob("task*_seed*")):
        return root
    candidates = [p for p in root.iterdir() if p.is_dir() and any(p.glob("task*_seed*"))]
    if not candidates:
        raise SystemExit(
            f"No batch found under {root}. Point this at the directory holding "
            "task*_seed*/ (and aggregate/, h3/)."
        )
    return max(candidates, key=lambda p: p.name)


def collect(batch: Path) -> dict:
    runs = sorted(batch.glob("task*_seed*"))
    evidence: dict[str, Any] = {
        "batch": batch.name,
        "path": str(batch),
        "manifest": _load(batch / "manifest.json"),
        "runs": [],
        "aggregate": _load(batch / "aggregate" / "aggregate.json"),
        "h3": _load(batch / "h3" / "h3_rate.json"),
        "calibrations": {},
        "missing": [],
    }
    for name, path in (("aggregate/aggregate.json", batch / "aggregate" / "aggregate.json"),
                       ("h3/h3_rate.json", batch / "h3" / "h3_rate.json")):
        if not path.exists():
            evidence["missing"].append(name)

    for cal_dir in sorted(batch.glob("h3/calibration_*")):
        report = _load(cal_dir / "audit_calibration.json")
        if report is not None:
            evidence["calibrations"][cal_dir.name] = {
                "levels": report.get("levels"), "summary": report.get("summary"),
                "config": report.get("config"),
            }

    for run in runs:
        decision = _load(run / "decision_metrics.json")
        summary = _load(run / "counterfactual_summary.json")
        provenance = _load(run / "provenance.json")
        nominated = _load(run / "nominated_targets.json")
        if decision is None:
            evidence["missing"].append(f"{run.name}/decision_metrics.json")
        h1 = (decision or {}).get("h1") or {}
        h2 = (decision or {}).get("h2") or {}
        evidence["runs"].append({
            "run": run.name,
            "config": (summary or {}).get("config"),
            "states_measured": (summary or {}).get("states_measured"),
            "states_skipped": (summary or {}).get("states_skipped"),
            "h1": {
                "verdict": h1.get("verdict"),
                "n_cells": len(h1.get("cells") or []),
                "pooled": h1.get("pooled"),
                "bootstrap_ci_95": h1.get("bootstrap_ci_95"),
                "spread": h1.get("spread"),
                "null_floor": h1.get("null_floor"),
                "null_floor_near_zero": h1.get("null_floor_near_zero"),
                "positive_control": h1.get("positive_control"),
                "dose_response": h1.get("dose_response"),
                "robustness": h1.get("robustness"),
                "cluster_spread": h1.get("cluster_spread"),
            },
            "h2": {
                "verdict": h2.get("verdict"),
                "grounding": h2.get("grounding"),
                "per_prompt": h2.get("per_prompt"),
                "referent_check": h2.get("referent_check"),
                "perturbation_action_effect": h2.get("perturbation_action_effect"),
            },
            "nominated": [r.get("feature_key") for r in ((nominated or {}).get("nominated") or [])],
            "nominated_detail": (nominated or {}).get("nominated"),
            "provenance": {
                "commit": ((provenance or {}).get("git") or {}).get("commit"),
                "dirty": ((provenance or {}).get("git") or {}).get("dirty"),
                "device": ((provenance or {}).get("device") or {}).get("gpu_name"),
                "checkpoint_sha256": ((provenance or {}).get("transcoder_checkpoint") or {}).get("digest"),
                "packages": (provenance or {}).get("packages"),
            },
        })
    return evidence


def render(evidence: dict) -> str:
    out: list[str] = []
    w = out.append
    w(f"BATCH {evidence['batch']}  ({len(evidence['runs'])} runs)")
    prov = next((r["provenance"] for r in evidence["runs"] if r["provenance"]["commit"]), None)
    if prov:
        w(f"  commit {prov['commit']} dirty={prov['dirty']}  device={prov['device']}  "
          f"checkpoint sha256 {(prov['checkpoint_sha256'] or '')[:16]}")
    if evidence["missing"]:
        w(f"  MISSING: {', '.join(evidence['missing'])}")

    w("\n== Per run ==")
    w(f"{'run':<20}{'cells':>6}{'Sel raw':>10}{'Sel adj':>9}{'A sel':>8}{'floor':>8}"
      f"{'moved':>7}  H1 / H2")
    for r in evidence["runs"]:
        p = r["h1"]["pooled"] or {}
        floor = (r["h1"]["null_floor"] or {}).get("max")
        moved = (r["h2"]["referent_check"] or {}).get("referent_moved")
        w(f"{r['run']:<20}{r['h1']['n_cells']:>6}{_f(p.get('sel_raw'), '.2f'):>10}"
          f"{_f(p.get('sel_adj_l2'), '.2f'):>9}"
          f"{_f((p.get('A_target') / p['A_placebo']) if p.get('A_placebo') else None, '.1f'):>8}"
          f"{_f(floor, '.3g'):>8}{str(moved):>7}  "
          f"{str(r['h1']['verdict'])[:18]} / {str(r['h2']['verdict'])[:24]}")
        if r["states_skipped"]:
            w(f"{'':<20}  skipped states: {r['states_skipped']}")

    agg = evidence["aggregate"]
    if agg:
        s = agg["summary"]
        w(f"\n== Pooled over {s['tasks']} tasks, {s['cells_total']} cells "
          f"({s.get('runs', '?')} runs) ==")
        w(f"{'quantity':<26}{'pooled':>10}{'task-cluster 95% CI':>26}")
        for key, label in (("sel_raw", "Sel raw"), ("sel_adj_l2", "Sel adj (decides)"),
                           ("sel_adj_px", "Sel adj, pixels")):
            ci = (s["bootstrap_ci_95"] or {}).get(key) or {}
            interval = "[" + _f(ci.get("low")) + ", " + _f(ci.get("high")) + "]"
            w(f"{label:<26}{_f((s['pooled'] or {}).get(key), '.3f'):>10}{interval:>26}")
        w(f"tasks with Sel adj > 1: {s['tasks_above_one']} of {s['tasks']}")
        if s.get("null_floor_violations"):
            w(f"NULL FLOOR VIOLATIONS: {s['null_floor_violations']}")
        w(f"H1 verdict: {s['verdict']}")
        w(f"H2 verdict: {s['h2_verdict']}")
        w("\n  per task: " + "  ".join(
            f"{r['task_id']}:{_f(r['sel_adj_l2'], '.2f')}" for r in agg["per_task"]))
        w("  H2 gate:  " + "  ".join(
            f"{r['task_id']}:{'moved' if r['referent_moved'] else 'stuck'}"
            for r in agg["h2_per_task"]))

    for name, cal in evidence["calibrations"].items():
        w(f"\n== Audit sensitivity: {name} ==")
        w(f"{'content':>9}{'members':>9}{'enrich median':>15}{'detected':>10}")
        for level in cal["levels"] or []:
            median = level.get("enrichment_median")
            w(f"{level['contamination']:>8.0%}{level['selective_members']:>9}"
              f"{('inf' if median is None else format(median, '.2f')):>15}"
              f"{level['detection_rate']:>9.0%}")
        w(f"  {cal['summary'].get('verdict')}")

    h3 = evidence["h3"]
    if h3:
        s = h3["summary"]
        w(f"\n== H3 over {s['audited']} audited circuits "
          f"(sources: {s.get('source_policy')}) ==")
        w(f"{'target':<22}{'parents':>8}{'exercised':>11}{'head':>6}{'enrich':>9}"
          f"{'p':>9}{'spec':>7}  verdict")
        for a in h3["audits"]:
            if a.get("status") != "read":
                w(f"{a.get('path', '')[-40:]:<22}{'':>50}  {a.get('status')}")
                continue
            w(f"{str(a.get('target')):<22}{a.get('parents') or 0:>8}"
              f"{_f(a.get('exercised'), '.1%'):>11}{a.get('head_nodes') or 0:>6}"
              f"{_f(a.get('head_enrichment'), '.3f'):>9}{_f(a.get('head_p'), '.4f'):>9}"
              f"{_f(a.get('specificity'), '.2f'):>7}  {str(a.get('verdict'))[:30]}")
        ci = s["clopper_pearson_95"]
        if s["audited"]:
            w(f"rate: {s['corroborated']}/{s['audited']} = {s['rate']:.0%}, "
              f"95% CI [{_f(ci['low'], '.1%')}, {_f(ci['high'], '.1%')}]")
        if s.get("detection_floor") is not None:
            w(f"detection floor: {s['detection_floor']:.0%}")
        if s.get("reading"):
            w(f"reading: {s['reading']}")
        w(f"H3 verdict: {s['verdict']}")
    return "\n".join(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("batch", type=Path,
                        help="The batch directory, or a parent holding several.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Where to write evidence.json (default: inside the batch).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch = find_batch(args.batch)
    evidence = collect(batch)
    print(render(evidence))
    out = args.output or (batch / "evidence.json")
    try:
        out.write_text(json.dumps(evidence, indent=2, default=str))
        size = out.stat().st_size / 1024
        print(f"\nWrote {out} ({size:.0f} KB) -- this one file carries every number above.")
    except OSError as exc:
        print(f"\nCould not write {out}: {exc}")


if __name__ == "__main__":
    main()

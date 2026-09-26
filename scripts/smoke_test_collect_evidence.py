#!/usr/bin/env python
"""Smoke test for the batch evidence collector.

The collector exists so a multi-gigabyte run can be read from a few kilobytes,
and its one job is to lose nothing on the way. The tests plant a batch whose
contents are known and check that every decision quantity survives, that a
stage which did not run is reported as missing rather than silently omitted,
and that the collector never computes a verdict of its own.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect_evidence import collect, find_batch, render  # noqa: E402

SCRIPT = Path(__file__).resolve().parent / "collect_evidence.py"


def _run_dir(batch: Path, task: int, seed: int = 1000, *, moved: bool = True,
             nominated: int = 2, skipped=None) -> Path:
    d = batch / f"task{task:02d}_seed{seed}"
    d.mkdir(parents=True)
    (d / "decision_metrics.json").write_text(json.dumps({
        "h1": {
            "verdict": "supported",
            "cells": [{}] * 96,
            "pooled": {"sel_raw": 8.0 + task, "sel_adj_l2": 4.0 + task, "sel_adj_px": 3.5,
                       "A_target": 0.4, "A_placebo": 0.04},
            "bootstrap_ci_95": {"sel_adj_l2": {"low": 2.0, "high": 6.0}},
            "spread": {"sel_adj_l2": {"mean": 4.0, "sd": 1.0, "min": 1.2, "max": 11.8}},
            "null_floor": {"mean": 0.0, "max": 0.0},
            "null_floor_near_zero": True,
            "positive_control": {"n": 8, "D": 12.0},
            "dose_response": {"target": {"elasticity_mean": 0.9, "monotone_in_dose": True,
                                         "rows": [{"dose": 0.5, "D": 4.0}]}},
            "robustness": {"sel_rel": 7.1, "agree_in_direction": True},
            "cluster_spread": {"n": 12, "unit": "state"},
        },
        "h2": {
            "verdict": "supported" if moved else "untestable: referent did not move",
            "grounding": {"mean": 0.73},
            "per_prompt": {"task": {"sel_raw": 8.0}, "alt": {"sel_raw": 0.4 if moved else 9.0}},
            "referent_check": {"behavioural_anchor_task": 17.0,
                               "behavioural_anchor_alt": 0.4 if moved else 9.7,
                               "referent_moved": moved},
            "perturbation_action_effect": 0.54,
        },
    }))
    (d / "counterfactual_summary.json").write_text(json.dumps({
        "config": {"suite": "libero_spatial", "task_id": task, "seed": seed},
        "states_measured": 12 - len(skipped or []),
        "states_skipped": skipped or [],
    }))
    (d / "provenance.json").write_text(json.dumps({
        "git": {"commit": "9115b25", "dirty": False},
        "device": {"gpu_name": "NVIDIA A100-SXM4-40GB"},
        "transcoder_checkpoint": {"digest": "b75ee195bbe11ede" + "0" * 48},
        "packages": {"lerobot": "0.6.1", "torch": "2.11"},
    }))
    (d / "nominated_targets.json").write_text(json.dumps({
        "nominated": [{"feature_key": f"L5:tau0.1:F{task}{i}", "target_z": 4.8,
                       "selectivity": 45.0, "layer_index": 5, "tau": 0.1}
                      for i in range(nominated)]
    }))
    return d


def _batch(n_tasks: int = 3, *, aggregate: bool = True, h3: bool = True,
           moved_upto: int = 99) -> Path:
    root = Path(tempfile.mkdtemp())
    batch = root / "20260925-120000"
    batch.mkdir(parents=True)
    for t in range(n_tasks):
        _run_dir(batch, t, moved=t < moved_upto, skipped=[{"state_index": 3}] if t == 1 else None)
    if aggregate:
        agg = batch / "aggregate"
        agg.mkdir()
        (agg / "aggregate.json").write_text(json.dumps({
            "summary": {
                "runs": n_tasks, "tasks": n_tasks, "cells_total": 96 * n_tasks,
                "pooled": {"sel_raw": 9.0, "sel_adj_l2": 5.0, "sel_adj_px": 4.2},
                "bootstrap_ci_95": {"sel_raw": {"low": 6.0, "high": 12.0},
                                    "sel_adj_l2": {"low": 2.4, "high": 7.7},
                                    "sel_adj_px": {"low": 2.0, "high": 6.4}},
                "tasks_above_one": n_tasks, "null_floor_violations": [],
                "decision_rule": "interval excludes 1",
                "verdict": "supported: the task-cluster interval excludes 1",
                "h2_verdict": f"testable on {min(moved_upto, n_tasks)} of {n_tasks}",
            },
            "per_task": [{"task_id": t, "n_cells": 96, "sel_raw": 9.0,
                          "sel_adj_l2": 5.0, "action_sel": 10.0} for t in range(n_tasks)],
            "h2_per_task": [{"task_id": t, "grounding": 0.73, "anchor_task": 17.0,
                             "anchor_alt": 0.4, "referent_moved": t < moved_upto,
                             "sel_task": 8.0, "sel_alt": 0.4} for t in range(n_tasks)],
        }))
    if h3:
        h3_dir = batch / "h3"
        (h3_dir / "calibration_task00_seed1000_L5").mkdir(parents=True)
        (h3_dir / "calibration_task00_seed1000_L5" / "audit_calibration.json").write_text(json.dumps({
            "config": {"size": 16},
            "levels": [{"contamination": 0.0, "selective_members": 0,
                        "enrichment_median": 0.95, "detection_rate": 0.05},
                       {"contamination": 0.25, "selective_members": 4,
                        "enrichment_median": 2.05, "detection_rate": 0.85},
                       {"contamination": 1.0, "selective_members": 16,
                        "enrichment_median": None, "detection_rate": 1.0}],
            "summary": {"smallest_detectable_contamination": 0.25,
                        "verdict": "the audit detects a set that is at least 25% content-carrying"},
        }))
        (h3_dir / "h3_rate.json").write_text(json.dumps({
            "summary": {"corroborated": 0, "audited": 20, "failed": 1, "rate": 0.0,
                        "clopper_pearson_95": {"low": 0.0, "high": 0.1684},
                        "clopper_pearson_95_one_sided": {"low": 0.0, "high": 0.1391},
                        "detection_floor": 0.25, "blind_pools": [],
                        "source_policy": "previous-layer",
                        "reading": "fewer than 14% of traced circuits are even 25% content-carrying",
                        "verdict": "falsified: at most 17% of circuits carry the content"},
            "calibration": {}, "audits": [
                {"status": "read", "target": "L5:tau0.1:F735", "parents": 120,
                 "exercised": 0.997, "head_nodes": 10, "head_enrichment": 0.92,
                 "head_p": 0.5652, "specificity": 1.02, "verdict": "falsified: not enriched",
                 "corroborated": False, "path": "x"},
                {"status": "missing", "path": "/tmp/never_traced/validation.json"},
            ]}))
    return batch


def test_every_decision_quantity_survives_collection() -> None:
    evidence = collect(_batch())
    assert len(evidence["runs"]) == 3
    run = evidence["runs"][0]
    # The quantities the paper's tables are filled from.
    for key in ("pooled", "bootstrap_ci_95", "spread", "null_floor", "positive_control",
                "dose_response", "robustness", "cluster_spread"):
        assert run["h1"][key] is not None, key
    for key in ("grounding", "per_prompt", "referent_check", "perturbation_action_effect"):
        assert run["h2"][key] is not None, key
    assert run["provenance"]["commit"] == "9115b25"
    assert run["provenance"]["checkpoint_sha256"].startswith("b75ee195")
    assert run["nominated"] == ["L5:tau0.1:F00", "L5:tau0.1:F01"]
    assert evidence["aggregate"]["summary"]["tasks"] == 3
    assert evidence["h3"]["summary"]["audited"] == 20
    assert len(evidence["calibrations"]) == 1


def test_a_stage_that_did_not_run_is_named_not_omitted() -> None:
    """A missing aggregate is evidence; silently printing the runs would hide it."""
    evidence = collect(_batch(aggregate=False, h3=False))
    assert evidence["aggregate"] is None and evidence["h3"] is None
    assert "aggregate/aggregate.json" in evidence["missing"]
    assert "h3/h3_rate.json" in evidence["missing"]
    assert "MISSING" in render(evidence)


def test_skipped_states_reach_the_report() -> None:
    """A state whose perturbation changed no pixels is dropped by the probe and
    must stay visible, or the cell count silently disagrees with the design."""
    evidence = collect(_batch())
    assert evidence["runs"][1]["states_skipped"]
    assert "skipped states" in render(evidence)


def test_the_rendered_report_is_small_enough_to_paste() -> None:
    text = render(collect(_batch(n_tasks=10)))
    assert len(text) < 12_000, len(text)
    assert text.count("\n") < 140


def test_the_report_carries_the_verdicts_verbatim() -> None:
    text = render(collect(_batch()))
    for expected in ("H1 verdict: supported", "H2 verdict: testable on 3 of 3",
                     "H3 verdict: falsified", "detection floor: 25%",
                     "the audit detects a set that is at least 25%"):
        assert expected in text, expected
    # Read, never re-derived: the rate string comes from the artefact's own fields.
    assert "0/20 = 0%" in text and "[0.0%, 16.8%]" in text


def test_a_stuck_referent_is_visible_per_run() -> None:
    text = render(collect(_batch(n_tasks=3, moved_upto=1)))
    assert text.count("False") >= 2, "tasks whose referent did not move must show as False"


def test_a_failed_audit_row_is_shown_not_dropped() -> None:
    text = render(collect(_batch()))
    assert "missing" in text, "a target that never traced must appear in the H3 table"


def test_find_batch_accepts_a_parent_and_picks_the_newest() -> None:
    batch = _batch()
    parent = batch.parent
    (parent / "20260101-000000").mkdir()
    _run_dir(parent / "20260101-000000", 0)
    assert find_batch(parent).name == batch.name
    assert find_batch(batch).name == batch.name


def test_a_directory_with_no_batch_refuses_rather_than_printing_nothing() -> None:
    empty = Path(tempfile.mkdtemp())
    try:
        find_batch(empty)
    except SystemExit as exc:
        assert "No batch found" in str(exc)
        return
    raise AssertionError("an empty directory must raise")


def test_end_to_end_writes_one_small_file() -> None:
    batch = _batch(n_tasks=10)
    out = subprocess.run([sys.executable, str(SCRIPT), str(batch)],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    path = batch / "evidence.json"
    assert path.exists()
    assert path.stat().st_size < 400_000, path.stat().st_size
    evidence = json.loads(path.read_text())
    assert len(evidence["runs"]) == 10
    assert "H3 verdict:" in out.stdout


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok {test.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    main()

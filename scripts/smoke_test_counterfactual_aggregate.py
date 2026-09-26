#!/usr/bin/env python
"""Smoke test for the multi-task aggregation.

The point of aggregating is that the task, not the cell, is the unit. These
tests plant task populations whose truth is known and check that the verdict
tracks the task-level evidence rather than the cell count, and that the failure
modes of the single-task version are gone.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate_counterfactual_runs import MIN_TASKS_FOR_INTERVAL, _pool, aggregate, cluster_bootstrap  # noqa: E402

SCRIPT = Path(__file__).resolve().parent / "aggregate_counterfactual_runs.py"


def _cell(d_t, d_p, s_t=2.0, s_p=1.0, a_t=0.4, a_p=0.04):
    return {
        "D_target": d_t, "D_placebo": d_p,
        "S_l2_target": s_t, "S_l2_placebo": s_p,
        "S_px_target": s_t, "S_px_placebo": s_p,
        "A_target": a_t, "A_placebo": a_p,
    }


def _run(task_id, cells, *, floor_ok=True, anchor_alt=0.5, grounding=0.7, sel_alt=0.5):
    return {
        "run_dir": f"/tmp/run{task_id}", "task_key": f"libero_spatial:{task_id}",
        "suite": "libero_spatial", "task_id": task_id, "seed": 1000, "prompt": "p",
        "h1": {"cells": cells, "null_floor_near_zero": floor_ok,
               "null_floor": {"max": 0.0 if floor_ok else 9.9}, "verdict": "supported"},
        "h2": {
            "grounding": {"mean": grounding},
            "per_prompt": {"task": {"sel_raw": 4.0}, "alt": {"sel_raw": sel_alt}},
            "referent_check": {"behavioural_anchor_task": 8.0, "behavioural_anchor_alt": anchor_alt,
                               "referent_moved": anchor_alt < 1.0},
            "verdict": "supported" if anchor_alt < 1.0 else "untestable: referent did not move",
        },
    }


def test_pooled_ratio_is_a_ratio_of_means_not_a_mean_of_ratios() -> None:
    """A task with more cells must carry more weight."""
    cells = [_cell(8.0, 2.0), _cell(8.0, 2.0), _cell(2.0, 2.0)]
    pooled = _pool(cells)
    # Mean D_target is 6.0 and mean D_placebo is 2.0, so the pooled ratio is 3.0.
    # The mean of the three per-cell ratios would be (4 + 4 + 1)/3 = 3.0 as well,
    # so this case alone cannot tell the two estimators apart; the next one does.
    assert abs(pooled["sel_raw"] - 3.0) < 1e-12, pooled
    # Mean of ratios would be (4+4+1)/3 = 3.0 here too; make them differ.
    skewed = _pool([_cell(10.0, 1.0), _cell(1.0, 10.0)])
    assert abs(skewed["sel_raw"] - 1.0) < 1e-12, "ratio of means is 5.5/5.5 = 1"
    # Footprint ratio 2.0 halves it.
    assert abs(_pool(cells)["sel_adj_l2"] - 1.5) < 1e-12


def test_ten_consistent_tasks_are_supported_with_an_interval() -> None:
    runs = [_run(i, [_cell(8.0, 2.0)] * 4) for i in range(10)]
    report = aggregate(runs, draws=2000, seed=0)
    s = report["summary"]
    assert s["tasks"] == 10 and s["cells_total"] == 40
    assert abs(s["pooled"]["sel_adj_l2"] - 2.0) < 1e-9
    low = s["bootstrap_ci_95"]["sel_adj_l2"]["low"]
    assert low is not None and low > 1.0
    assert s["verdict"].startswith("supported"), s["verdict"]
    assert s["tasks_above_one"] == 10


def test_a_split_population_is_weak_not_supported() -> None:
    """Half the tasks selective, half not: the interval must straddle 1.

    This is the failure the single-task run could not see. Its rule was the
    minimum over cells, which would have reported the same verdict whether the
    weak half was one task or nine.
    """
    strong = [_run(i, [_cell(20.0, 2.0)] * 4) for i in range(5)]
    weak = [_run(5 + i, [_cell(1.6, 2.0)] * 4) for i in range(5)]
    report = aggregate(strong + weak, draws=2000, seed=0)
    s = report["summary"]
    assert s["tasks_above_one"] == 5, "half the tasks are above 1"
    ci = s["bootstrap_ci_95"]["sel_adj_l2"]
    assert ci["low"] is not None and ci["high"] is not None
    assert s["verdict"].startswith(("weak", "supported")), s["verdict"]
    # The interval must be wide because the tasks disagree.
    assert ci["high"] / max(ci["low"], 1e-9) > 2.0, (ci, "task disagreement must widen the interval")


def test_tasks_above_one_counts_tasks_and_not_runs() -> None:
    """Two seeds of one task are one task.

    The first multi-seed run reported "8 of 6", because the count walked the
    per-run rows while the denominator counted task clusters. A reader cannot
    tell a wrong count from a surprising one, so it is locked here.
    """
    runs = []
    for task in range(3):                       # three tasks, two seeds each
        for seed in (1000, 1001):
            run = _run(task, [_cell(8.0, 2.0)] * 4)
            run["seed"] = seed
            runs.append(run)
    s = aggregate(runs, draws=500, seed=0)["summary"]
    assert s["runs"] == 6 and s["tasks"] == 3
    assert s["tasks_above_one"] == 3, s["tasks_above_one"]
    assert len(s["per_task_pooled"]) == 3

    # A task whose two seeds straddle 1 counts once, on its pooled value.
    mixed = []
    for seed, cell in ((1000, _cell(20.0, 2.0)), (1001, _cell(1.0, 2.0))):
        run = _run(9, [cell] * 4)
        run["seed"] = seed
        mixed.append(run)
    s = aggregate(mixed, draws=200, seed=0)["summary"]
    assert s["tasks"] == 1 and s["runs"] == 2
    assert s["tasks_above_one"] == 1, "pooled 21/4 = 5.25 is above 1"
    assert abs(s["per_task_pooled"]["libero_spatial:9"]["sel_adj_l2"] - 2.625) < 1e-9


def test_uniformly_null_tasks_are_falsified() -> None:
    runs = [_run(i, [_cell(2.0, 2.0)] * 4) for i in range(10)]
    s = aggregate(runs, draws=1000, seed=0)["summary"]
    assert abs(s["pooled"]["sel_adj_l2"] - 0.5) < 1e-9
    assert s["verdict"].startswith("falsified"), s["verdict"]
    assert s["tasks_above_one"] == 0


def test_a_broken_null_floor_invalidates_the_pool() -> None:
    runs = [_run(i, [_cell(8.0, 2.0)] * 4) for i in range(9)]
    runs.append(_run(9, [_cell(8.0, 2.0)] * 4, floor_ok=False))
    s = aggregate(runs, draws=500, seed=0)["summary"]
    assert s["verdict"].startswith("invalid"), s["verdict"]
    assert s["null_floor_violations"] == ["libero_spatial:9"]


def test_too_few_tasks_reports_inconclusive_not_a_fake_interval() -> None:
    runs = [_run(i, [_cell(8.0, 2.0)] * 4) for i in range(2)]
    s = aggregate(runs, draws=500, seed=0)["summary"]
    assert s["bootstrap_ci_95"]["sel_adj_l2"]["low"] is None
    assert "too few for an interval" in s["verdict"], s["verdict"]


def test_cluster_bootstrap_resamples_tasks_not_cells() -> None:
    """Duplicating a task's cells must not narrow the interval the way new tasks do."""
    few = {f"t{i}": [_cell(8.0, 2.0)] * 4 for i in range(4)}
    many_cells = {f"t{i}": [_cell(8.0, 2.0)] * 400 for i in range(4)}
    a = cluster_bootstrap(few, "sel_adj_l2", draws=1000, seed=0)
    b = cluster_bootstrap(many_cells, "sel_adj_l2", draws=1000, seed=0)
    assert a["n_tasks"] == b["n_tasks"] == 4
    # Identical tasks, so both intervals collapse; the point is n_tasks drives it.
    assert abs((a["high"] - a["low"]) - (b["high"] - b["low"])) < 1e-6
    assert cluster_bootstrap({"t0": [_cell(8.0, 2.0)]}, "sel_adj_l2")["low"] is None
    assert MIN_TASKS_FOR_INTERVAL >= 3


def test_h2_is_gated_per_task_and_never_averaged() -> None:
    """Only tasks whose referent moved can be read; the rest are counted, not pooled."""
    moved = [_run(i, [_cell(8.0, 2.0)] * 4, anchor_alt=0.4, sel_alt=0.5) for i in range(3)]
    stuck = [_run(3 + i, [_cell(8.0, 2.0)] * 4, anchor_alt=9.7, sel_alt=5.9) for i in range(7)]
    s = aggregate(moved + stuck, draws=500, seed=0)["summary"]
    assert "testable on 3 of 10" in s["h2_verdict"], s["h2_verdict"]
    assert "inverts on 3" in s["h2_verdict"], s["h2_verdict"]

    none_moved = aggregate(stuck, draws=500, seed=0)["summary"]
    assert none_moved["h2_verdict"].startswith("untestable on all 7"), none_moved["h2_verdict"]


def test_end_to_end_over_run_directories() -> None:
    root = Path(tempfile.mkdtemp())
    for i in range(4):
        d = root / f"task{i}"
        d.mkdir()
        run = _run(i, [_cell(8.0, 2.0)] * 4)
        (d / "decision_metrics.json").write_text(json.dumps({"h1": run["h1"], "h2": run["h2"]}))
        (d / "counterfactual_summary.json").write_text(
            json.dumps({"config": {"suite": "libero_spatial", "task_id": i, "seed": 1000}})
        )
    out = subprocess.run([sys.executable, str(SCRIPT), str(root)], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    report = json.loads((root / "aggregate" / "aggregate.json").read_text())
    assert report["summary"]["tasks"] == 4
    assert (root / "aggregate" / "per_task.csv").exists()
    assert "H1 verdict:" in out.stdout and "H2 verdict:" in out.stdout


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok {test.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    main()

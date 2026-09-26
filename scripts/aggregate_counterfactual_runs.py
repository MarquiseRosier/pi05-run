#!/usr/bin/env python
"""Pool counterfactual probe runs across tasks, with the task as the unit.

The single-task run decided H1 on the minimum adjusted selectivity over its
eight cells. That was a crutch for having four independent clusters, and it was
fragile: in the reported run the two states gave 1.3 and 11.8, so the verdict
turned on the weakest cell clearing 1.0 by a quarter.

Cells are not independent. Doses within a (state, draw) share a render and a
noise sample; draws within a state share a robot pose; states within a task
share a scene and a policy trajectory. The one unit that is exchangeable across
the design is the **task**, so that is what this resamples. The interval is a
cluster bootstrap: draw tasks with replacement, pool every cell belonging to the
drawn tasks, and recompute the estimand. With ten tasks it is coarse, and the
number of clusters is printed next to it.

The estimand is the pooled ratio of means rather than the mean of per-task
ratios, so a task contributes in proportion to the cells it actually produced.
A task where the object was occluded at most states therefore does not carry the
same weight as one where every state measured.

H2 is aggregated differently, because it is gated rather than averaged. Each
task either moved its behavioural referent or did not, and only the ones that
did can be read at all. The output says how many were testable and what they
said, which is the only honest summary when the gate fails on some scenes.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

MIN_TASKS_FOR_INTERVAL = 3


def load_runs(paths: list[Path]) -> list[dict[str, Any]]:
    """Read every probe run under the given directories."""
    runs = []
    for root in paths:
        candidates = (
            [root] if (root / "decision_metrics.json").exists() else sorted(root.glob("*/decision_metrics.json"))
        )
        for candidate in candidates:
            run_dir = candidate if candidate.is_dir() else candidate.parent
            decision = json.loads((run_dir / "decision_metrics.json").read_text())
            summary_path = run_dir / "counterfactual_summary.json"
            config = json.loads(summary_path.read_text()).get("config", {}) if summary_path.exists() else {}
            runs.append(
                {
                    "run_dir": str(run_dir),
                    "task_key": f"{config.get('suite', '?')}:{config.get('task_id', '?')}",
                    "suite": config.get("suite"),
                    "task_id": config.get("task_id"),
                    "seed": config.get("seed"),
                    "prompt": (config.get("prompt_variants") or {}).get("task"),
                    "h1": decision.get("h1", {}),
                    "h2": decision.get("h2", {}),
                }
            )
    return runs


def _pool(cells: list[dict[str, Any]]) -> dict[str, float | None]:
    """Pooled ratios over a set of cells: ratios of means, not means of ratios."""
    if not cells:
        return {"sel_raw": None, "sel_adj_l2": None, "sel_adj_px": None, "n_cells": 0}
    mean = lambda key: float(np.mean([c[key] for c in cells]))  # noqa: E731
    d_t, d_p = mean("D_target"), mean("D_placebo")
    raw = (d_t / d_p) if d_p > 0 else None
    out: dict[str, float | None] = {"sel_raw": raw, "n_cells": len(cells), "D_target": d_t, "D_placebo": d_p}
    for name in ("l2", "px"):
        s_t, s_p = mean(f"S_{name}_target"), mean(f"S_{name}_placebo")
        ratio = (s_t / s_p) if s_p > 0 else None
        out[f"sel_adj_{name}"] = (raw / ratio) if (raw is not None and ratio) else None
        out[f"S_{name}_ratio"] = ratio
    out["A_target"] = mean("A_target")
    out["A_placebo"] = mean("A_placebo")
    return out


def cluster_bootstrap(
    cells_by_task: dict[str, list[dict[str, Any]]], key: str, *, draws: int = 10000, seed: int = 0
) -> dict[str, Any]:
    """Percentile interval from resampling whole tasks with replacement."""
    tasks = sorted(cells_by_task)
    if len(tasks) < MIN_TASKS_FOR_INTERVAL:
        return {"n_tasks": len(tasks), "low": None, "high": None,
                "reason": f"fewer than {MIN_TASKS_FOR_INTERVAL} task clusters"}
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        picked = rng.integers(0, len(tasks), size=len(tasks))
        pooled_cells = [cell for i in picked for cell in cells_by_task[tasks[i]]]
        value = _pool(pooled_cells).get(key)
        if value is not None and np.isfinite(value):
            samples.append(value)
    if not samples:
        return {"n_tasks": len(tasks), "low": None, "high": None, "reason": "no finite resample"}
    arr = np.asarray(samples)
    return {
        "n_tasks": len(tasks),
        "draws": int(arr.size),
        "low": float(np.percentile(arr, 2.5)),
        "high": float(np.percentile(arr, 97.5)),
    }


def aggregate(runs: list[dict[str, Any]], *, draws: int, seed: int) -> dict[str, Any]:
    cells_by_task: dict[str, list[dict[str, Any]]] = {}
    per_task = []
    floor_violations = []
    for run in runs:
        h1 = run["h1"]
        cells = h1.get("cells") or []
        if not cells:
            continue
        cells_by_task.setdefault(run["task_key"], []).extend(cells)
        pooled = _pool(cells)
        if not h1.get("null_floor_near_zero", True):
            floor_violations.append(run["task_key"])
        per_task.append(
            {
                "task_key": run["task_key"],
                "task_id": run["task_id"],
                "seed": run["seed"],
                "n_cells": pooled["n_cells"],
                "sel_raw": pooled["sel_raw"],
                "sel_adj_l2": pooled["sel_adj_l2"],
                "sel_adj_px": pooled["sel_adj_px"],
                "action_sel": (pooled["A_target"] / pooled["A_placebo"]) if pooled["A_placebo"] else None,
                "null_floor_max": (h1.get("null_floor") or {}).get("max"),
                "verdict_single_task": h1.get("verdict"),
            }
        )
    per_task.sort(key=lambda r: (str(r["task_id"])))

    all_cells = [c for cells in cells_by_task.values() for c in cells]
    pooled = _pool(all_cells)
    ci = {key: cluster_bootstrap(cells_by_task, key, draws=draws, seed=seed)
          for key in ("sel_raw", "sel_adj_l2", "sel_adj_px")}
    above = [r for r in per_task if r["sel_adj_l2"] is not None and r["sel_adj_l2"] > 1.0]

    primary = pooled.get("sel_adj_l2")
    low = (ci["sel_adj_l2"] or {}).get("low")
    if floor_violations:
        verdict = f"invalid: the null floor is not near zero on {len(floor_violations)} task(s)"
    elif primary is None:
        verdict = "not measured"
    elif low is None:
        verdict = (
            f"inconclusive: pooled adjusted selectivity {primary:.2f} but only "
            f"{len(cells_by_task)} task cluster(s), too few for an interval"
        )
    elif low > 1.0:
        verdict = (
            f"supported: pooled adjusted selectivity {primary:.2f}, task-cluster 95% CI "
            f"[{low:.2f}, {ci['sel_adj_l2']['high']:.2f}] excludes 1 over {len(cells_by_task)} tasks"
        )
    elif primary > 1.0:
        verdict = (
            f"weak: pooled adjusted selectivity {primary:.2f} exceeds 1 but the task-cluster CI "
            f"[{low:.2f}, {ci['sel_adj_l2']['high']:.2f}] includes it"
        )
    else:
        verdict = f"falsified: pooled adjusted selectivity {primary:.2f} does not exceed 1"

    # H2 is gated, not averaged: a task counts only if its referent moved.
    h2_rows, testable = [], []
    for run in runs:
        h2 = run["h2"]
        check = h2.get("referent_check") or {}
        row = {
            "task_key": run["task_key"],
            "task_id": run["task_id"],
            "grounding": (h2.get("grounding") or {}).get("mean"),
            "anchor_task": check.get("behavioural_anchor_task"),
            "anchor_alt": check.get("behavioural_anchor_alt"),
            "referent_moved": check.get("referent_moved"),
            "sel_task": (h2.get("per_prompt", {}).get("task") or {}).get("sel_raw"),
            "sel_alt": (h2.get("per_prompt", {}).get("alt") or {}).get("sel_raw"),
            "verdict": h2.get("verdict"),
        }
        h2_rows.append(row)
        if check.get("referent_moved"):
            testable.append(row)
    h2_rows.sort(key=lambda r: str(r["task_id"]))
    inverted = [r for r in testable if r["sel_alt"] is not None and r["sel_alt"] < 1.0]
    if not testable:
        h2_verdict = (
            f"untestable on all {len(h2_rows)} tasks: no sibling prompt moved the behavioural referent"
        )
    else:
        h2_verdict = (
            f"testable on {len(testable)} of {len(h2_rows)} tasks; selectivity inverts on "
            f"{len(inverted)} of those"
        )

    return {
        "summary": {
            "runs": len(runs),
            "tasks": len(cells_by_task),
            "cells_total": len(all_cells),
            "pooled": pooled,
            "bootstrap_ci_95": ci,
            "tasks_above_one": len(above),
            "null_floor_violations": floor_violations,
            "decision_rule": (
                "the task is the resampling unit; H1 is supported when the task-cluster 95% CI on the "
                "pooled footprint-adjusted selectivity excludes 1, weak when only the point estimate "
                "does, falsified otherwise, and invalid if any task's null floor is not near zero"
            ),
            "verdict": verdict,
            "h2_verdict": h2_verdict,
        },
        "per_task": per_task,
        "h2_per_task": h2_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", type=Path, nargs="+", help="Probe run directories, or a parent of them.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    runs = load_runs(args.run_dirs)
    if not runs:
        raise SystemExit(f"No probe runs with decision_metrics.json under {args.run_dirs}")
    report = aggregate(runs, draws=args.draws, seed=args.seed)
    s = report["summary"]
    out_dir = args.output_dir or (args.run_dirs[0] / "aggregate")
    out_dir.mkdir(parents=True, exist_ok=True)

    def f(v, spec=".2f"):
        return "n/a" if v is None else format(float(v), spec)

    print(f"pooled over {s['tasks']} tasks, {s['runs']} runs, {s['cells_total']} cells\n")
    print(f"{'task':>6} {'cells':>6} {'Sel raw':>9} {'Sel adj':>9} {'action sel':>11}  single-task verdict")
    for row in report["per_task"]:
        print(f"{str(row['task_id']):>6} {row['n_cells']:>6} {f(row['sel_raw']):>9} "
              f"{f(row['sel_adj_l2']):>9} {f(row['action_sel']):>11}  "
              f"{(row['verdict_single_task'] or '')[:34]}")
    p, ci = s["pooled"], s["bootstrap_ci_95"]
    print(f"\npooled Sel raw     {f(p['sel_raw'])}   task-cluster 95% CI "
          f"[{f(ci['sel_raw'].get('low'))}, {f(ci['sel_raw'].get('high'))}]")
    print(f"pooled Sel adj     {f(p['sel_adj_l2'])}   task-cluster 95% CI "
          f"[{f(ci['sel_adj_l2'].get('low'))}, {f(ci['sel_adj_l2'].get('high'))}]")
    print(f"tasks with adjusted selectivity above 1: {s['tasks_above_one']} of {s['tasks']}")
    print(f"\nH1 rule: {s['decision_rule']}")
    print(f"H1 verdict: {s['verdict']}")

    print(f"\n{'task':>6} {'g':>7} {'anchor task':>12} {'anchor alt':>11} {'moved':>6}  H2")
    for row in report["h2_per_task"]:
        print(f"{str(row['task_id']):>6} {f(row['grounding'],'.3f'):>7} {f(row['anchor_task']):>12} "
              f"{f(row['anchor_alt']):>11} {str(row['referent_moved']):>6}  {(row['verdict'] or '')[:30]}")
    print(f"\nH2 verdict: {s['h2_verdict']}")

    (out_dir / "aggregate.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, rows in (("per_task.csv", report["per_task"]), ("h2_per_task.csv", report["h2_per_task"])):
        if rows:
            with (out_dir / name).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
    print(f"\nWrote {out_dir / 'aggregate.json'}, per_task.csv and h2_per_task.csv")


if __name__ == "__main__":
    main()

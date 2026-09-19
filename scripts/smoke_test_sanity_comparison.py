#!/usr/bin/env python
"""Smoke test for the transcoder sanity-comparison report.

The report's headline claim is a paired closed-loop comparison, so these checks
pin the parts that decide what it says: per-episode extraction from the
``lerobot-eval`` ``eval_info.json`` layout, the paired McNemar statistics, and a
verdict that stays correct when success rates are not 100%.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_pi05_transcoder_sanity_runs import (
    _per_episode_successes,
    _verdict,
    mcnemar_exact_p,
    paired_summary,
    significance_summary,
    summarize_run,
    wilson_interval,
)


def _eval_info(successes_by_task: dict[str, list[bool]], *, start_seed: int = 1000) -> dict:
    """Build an eval_info.json shaped like lerobot_eval.eval_policy_all output."""
    per_task = []
    all_successes: list[bool] = []
    for task_id, successes in successes_by_task.items():
        per_task.append(
            {
                "task_group": "libero_spatial",
                "task_id": task_id,
                "metrics": {
                    "per_episode": [
                        {
                            "episode_ix": i,
                            "sum_reward": float(ok),
                            "max_reward": float(ok),
                            "success": ok,
                            "seed": start_seed + i,
                        }
                        for i, ok in enumerate(successes)
                    ],
                    "aggregated": {"pc_success": 100.0 * sum(successes) / len(successes)},
                },
            }
        )
        all_successes.extend(successes)
    return {
        "per_task": per_task,
        "overall": {
            "pc_success": 100.0 * sum(all_successes) / len(all_successes),
            "n_episodes": len(all_successes),
            "eval_s": 12.5,
        },
    }


def _write_run(root: Path, name: str, info: dict) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "eval_info.json").write_text(json.dumps(info), encoding="utf-8")
    return run_dir


def test_per_episode_extraction_keys_on_task_and_seed() -> None:
    info = _eval_info({"0": [True, False], "1": [True, True]})
    results = _per_episode_successes(info)
    assert len(results) == 4
    assert results[("libero_spatial", "0", 1000)] is True
    assert results[("libero_spatial", "0", 1001)] is False
    assert results[("libero_spatial", "1", 1001)] is True


def test_per_episode_extraction_survives_a_missing_per_task_block() -> None:
    assert _per_episode_successes({}) == {}
    assert _per_episode_successes({"overall": {"pc_success": 50.0}}) == {}
    assert _per_episode_successes({"per_task": "not-a-list"}) == {}


def _summaries(original_info: dict, replace_info: dict):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        original = summarize_run("original-probe-sanity", _write_run(root, "orig", original_info))
        replace = summarize_run("replace-sanity", _write_run(root, "repl", replace_info))
        return original, replace


def test_identical_outcomes_give_perfect_agreement() -> None:
    info = _eval_info({"0": [True, False, True], "1": [True, True, False]})
    original, replace = _summaries(info, json.loads(json.dumps(info)))
    paired = paired_summary(original, replace)
    assert paired["n_paired"] == 6
    assert paired["discordant"] == 0
    assert paired["agreement_rate"] == 1.0
    assert paired["mcnemar_exact_p"] == 1.0
    assert paired["success_delta_pp"] == 0.0


def test_equal_totals_with_different_episodes_are_still_caught_as_discordant() -> None:
    """The aggregate rate hides this; the paired view must not."""
    original, replace = _summaries(
        _eval_info({"0": [True, False]}),
        _eval_info({"0": [False, True]}),
    )
    assert original.pc_success == replace.pc_success == 50.0
    paired = paired_summary(original, replace)
    assert paired["discordant"] == 2, "swapped outcomes must register as two discordant pairs"
    assert paired["agreement_rate"] == 0.0
    assert paired["original_only_success"] == 1
    assert paired["replace_only_success"] == 1
    # b == c, so McNemar cannot reject, but agreement_rate exposes the disagreement.
    assert paired["mcnemar_exact_p"] == 1.0


def test_one_sided_degradation_is_detected() -> None:
    original, replace = _summaries(
        _eval_info({"0": [True] * 10, "1": [True] * 10}),
        _eval_info({"0": [False] * 8 + [True] * 2, "1": [True] * 10}),
    )
    paired = paired_summary(original, replace)
    assert paired["original_only_success"] == 8
    assert paired["replace_only_success"] == 0
    assert paired["mcnemar_exact_p"] < 0.05
    assert paired["success_delta_pp"] == -40.0

    stats = significance_summary(original, replace)
    assert stats["primary_test"].startswith("McNemar")
    assert stats["primary_p"] < 0.05
    verdict = _verdict(original, replace, [], stats)
    assert "p < 0.05" in verdict


def test_verdict_handles_realistic_sub_100_percent_runs() -> None:
    """A 30-episode LIBERO run will not be 100%; the verdict must not call that a failure."""
    original, replace = _summaries(
        _eval_info({"0": [True] * 7 + [False] * 3}),
        _eval_info({"0": [True] * 7 + [False] * 3}),
    )
    stats = significance_summary(original, replace)
    verdict = _verdict(original, replace, [], stats)
    assert "Both runs failed" not in verdict
    assert "No detected closed-loop success difference" in verdict
    assert "70.0%" in verdict


def test_verdict_flags_a_worthless_comparison_when_both_arms_fail() -> None:
    original, replace = _summaries(
        _eval_info({"0": [False] * 5}),
        _eval_info({"0": [False] * 5}),
    )
    stats = significance_summary(original, replace)
    verdict = _verdict(original, replace, [], stats)
    assert "Both runs failed every episode" in verdict


def test_unmatched_episodes_are_counted_not_silently_dropped() -> None:
    original, replace = _summaries(
        _eval_info({"0": [True, True, True]}),
        _eval_info({"0": [True, True]}),
    )
    paired = paired_summary(original, replace)
    assert paired["n_paired"] == 2
    assert paired["unmatched_original"] == 1
    assert paired["unmatched_replace"] == 0


def test_mcnemar_and_wilson_match_known_values() -> None:
    assert mcnemar_exact_p(0, 0) == 1.0
    assert mcnemar_exact_p(3, 3) == 1.0
    # b=5, c=0 -> two-sided exact p = 2 * 0.5**5
    assert abs(mcnemar_exact_p(5, 0) - 2 * 0.5**5) < 1e-12
    low, high = wilson_interval(1, 1)
    assert abs(low - 0.2065) < 1e-3 and high == 1.0


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok {test.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    main()

# Pi0.5 Transcoder Equivalence — Handoff

Status as of 2026-09-19. Supersedes the earlier n=1 sanity handoff.

## Where things stand

The n=1 sanity check passed (original/probe and replace both succeeded on one
LIBERO spatial task). That result stands but proves almost nothing: a 1/1
success rate has a Wilson 95% CI of roughly [20.7%, 100%].

The work since then is about producing a statistic that can actually speak to
how much error the transcoder substitution adds. **No high-N run has been
executed yet.** The tooling is committed, tested, and pushed; the runs are the
next step and require Colab.

## Two statistics, deliberately separate

**1. Closed-loop success (paired).**
`scripts/compare_pi05_transcoder_sanity_runs.py` compares the two branch runs.
Both arms replay the same tasks from the same seeds — the runner never passes
`--seed`, so `cfg.seed` stays at its default of 1000 and `start_seed` restarts
at 1000 for *each* task. Episodes are therefore matched pairs, keyed on
`(task_group, task_id, seed)`. Seed alone would collide across tasks.

Primary test: **McNemar's exact** on the discordant pairs. Fisher's exact on
pooled counts is reported too, but it discards the pairing and is less
sensitive. The report also gives the per-episode agreement rate, which catches
what both aggregate tests miss: two runs can post identical success rates while
disagreeing on every episode.

**2. Same-observation action error (the "how much error" number).**
`scripts/eval_pi05_transcoder_action_equivalence.py` runs both modes on
identical dataset observations and passes the *same* flow-matching noise tensor
to each pass, so any surviving difference is attributable to the MLP
substitution rather than to sampling.

Headline metric: `executed_rel_l2` = `||Δ|| / ||original||` over the first
`n_action_steps` actions. It is scale-free, so it does not depend on normalized
action units.

Read it against `control_metrics`, produced by `--control-batches`, which
repeats the baseline against itself to establish the run-to-run nondeterminism
floor. Only the excess over that floor is transcoder error.

## What was wrong before, and is now fixed

Three defects in the first draft of the workflow. All would have silently
corrupted the 30-episode runs it was written for.

1. `_verdict` tested `pc_success == 100.0`. A realistic LIBERO run is never
   100%, so any real result would have been reported as "Both runs failed."
2. The comparison read only `overall`, discarding the per-episode outcomes
   `lerobot-eval` writes under `per_task -> metrics -> per_episode` — exactly
   the data a paired test needs.
3. The action probe relied on restoring RNG state between passes.
   `PI05Policy.predict_action_chunk` forwards `noise` straight through to
   `sample_actions`, so the noise is now injected explicitly.

Locked by `scripts/smoke_test_sanity_comparison.py` and
`scripts/smoke_test_action_equivalence.py` (18 checks total). One of them pins
the lerobot noise-injection API so an upstream change fails loudly instead of
silently unpairing the comparison.

Run them with:

```bash
python scripts/smoke_test_sanity_comparison.py
PYTHONPATH=src python scripts/smoke_test_action_equivalence.py
```

(`PYTHONPATH=src` is only needed where `pi05_mi` is not pip-installed.)

## Branches

| Branch | Role |
| --- | --- |
| `feature/marquise-transcoder-feature-inspection` | Neutral. Scripts, docs, tests. Merge source. |
| `feature/marquise-transcoder-sanity-original` | `RUN_MODE="probe"` — original behavior, transcoder logging on. |
| `feature/marquise-transcoder-sanity-replace` | `RUN_MODE="replace"` — action-expert MLPs substituted. |

Both run branches default to `TASK_IDS="[0,1,2,3,4,5,6,7,8,9]"`, `EPISODES=3`.
`n_episodes` is **per task**, so that is 10 × 3 = 30 rollouts per arm, 30 paired
comparisons. `CAPTURE_ACTIVATIONS=False` and `TRANSCODER_MAX_CHUNKS=1000` so the
longer run is not silently truncated.

Colab:

- original/probe: https://colab.research.google.com/github/MarquiseRosier/pi05-run/blob/feature/marquise-transcoder-sanity-original/notebooks/pi05_libero_transcoder_colab_jayden.ipynb
- replace: https://colab.research.google.com/github/MarquiseRosier/pi05-run/blob/feature/marquise-transcoder-sanity-replace/notebooks/pi05_libero_transcoder_colab_jayden.ipynb

## Next steps

1. Run both branch notebooks to completion (30 rollouts each).
2. Run the **Run Transcoder Equivalence Metrics And Sanity Comparison** cell
   from either notebook. It runs both scripts and renders the report inline.
3. Read, in this order:
   - `significance.paired.agreement_rate` and `mcnemar_exact_p`
   - `metrics.executed_rel_l2.mean` / `.p95` against
     `control_metrics.executed_rel_l2.mean`
   - `significance.paired.per_task` for task-specific regressions
4. If the paired test comes back non-significant, **do not report that as
   equivalence.** With 30 pairs, McNemar only detects fairly large effects.
   Report the agreement rate and the relative-L2 distribution as the substantive
   result, and state the minimum detectable effect if an equivalence claim is
   wanted.

## Open items

- Power is still modest at n=30. If a tighter bound is needed, raise `EPISODES`
  or add suites (`libero_object`, `libero_goal`, `libero_10`); the pairing key
  already includes `task_group`, so mixing suites is safe.
- No equivalence-test (TOST-style) bound is computed yet. The current output
  supports "no detected difference", not "equivalent within δ".
- `action_chunk_comparison.csv` compares closed-loop chunks *by chunk index*.
  Once the trajectories diverge those chunks describe different states, so treat
  it as descriptive only — the same-observation probe is the rigorous number.

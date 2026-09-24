<!-- Canonical copy: ~/.claude/skills/diagnose-fix-loop/experiments.md. This repo copy is for collaborators; keep them identical. -->

# Research Experiment Loop — hypothesis → definition → result shape → notebook

The notebook is the **last** thing written. Start from the code and the hypothesis gets fitted to
what the code happens to measure. Start from what would falsify the claim, define the measurement
that decides it, draw the table that reports it, and only then build the thing that fills the table.
The 10-point validity checklist at the end is the gate at two points: before any code (Phase 0) and
before any verdict is read (Phase 4).

Everything here was paid for on the Pi0.5 counterfactual probe (`pi05-run`); the worked mapping is at
the bottom.

---

## Phase 0 — Pre-registration (before any code)

Write `docs/experiments/<slug>/preregistration.md` (or the LaTeX section directly). It must contain,
in this order:

1. **Claim, plainly.** One paragraph a layperson can read: what is manipulated, what is measured, what
   would count as the claim being wrong.
2. **Hypotheses H1..Hn.** For each:
   - the statement;
   - the observation that falsifies it;
   - the **decision quantity** — one number or ratio, named, with its equation;
   - the **decision rule** with numeric thresholds, and a *data-independent* reason for every
     threshold (conventional, structural, or "the weakest rule that excludes X");
   - the **verdict set**, closed: e.g. `supported | weak | falsified | invalid` for a measured effect,
     `supported | partial | falsified | untestable` for a gated one, `supported | falsified |
     inconclusive` for a downstream audit. Every possible outcome maps to one string.
3. **Manipulation.** What changes; what is held fixed; how "held fixed" is *verified at run time*
   (re-render identity, exact revert, shape guard), not asserted.
4. **Controls.** All four, each with what it rules out:
   - **null floor** — the same input twice; the bound it must stay under (e.g. ≤ 5 % of the placebo);
   - **matched placebo** — matched on what, differing in what, and the honest name for the difference
     (e.g. "non-referent", not "irrelevant");
   - **positive control** — a manipulation *known* to matter, fixing the scale of the outcome;
   - **dose response** — the doses, the normaliser, the linearity statistic (elasticity);
   - **manipulation check** for every *gated* manipulation (a prompt swap, a mode switch): a
     quantity independent of the decision quantity that shows the manipulation did what it claims,
     not merely that it was processed. "The action changed" is processing; "the action re-anchored
     on the other object" is the manipulation. Without it the gate passes on words.
5. **Replication unit.** What a *cell* is; what cells share (render, noise draw, pose); the *cluster*
   that is independent; which spreads and intervals are reported and at which level; n for each.
6. **Outcome measure and alternatives.** The primary summary and why; at least two scale-free
   alternatives; the agreement rule ("robust only if all fall on the same side of 1").
7. **Selection versus test.** Everything that is *selected* (targets, features, examples): on which
   data; everything *tested*: on which data; a sentence proving no overlap.
8. **Guards that abort.** The list. Each must actually raise, not print.
9. **Provenance.** Commit + dirty flag, library versions, device, checkpoint content hash, seeds
   (derived per cell), full CLI config.
10. **Scope.** What is not claimed; the first extension that would widen it.

**Gate:** run the checklist (below) against this document. Every item is `met` or `residual, stated`
before a line of experiment code exists.

---

## Phase 1 — Result shape (still before code)

Draw every results table with placeholders, **and give every cell the artefact path that fills it**.

```latex
\begin{table}[t]
\caption{... Filled from \texttt{decision\_metrics.json} under \texttt{h1}: \texttt{pooled},
\texttt{null\_floor}, \texttt{positive\_control}, \texttt{spread}, \texttt{cluster\_spread},
\texttt{bootstrap\_ci\_95}, \texttt{robustness}.}
...
Null control (max)            & \RESULT{val} & ... \\
Positive control              & \RESULT{val} & ... \\
Treatment                     & \RESULT{val} & ... \\
Placebo                       & \RESULT{val} & ... \\
\hline
Ratio                         & \RESULT{val} & ... \\
\multicolumn{k}{l}{adjusted: pooled \RESULT{}; cells min/max \RESULT{}; clusters n=\RESULT{}; 95\% CI [\RESULT{},\RESULT{}]} \\
\multicolumn{k}{l}{robustness: alt-1 \RESULT{}, alt-2 \RESULT{}; direction agrees: \RESULT{yes or no}} \\
```

**Rule:** a table cell with no source key means the experiment does not yet measure it → back to
Phase 0. A key with no table cell means the code computes something the paper does not report → ask
why.

### Standard artefacts (one directory per run)

| Artefact | Contents |
|---|---|
| `decision_metrics.json` | per hypothesis: `cells`, `pooled`, `spread`, `cluster_spread`, `bootstrap_ci_95`, `null_floor`, `positive_control`, `robustness`, `dose_response`, `thresholds` (numeric), `decision_rule` (prose), `verdict` (one string from the closed set) |
| `<h>_cells.csv` | one row per cell, flat |
| `<name>_summary.json` | `config` (every CLI value), `provenance`, every measurement, shapes |
| `provenance.json` | git commit/dirty, python/platform, package versions, device, checkpoint sha256, argv |
| `observation_shapes.json` | tensor shapes handed to the model, checked on every pass |
| raw store (`latents/` etc.) | **every** raw vector, compressed. Never top-K as the only record — a truncated record turns measurements into bounds |
| `nominated_*.json` | any selection: `criteria` (numeric), `rejected` counts per criterion, the nominees. Downstream **reads** this; it never re-ranks |
| `validation.json` | any downstream audit: `mode`, exercised/coverage fraction, per-condition `circuit_mean`, `random_mean`, `ratio`, `monte_carlo_p`, `draws`, `h<k>_verdict` |
| images | baseline / treated / diff per condition, for eyes |

---

## Phase 2 — Scripts (the measurement; testable without the GPU)

- **Pure decision functions.** `compute_decision_metrics(measurements, ...)` takes plain dicts and
  returns the JSON. It has **no** I/O and no model. Its unit test uses a synthetic fixture whose
  answers are known *exactly* (ratios 4 and 2, elasticity 1, CI collapsing to a point) and asserts
  every verdict branch (`supported`, `weak`, `falsified`, `invalid`, `partial`, `untestable`).
- **Guards have tests that prove they abort.** A guard without a failing-path test is a print.
- **Store raw, compress, index.** A small store class with `write_block` / `open` / `mean_abs` /
  `exercised_mask` and a round-trip test.
- **Selection writes its rule.** `criteria` + `rejected` to disk. One rule, one place.
- **Seeds derive per cell** (`seed + 1009*state + draw`) so any single cell can be regenerated.
- **Provenance** is a module call, not a comment.
- **Weights-loaded guard** on every policy load: loaders that swallow errors exist.
- Smoke tests run on CPU in < 1 min each. Run the whole suite **twice** before pushing.
- Every CLI flag has a default that is the reported configuration ("the script has everything set").

---

## Phase 3 — The notebook (carries out the experiment; decides nothing)

Fixed cell order: `Controls → runtime/mount → install → clone → assets → checkpoint → one cell per
stage`. Rules:

- **Zero required inputs.** Defaults *are* the reported config. Every control that is a threshold is
  also written to the artefact by the script it is passed to.
- **No computation the scripts don't do.** The notebook streams a subprocess, then renders tables
  straight from the artefacts and prints `verdict` next to `decision_rule`. If a number is needed
  that only the notebook computes, move it into a tested script.
- **"What to read in the output, in order"** markdown before each stage cell: the guard lines first
  (`All keys loaded`, liveness, null floor `= 0`), then the decision block, then the selection.
- **Downstream reads upstream files.** A stage that needs a nominee reads `nominated_*.json`; if it
  is missing or empty it raises with the rejection counts. It never re-ranks.
- Kill the subprocess on interrupt; show the log tail on non-zero exit.
- Keep runtime honest: state the cell count and forward-pass count the defaults imply.

---

## Phase 4 — Run, read, fill (the loop)

1. Fresh run on the pushed HEAD; capture the run directory and `provenance.json → git.commit`. If
   `dirty=true`, the run is not the code you think it is.
2. **Read verdicts off the artefacts. Never re-derive them by hand.** If a verdict surprises you,
   inspect the cells and the images, then fix the *rule* or the *measurement* in code, with a test,
   and rerun — do not adjust the reading.
3. If `robustness.agree_in_direction` is false, report the disagreement; do not pick the summary that
   agrees.
4. A guard abort is a defect → the diagnose-fix loop applies (one root cause, one test, fresh run).
5. Fill placeholders from the named keys. Each `\RESULT{}` has exactly one source.
6. DONE = two clean runs whose verdicts agree.

---

## The 10-point validity checklist (gate at Phase 0 and Phase 4)

| # | Requirement | How to check |
|---|---|---|
| 1 | Falsifiable hypotheses; rules and thresholds fixed and justified before data | every threshold is a numeric field in the artefact **and** has a written data-independent reason |
| 2 | Isolated manipulation, verified at run time | identity / revert / restore / shape checks exist and abort; diff images written |
| 3 | Complete control set | null floor with a bound, matched placebo with an honest name, **positive control**, dose response with elasticity, a **manipulation check** on every gated manipulation |
| 4 | Correct replication unit, honest uncertainty | cell and cluster defined; dependence stated; spread at both levels; n next to every interval; the decision rule does not assume independence |
| 5 | Construct validity, robustness across summaries | primary summary justified; ≥ 2 scale-free alternatives; `agree_in_direction` flag |
| 6 | No circularity | the thing selected on data A is excluded from the test on data A; downstream selectors never read the test data |
| 7 | Guards abort, not warn | each guard has a failing-path test |
| 8 | Every outcome has a meaning | closed verdict set; gates for `untestable` / `inconclusive`; threats name the residual ambiguity of a null |
| 9 | Provenance | commit, dirty, versions, device, checkpoint hash, per-cell seeds, full config |
| 10 | Scope honesty | what is not claimed; first extension named |

A `partial` on 1, 3, 4, 5 or 9 is a Phase 0 defect. A `gap` on 6 or 7 invalidates the result.

---

## Worked mapping — the counterfactual probe (`pi05-run`)

| Phase | Where |
|---|---|
| 0 pre-registration | `docs/paper/counterfactual_probe_section.tex` §Hypotheses, §Method, §Verdicts (thresholds + reasons) |
| 1 result shape | same file, Tables `cf-main`, `cf-dose`, `cf-prompt`, `cf-layers`, `cf-validation`, each caption naming its keys |
| 2 scripts | `scripts/probe_pi05_transcoder_counterfactual.py` (`compute_decision_metrics`, guards, store), `report_pi05_counterfactual_features.py` (nomination → `nominated_targets.json`), `validate_circuit_with_counterfactual.py` (H3, parents only, exercised-matched null); `src/pi05_mi/{counterfactual_store,provenance,pi05_weights}.py`; `scripts/smoke_test_*.py` |
| 3 notebook | `notebooks/pi05_transcoder_counterfactual.ipynb` |
| 4 loop | `.loops/methodology-audit/rca.md`; checklist assessment `docs/paper/validity_checklist.md` |

## Anti-patterns this project paid for

- **Selection then test on the same evidence** — the trace target, nominated for responding, was scored inside the circuit it anchored.
- **Unseeded stochasticity reported as run-to-run spread** — the shared noise came from the global RNG.
- **Top-K as the only record** — placebo responses became bounds; circuit coverage became partial; the null became "top-K responders".
- **Two rules for one selection** — the report ranked by z with a placebo filter; the notebook re-ranked by depth without it.
- **A claimed abort that only warned** — the policy loader returned random weights on a failed load.
- **Tables labelled with a quantity the code did not compute** — "mean per-feature response" for an L2 layer norm.
- **Thresholds only in prose** — unfalsifiable in practice, because prose can be re-read.
- **Uncertainty over non-independent cells** without saying so.
- **No positive control** — a latent delta of 8.3 with nothing to scale it against.
- **A gate that checks processing, not the manipulation** — prompt grounding g = 0.73 passed while
  the action stayed 10× anchored on the original bowl; the sibling phrase was true of both bowls.
  The verdict read "partial" when it was "untestable".
- **A normaliser on part of the input** — perturbation size measured on one of two cameras; the
  static camera's footprint never changed between states while the response tripled.
- **A statistic that cannot distinguish the hypothesis from its nearest rival** — "the traced
  parents respond more than random" was 1.26x and significant, and equally 1.26x for an unrelated
  language manipulation. The rival hypothesis, "these are just responsive features", predicts the
  same observation. Ask of every decision statistic: what else would produce this number, and is
  that measured? Here the fix was a ratio of ratios plus a second manipulation as a specificity
  control.
- **A control that also falsifies the hypothesis's legitimate variants** — specificity against a
  language swap rejected "generically responsive" sets, but a real referent circuit integrating
  language produces the same pattern. Before making a control a falsifier, ask which *true*
  hypotheses it would also reject; if any, it is a qualifier.
- **Fixing the statistic but not the saturation** — the corrected ratio-of-ratios still had p at
  its floor over 1721 nodes. Where n is large, decide on a pre-registered head of the ranking
  where p is informative, and report the rest.
- **A saturated p-value read as strong evidence** — with 1721 nodes every p hit its floor of
  1/(M+1) regardless of effect size. Past a few hundred units, significance stops discriminating
  and only the effect size carries the verdict; say so next to the number.

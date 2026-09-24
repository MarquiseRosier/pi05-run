# Validity checklist for the counterfactual perturbation experiment

Ten requirements a research reviewer would hold this experiment to, tailored
to a controlled input intervention on a vision-language-action policy with a
sparse-transcoder readout. Each item states the requirement, why it matters
here, where the experiment meets it (file, function or artefact), and what
remains open. "Status" is after the changes of 2026-09-24.

| # | Requirement | Status |
|---|-------------|--------|
| 1 | Falsifiable hypotheses with decision rules and thresholds fixed and justified before data | met |
| 2 | Isolated manipulation, verified rather than assumed | met |
| 3 | Complete control set: null floor, matched placebo, positive control, dose response | met |
| 4 | Correct replication unit with honest uncertainty | met, small n disclosed |
| 5 | Construct validity of the outcome and robustness across summaries | met |
| 6 | No circularity between selection and test | met |
| 7 | Instrument checks that abort rather than warn | met |
| 8 | Every outcome, including a null, has a stated meaning | met |
| 9 | Provenance: seeds, config, code hash, versions, checkpoint hash | met |
| 10 | Scope honesty | met; generalisation untested by design |

## 1. Falsifiable hypotheses, rules fixed before data

**Requirement.** Each hypothesis names the observation that falsifies it, the
rule is applied by code rather than by eye, and every threshold has a stated
reason that does not depend on the data.

**Why here.** Selectivity ratios and enrichment scores can always be read as
"large enough"; without a fixed rule the verdict is the experimenter's.

**Evidence.** H1/H2 rules and thresholds: `compute_decision_metrics` in
`scripts/probe_pi05_transcoder_counterfactual.py`, written to
`decision_metrics.json` under `h1.decision_rule`, `h1.thresholds`,
`h2.decision_rule`, `h2.thresholds`. H3 rule: `h3_verdict` in
`scripts/validate_circuit_with_counterfactual.py`, written to
`validation.json` with `alpha` and `min_exercised_fraction`. Nomination rule:
`nominated_targets.json` → `criteria`. Rationale for each threshold: LaTeX
"Why these thresholds".

**Was.** Rules in code and prose; thresholds as constants only; no rationale.

## 2. Isolated manipulation

**Requirement.** Between the two arms of a pair, only the intended variable
differs, and this is checked in the run rather than asserted.

**Why here.** A closed-loop comparison would let trajectories diverge; a stale
render would give zero everywhere; an unrestored baseline would contaminate
every later pair.

**Evidence.** Frozen state with re-render only: `rerender_observation`
(`force_update=True`). Shared, seeded noise: `sample_shared_noise` with
`noise_seed_for`. Liveness check and exact-revert check before measurement;
exact-restore check after every state's perturbed renders (`residual`).
Shape guard on every pass (`shape_report`). Difference images written per
condition.

**Residual.** Target and placebo differ in screen position as well as
referent status; H2 addresses referent status, position is a stated
confound. The LaTeX now calls the placebo the non-referent instance, not an
irrelevant object, because the policy must discriminate against it.

## 3. Complete control set

**Requirement.** A null that measures the floor, a placebo matched on
everything but the variable of interest, a positive control that fixes the
scale of a manipulation known to matter, and a dose response.

**Why here.** Latent deltas are unitless; without a positive control a
response of 8.3 has no meaning. Without a matched placebo, "responds to the
bowl" and "responds to any recolour" are indistinguishable.

**Evidence.** Null: condition `null` in every cell; verdict `invalid` if its
maximum exceeds 5 % of the placebo response. Placebo: same-mesh sibling
instance chosen from the BDDL by `auto_select_targets`. Positive control:
condition `prompt_swap`, the latent response of the sibling prompt over
identical pixels, reported as `h1.positive_control` with target and placebo
as fractions of it. Dose response: `dose_response`, elasticity η with S the
image-difference norm, monotonicity per condition.

**Was.** No positive control.

## 4. Replication unit and uncertainty

**Requirement.** The unit over which uncertainty is computed is stated, its
dependence structure is acknowledged, and n is next to every interval.

**Why here.** Cells are (state × draw × dose). Doses share a render and a
noise draw; draws share a robot pose. Treating cells as independent
overstates precision.

**Evidence.** Cell-level spread and percentile bootstrap (`h1.spread`,
`h1.bootstrap_ci_95`, n reported); cluster-level spread over (state, draw)
with doses pooled within (`h1.cluster_spread`); the pre-registered rule uses
the minimum over cells, which needs no independence assumption. LaTeX
"Uncertainty" states the dependence.

**Residual.** With 2 states × 2 draws the clusters number four. The
intervals are coarse and say so. More states is the lever if tighter bounds
are needed.

## 5. Construct validity and robustness

**Requirement.** The headline statistic measures what the claim is about, and
the conclusion does not depend on which reasonable summary was chosen.

**Why here.** The pooled layer response D is an L2 over 16 384 features
averaged over 18 layers; layer 17 deltas run ~46 while layer 2 deltas run
~0.1, so D is mostly the deep layers.

**Evidence.** `h1.robustness`: ratio on the relative L2 (delta norm over
baseline norm per layer and step), geometric mean of per-layer
selectivities with range and count above 1, and `agree_in_direction`
across the three summaries. Per-layer table in `layer_selectivity.csv`.
Per-feature work uses |δ_i| and the layer-standardised z, never D.

**Was.** D only.

## 6. No circularity

**Requirement.** Nothing is selected on the same evidence it is tested with.

**Why here.** The trace target is nominated for responding to the probe; if
it were scored inside the circuit, enrichment would be guaranteed.

**Evidence.** `split_target_and_parents` excludes `kind == "target"` and the
configured target key; parents come from the tracer, which reads dataset
observations and replacement-model gradients and never reads probe output
(no reference to probe artefacts in `trace_pi05_transcoder_circuit.py` or
`collect_pi05_transcoder_features.py`). Locked by
`test_store_target_node_is_excluded_from_the_evidence`.

**Was.** Target node scored inside the circuit mean.

## 7. Instrument checks that abort

**Requirement.** A failed precondition stops the run; it does not print and
continue to a plausible number.

**Evidence.** Weight load: `assert_weights_loaded` after every
`make_policy` (probe, discovery, trace, train), on a recorder installed on
`PI05Policy.load_state_dict`. Liveness, exact revert, exact restore, zero-pixel
perturbation, shape drift: `RuntimeError` in the probe. Null floor above 5 %:
H1 `invalid` rather than decided. Nomination with no eligible feature:
raises in the notebook with the per-criterion rejection counts.

**Was.** A failed vision-weight load printed a warning and ran on random
weights.

## 8. Every outcome has a meaning

**Requirement.** Before the run, each possible result is mapped to a reading,
including the null and the untestable.

**Evidence.** H1: supported / weak / falsified / invalid. H2: supported /
partial / falsified / untestable (gated on g ≥ 0.01), plus "not applicable"
when H1 gives no selectivity to follow. H3: supported / falsified /
inconclusive (exercised fraction < ½), with unexercised parents identified
rather than unknown. LaTeX "Threats" states the residual ambiguity of a
non-responding exercised parent and the caveat that g shows the prompt was
processed, not that its referent was resolved.

## 9. Provenance

**Requirement.** A reported number can be tied to the exact code, libraries,
device and weights that produced it, and the run is reproducible from the
recorded seeds.

**Evidence.** `provenance.json` and `counterfactual_summary.json →
provenance`: git commit and dirty flag, Python, platform, versions of torch,
numpy, lerobot, transformers, mujoco, robosuite, libero, safetensors, CUDA
device, SHA-256 of the transcoder checkpoint, and argv. Seeds: env seed and
per-cell noise seeds (`noise_seed_for`). Config: `config` block with every
CLI value.

**Was.** Config and seed only; noise unseeded.

## 10. Scope honesty

**Requirement.** The claim is no wider than the sampling, and what a wider
claim would need is stated.

**Evidence.** LaTeX "Scope" and "Is colour the right variable?": one task,
one seed, small numbers of states, draws and doses; colour carries no task
information in this suite so behavioural nulls are expected. A second task
or object type would be the first extension.

## Open items that need the GPU run

- Fill the `\RESULT{}` placeholders from `decision_metrics.json`,
  `layer_selectivity.csv`, `nominated_targets.json`, `validation.json`,
  `observation_shapes.json` and `provenance.json`.
- Read H1, H2 and H3 off the recorded verdicts; do not re-derive them.
- If `h1.robustness.agree_in_direction` is false, report the disagreement
  rather than choosing the summary that agrees.

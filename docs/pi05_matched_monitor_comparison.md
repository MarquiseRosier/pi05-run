# Dense linear probes versus transcoder probes

## What this comparison asks

Does a frozen transcoder representation improve prediction of upcoming harmful behavior,
compared with a simple linear classifier reading the original computation at the same layer?
The transcoder is a representation, not itself a safe/unsafe classifier. Both arms need a
trained classifier. Neither arm changes the policy or demonstrates prevention by itself.

The matched comparison is implemented in the existing occlusion pilot. The notebook defaults
to requiring an action-expert checkpoint, so it cannot silently run only dense baselines.
This is CPU-tested experimental infrastructure; full Colab/GPU integration remains to be run.

## Training audit: paper versus repository

The supplied manuscript, lines 69-73, describes 18 **PaliGemma backbone** MLP transcoders,
4,096 features each, TopK sparsity with k=64, trained to reconstruct MLP outputs from inputs.
Lines 82-86 report an 8,192-feature dictionary and held-out reconstruction/action-change tests.
Lines 98-111 describe the separate logistic monitors. The supplied text does not give enough
training detail to reproduce those dictionaries: optimizer settings, complete dataset splits,
token selection, training schedule and checkpoint identity are not all specified.

The repository's `TimeConditionedTranscoder` instead targets **action-expert** MLPs. It uses
time-dependent scale/shift, a ReLU encoder, and a linear decoder. Its loss is:

```text
x_mod = x * (1 + scale(t)) + shift(t)
z = ReLU(W_enc x_mod + b_enc)
y_hat = W_dec z + b_dec
loss = E[sum_j (y_hat_j - y_j)^2] / sum_j Var(y_j) + lambda * E[sum_i |z_i|]
```

Here x/y are the original MLP input/output, t is denoising time, and z is the sparse code.
The defaults in `train_pi05_transcoders.py` are expansion factor 16 and lambda=1e-4.
Those are script defaults, **not verified settings of a particular trained checkpoint**.
The action-expert transcoder implementation is identical in the fetched
`origin/feature/akhidre-causal-validation` branch and this working branch at inspection time.

In plain language, the objective rewards copying the original layer using relatively few
active features. It does not reward recognizing impending harm or ensuring that each feature
has a human-readable meaning. Better reconstruction need not produce a better safety monitor.

Before interpreting a run, record the actual checkpoint hash, model/component, feature widths,
collection mode, sparsity penalty and training metrics. The current training checkpoint writer
stores `configs`, `collection_mode`, `lambda_l1`, `metrics_history`, and related settings.
It does not embed every training episode/split and command-line option; obtain those from the
original training command/config/log. Audit overlap with monitor test episodes. Exposure of
the frozen dictionary to all tasks is not the same as supervision on test outcomes, but limits
claims about a wholly unseen-task representation.

## Matched protocol now implemented

1. Keep the original VLA and checkpoint frozen, in **probe** mode. Do not replace MLP outputs.
2. Collect MLP input, original output and sparse code on the same action-expert calls at
   layers 0, 8 and 17. Check that all three have matching call counts. Average action tokens,
   then average denoising calls, identically across arms.
3. The fast pilot uses fixed, label-independent Gaussian projections to at most 256 dimensions
   per representation/layer. This matches classifier width when all native widths exceed 256.
   Record native/saved widths and projection seeds in `matched_feature_schema.json`.
   Use `COMPARISON_WIDTH=0` in a **new run** to retain native widths; this costs more storage,
   RAM and fitting time and does not match parameter counts. Full-width confirmation and
   repeated projection seeds remain necessary before a broad negative representation claim.
4. Fit the same standardized, L2-regularized logistic regression for each arm. All arms include
   identical robot-state inputs. Fit only on training episodes. Use the same C and label window;
   this is a fixed pilot setting, not a completed hyperparameter search.
5. Keep every variant of one task/initial-state pair in the same split. Calibrate episode-level
   false stops on good calibration episodes only; test both familiar conditions and held-out screen.
6. Report overall and within-condition AUPRC/AUROC; early-warning numerators/denominators;
   false-stop rates; and **paired** starting-state-group bootstrap differences, sparse minus dense.
   The interval conditions on the fitted models and does not include training variability.
7. Report all three layers descriptively. Do not choose a winning layer from test results or
   present unadjusted per-layer intervals as a family-wide superiority test. A later confirmatory
   run must select layers/hyperparameters in grouped training validation and freeze the test.

The three matched arms per layer are:

| Family | Readout | Purpose |
| --- | --- | --- |
| `expert_input_lNN` | Original MLP input + state | Direct linear baseline at the transcoder's source |
| `expert_output_lNN` | Original MLP output + state | Original computation as the dense comparator |
| `expert_transcoder_lNN` | Frozen sparse code + state | Tests whether sparse representation helps the same classifier |

The legacy `dense` family is a sampled **backbone** representation and `transcoder` is sampled
action-expert code. They remain descriptive references, not a clean transcoder ablation. The
legacy `vision` family contains coarse RGB-grid statistics; it is **not** a sufficient CV baseline
for the planned image-encoder comparison. This change does not implement that separate experiment.

Pooling loses token/time structure. A negative result here applies to this pooled monitor,
not to every possible temporal or token-aware readout. Monitor times involve fresh policy forwards,
including when previously generated actions are still queued; all extra computation must count
in a future end-to-end latency study. Simulated time currently pauses during that computation.

## Run and collect the report

Open `notebooks/pi05_occlusion_monitor_pilot.ipynb` from the updated pilot branch. Set:

```python
RUN_NAME = "matched-probes-001"  # New folder; do not overwrite old pilot captures.
REQUIRE_MATCHED_COMPARISON = True
CHECKPOINT = "/content/drive/MyDrive/.../step_XXXXXX.pt"
COMPARISON_WIDTH = 256
```

Use the actual trained checkpoint path; the placeholder above is not a real artifact.
The notebook runs collection and fitting in one invocation and displays its report.
The 15-minute budget is a cap, not a guarantee of enough complete episodes or finished fitting.
If analysis times out after collection, rerun the fitter on persisted data without rerunning rollouts:

```bash
python scripts/fit_pi05_occlusion_monitor.py --run-dir /persistent/run/path --allow-partial
```

Only complete paired-condition groups enter partial analysis. Record time-budget selection bias.
Final fixed-plan runs should finish collection and omit `--allow-partial`.

Artifacts live in the existing persistent run folder:

- `manifest.json`: protocol and checkpoint hash; changed settings require a new directory.
- `matched_feature_schema.json`: component, layers, widths, pooling and projection seeds.
- `episodes/*/features.npz`: matched per-layer observations beside legacy features.
- `run_state.json`: phase/status and completed progress.
- `monitor_future_harm/<signature>/report.json` and `report.md`: metrics and paired comparisons.
- The same analysis directory stores fitted classifiers and held-out per-step scores.

The analysis signature includes actual episode/feature file contents, so changed data do not
silently reuse earlier fitted results. Old runs without matched captures cannot supply this
comparison retrospectively; the missing original expert features must be collected.

## When to adjust training

First evaluate the existing dictionary. If the sparse probe loses, inspect held-out reconstruction
error, active-feature counts, dead features and activation-distribution shift separately on clean,
occluded and random-box samples. Obtain these diagnostics before attributing a loss to training.

Then isolate one change at a time: training-distribution coverage, sparsity/width tradeoff, or
inference-trajectory collection versus random-timestep collection. Train candidate dictionaries
only on permitted training data and select settings without touching the final monitor test.
Do not simultaneously change dictionary, classifier capacity, pooling and layer selection.

Adding harm-label supervision to dictionary training is a distinct supervised method. Compare it
with an equally supervised dense alternative and report that it no longer tests the original
unsupervised transcoder. A failed steering experiment does not decide monitoring quality; a
successful monitor does not establish causal control or actual harm prevention.

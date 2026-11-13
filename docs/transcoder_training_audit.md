# Adversarial training-method review

Date: 2026-11-12. Scope: the new trajectory-training notebook and its generated definitions, not the historical checkpoint's training run.

**Verdict: the inspected implementation follows the intended trajectory-based, paper-informed training recipe. CPU checks support its mechanics. Real LeRobot/GPU execution and scientific usefulness remain unverified. Do not describe this as an exact DifFRACT reproduction or confirmed attribution improvement.**

## Verified mechanics

| Potential failure | What the implementation does / evidence |
|---|---|
| Fresh noise substituted at late timesteps | Calls native `predict_action_chunk` with initial noise once; hooks each actual `denoise_step`. A test explicitly checks that the second call receives evolved actions. |
| Training changes the teacher | Pretrained-load completeness guard; policy evaluation mode; frozen parameters; inference-only collection; an explicit runtime assertion rejects an unfrozen teacher. |
| Training on the student's own outputs | Collection returns original MLP outputs and records their inputs, outputs, and current time. Replacement is confined to fidelity checks. |
| Incomplete or misaligned buffers | Exact expected record and denoising-call counts; finite-value guards; float32 CPU buffers; no ring truncation. |
| Incorrect objective | Independent scalar-loss test checks minibatch population variance, squared error, summed L1, clipping, AdamW update, and decoder projection. |
| Wrong student architecture | Separate time-conditioned sparse model per layer; inherited time/initialization implementation inspected; strict explicit architecture loader. |
| Validation leakage into optimization | Disjoint episode/frame plans; optimizer only receives training records; selection uses validation; final evaluation uses test. |
| Restart silently changes training | Frozen protocol, dataset/teacher file hashes, model/Adam/cursor restoration, deterministic sampling, immutable snapshots. CPU interrupted/resumed export matches uninterrupted export. |
| Validation interruption duplicates updates | Committed optimizer state and separate validation marker; retry validation without repeating updates. |
| Notebook differs from tested source | Generated literal definitions are compared with the saved notebook and executed in a synthetic notebook namespace. |

## Paper alignment and deliberate departures

The alignment reference is [DifFRACT, Appendix E.1–E.3](https://arxiv.org/html/2606.15796v1#A5). Its separate [attribution validation, Appendix K.3](https://arxiv.org/html/2606.15796v1#A11.S3), is outside this training implementation.

| Component | This implementation |
|---|---|
| Core matches | Native denoising input/output collection; independent time-conditioned ReLU transcoders; 16× expansion; tied initialization followed by decoder normalization; minibatch variance-normalized MSE plus L1; AdamW, zero weight decay, cosine schedule. |
| Teacher / domain | Pi0.5 robot policy, recorded LIBERO observations, 18 action-expert MLPs. |
| Denoising | 10 Pi0.5 calls; VLA sinusoidal periods retained. |
| Sparsity setting | Fixed `λ=1e-4`; not established as optimal across layers. |
| Batch / buffer | 1,024 sampled records / 32,000-record buffer per layer. |
| Budget | 256 cycles but only 8.39M sampled records/layer—about 3.3% of the paper's approximately 256M. Equal cycle counts do not imply equal training. |
| Additional choices | Gradient clipping at 1; sequential dictionary optimization; validation-based checkpoint selection; episode holdout. |

## Review fixes

1. Reference variance now restores from the saved checkpoint. It is not recomputed after a restart. A regression test deletes the local cache and checks restoration without reference recollection.
2. Added an explicit frozen/evaluation-mode teacher assertion.
3. Strengthened the collection test to distinguish an evolved trajectory from fresh-noise inputs. Added an independent numerical optimizer/loss check.

## What could still go wrong scientifically?

- **Too little training:** the schedule is a finite budget, not a convergence guarantee. Inspect learning curves before choosing a separate longer run; do not silently extend a frozen run's schedule.
- **Wrong sparsity tradeoff:** low copying error with many active features may remain hard to interpret. A single lambda does not establish the best tradeoff. Check L0 and feature usage as well as nMSE.
- **Narrow evaluation:** 24 validation frames and 16 test frames, one noise draw each; no task holdout, no occluded or model-generated rollout states. Frames can be correlated within episodes.
- **Local fidelity is insufficient:** the notebook tests one replaced MLP at a time. Simultaneous replacement of all selected transcoders remains a required follow-up before claiming whole-model fidelity.
- **Attribution is untested:** feature meaning, cross-layer influence, intervention predictions, steering, task success, and safety are separate experiments.
- **Runtime dependence:** current teacher/runtime hashes freeze this run, but do not recover the old checkpoint's missing provenance. Float32 activation storage does not turn a bfloat16 teacher into a float32 teacher.
- **Operational limits:** full dataset availability, GPU memory, video decoding, and real policy integration require a Colab smoke run. CPU fixtures cannot certify these.

## Reproduction and implementation map

```bash
python scripts/embed_pi05_trajectory_training_notebook.py
python -m pytest -q tests/test_trajectory_training.py tests/test_transcoder_methodology.py tests/test_transcoder_audit.py tests/test_matched_steering.py
```

- [Collection, training orchestration, validation, export](../scripts/run_pi05_trajectory_training.py)
- [Optimizer, episode plan, checkpoints, dashboard](../src/pi05_mi/trajectory_training.py)
- [Adapted architecture](../src/pi05_mi/transcoder_methodology.py)
- [Notebook generator](../scripts/embed_pi05_trajectory_training_notebook.py)
- [Regression tests](../tests/test_trajectory_training.py)

Verification: **50 CPU tests passed** across the four suites above, including notebook execution with a synthetic teacher and interruption/resume checks. GPU training was not run in this review. Test results are implementation evidence, not evidence of learned circuits.

# Budgeted occlusion-monitor pilot

This is new experimental infrastructure, **not a reproduction of the paper's missing 600 episodes**.
The first question is whether a small logistic monitor predicts future harm on unseen starting scenes
and an unseen occluder condition. Actual stopping / collision prevention is a subsequent experiment.

## Run in Colab on A100 or L4

`REQUIRED_GPU="Auto"` accepts A100 or L4; the VRAM check is 22 GiB minimum to
accommodate the memory reported by an L4. `INFERENCE_BATCH_SIZE=0` automatically
uses batch 1 on L4 and batch 4 on A100. Explicit positive values override the batch.
The large `a100` preset still describes the episode plan, independent of GPU type.
L4 throughput and end-to-end memory fit require a real GPU smoke run; a 15-minute
cap does not promise completed analysis. Changing the effective batch requires a new run folder.

With matched comparison enabled, leave `CHECKPOINT` blank to use the checkpoint
referenced across Ahmed's, Jayden's and Marquise's feature/sanity branches:

```
DRIVE_ROOT/transcoders/pi05_libero/allframes_80-10-10_epoch1_b8_exp16_latest_lambda1e-4/step_027233.pt
```

If that path is absent, discovery searches the Drive/local transcoder and checkpoint
folders for the same filename. It rejects ambiguous fallback matches. An explicit
path is respected and never silently substituted. The binary weights are not in Git.
Only the selected three layer dictionaries are loaded to GPU; the full checkpoint is
read on CPU. This preserves the original MLP outputs and selected-layer measurements.

The collector uses `lerobot[libero]==0.4.4` with an explicit Transformers override in
[`scripts/pi05_transformers_overrides.txt`](../scripts/pi05_transformers_overrides.txt).
The published [PyPI 0.4.4 wheel](https://pypi.org/pypi/lerobot/0.4.4/json) omits the
`pi` extra present in the GitHub source. Its `libero` extra requests stock Transformers
>=4.57.1, but its Pi0.5 code requires the `fix/lerobot_openpi` fork (version 4.53.3).
Adding `[pi]` or reinstalling stock Transformers does not fix that mismatch.
The uv override selects fork commit `dcddb970176382c0fcf4521b0c0e6fc15894dfe0`
while still resolving the fork's dependencies, including tokenizers.
Later LeRobot versions changed both the
LIBERO config class and simulator integration; importing `LiberoEnvConfig` under the
old name is not sufficient for this collector's raw observations and simulator access.
The notebook installs into `sys.executable`, the active notebook kernel, and runs
collection and fitting directly in that kernel's main thread. It does not launch
Python worker processes for either phase. Output is live in the cell and copied to
phase logs; old logs are archived separately on reruns. Exceptions keep their original
tracebacks. The kernel's interrupt handlers remain intact.
The four **Pilot Definitions** cells contain the actual configuration/occlusion helpers,
collector, fitter, and budget runner as Python source. The launch cell calls these
definitions; it does not import `occlusion_pilot.py` or the collection/fitting/runner
scripts from `/content/groot-run`. Model, simulator, and transcoder support libraries
still come from the cloned repository. Run all four definition cells before collection.
For repository maintenance, `python scripts/embed_pi05_monitor_notebook.py` refreshes
these cells from the CLI implementations; it is never run by the notebook.
The Colab installer explicitly targets system Python with `--system --break-system-packages`.
External setup commands stream stdout/stderr into the cell and include their last
100 output lines in any failure, so dependency/build errors survive the notebook traceback.
Setup installs `cmake==3.31.6`, setuptools and wheel before building `egl-probe==1.0.2`.
This transitive LIBERO dependency uses CMake 2.8 policy settings that CMake 4 rejects.
Only `egl-probe` disables build isolation to use the preinstalled build tools; the
main install keeps CMake pinned so a later resolution cannot upgrade it back to 4.x.

If setup replaces already-imported packages, restart the Colab session once and rerun
from the top. The old `/content/lerobot-venv` is not used: Python 3.12 extension modules
cannot be imported safely into a Python 3.13 notebook kernel. Cache/rendering variables
are set before model-library imports. Setup and collection validate Pi0.5 compatibility
in the actual kernel. LeRobot and the Transformers fork commit are pinned;
`runtime.json` records the actual dependency versions.

If an older open notebook fails with `cannot import name 'LiberoEnv'` or
`An incorrect transformer version is used`, update the pilot
branch and reopen the updated notebook. Run setup, restart the session if prompted,
then run all cells. Model caches/Drive outputs are preserved. For a standalone CLI
environment, the repair helper remains available:

```bash
python scripts/pi05_monitor_runtime.py --repair --uv /content/uv-bin/uv --require-cuda
```

Use a new `RUN_NAME` after changing runtime/protocol; do not mix versions under an old
manifest. Then rerun only collection/reporting in the already-prepared session. Open
the latest notebook separately to pick up changed cells: git pull cannot replace cells
already loaded in a browser. The updated notebook resolves a blank checkpoint when
matched comparison is enabled; older notebook cells do not have this discovery step.
In particular, moving an existing run from batch 4 to batch 1 requires a fresh name.
The manifest error now lists changed fields (for example `config.batch_size: 4 -> 1`)
and leaves the prior manifest intact. The notebook defaults to `matched-probes-notebook-001`
to separate the embedded workflow from earlier script-based attempts.

When collection produces no complete paired groups for any required split, the runner
skips fitting. On exceptions, it records the failure in `run_state.json` and the phase
log and raises the original exception directly; the cell does not display an old summary.
Full simulator/GPU validation must still occur on Colab.

The pretrained PaliGemma checkpoint stores its tied token-embedding tensor under
`lm_head.weight`, omitting the duplicate `embed_tokens.weight` key. The loader restores
missing aliases only for parameters that are actually shared in the instantiated model.
Strict loading remains enabled; unrelated missing weights still stop the run.

For the current matched linear-probe/transcoder experiment, see
[the comparison protocol](pi05_matched_monitor_comparison.md). The notebook now defaults to
requiring a trained action-expert checkpoint and saving matched input/output/sparse features.
Set `REQUIRE_MATCHED_COMPARISON=False` only for an intentional dense-only legacy pilot.

Open `notebooks/pi05_occlusion_monitor_pilot.ipynb` and run all. Setup installs/restores the existing
LIBERO environment first. **The 15-minute budget starts after setup**, includes policy loading, and
reserves about two minutes for analysis. Cold downloads/installations have no 15-minute guarantee.
The deadline is cooperative: collection checks it between simulator/inference steps,
and a model load, inference call, or fitting operation already in progress may overrun it.
There is no background process to kill at the deadline.
Batch size defaults to four. This is genuine batched policy inference with one policy resident on the GPU;
simulator steps are serialized on CPU and image writes use four workers. It is not four separate GPU models.
Measure peak GPU memory and batch latency before increasing batch size, especially with optional transcoders.

The collection plan has 1,010 episodes over 240 distinct starts across ten tasks. It is a **ceiling for
repeated budgeted sessions, not a claim that 1,010 episodes fit in 15 minutes**. Training, calibration,
and test groups are interleaved in execution order to obtain an initial evaluable subset quickly.

```bash
python scripts/run_pi05_monitor_budget.py \
  --output-dir /persistent/path/occlusion-pilot-001 \
  --minutes 15 --preset a100 --batch-size 4
```

Rerun the identical command to extend the same collection. To inspect only the plan:

```bash
python scripts/run_pi05_occlusion_pilot.py --preset a100 \
  --batch-size 4 --output-dir /persistent/path/occlusion-pilot-001 --plan-only
```

Presets: smoke = 13 episodes / 3 starts; pilot = 52 / 12; a100 = 1,010 / 240.
Every episode retains the 280-step horizon. A short horizon would confound failure with insufficient time.
Use a fresh output folder for changed configurations. `screen` is never used for training or calibration.
Related clean/perturbed episodes share explicit LIBERO initial-state index, simulator seed, and per-time noise.
Initial-state hashes must match before analysis. Episode splits, not individual frames, determine separation.

## What is collected

- The notebook installs a renderer-local compatibility fix for robosuite 1.4 segmentation
  with NumPy 2. Native pixel buffers stay `uint8`; 24-bit ID decoding widens to `uint32`,
  then returns the usual signed `HxWx2` object-type/object-ID array. An error saying
  `Python integer 256 out of bounds for uint8` concerns this arithmetic, not camera width.
- Target identity comes from the active simulator's `bddl_file_name`, the BDDL placement goal,
  and LIBERO's object registry. LeRobot 0.4.4 does not store `_task_bddl_file` on its wrapper.
  Each episode's `target.json` records the BDDL hash, goal, exact target/distractor instances,
  body IDs and disjoint geometry IDs. Ambiguous or unmapped instances fail explicitly.
- Every saved camera frame has an accompanying `*_bowl_masks.npz` with separate boolean
  `<instance>__before` and `<instance>__after` masks, aligned to RGB before/after the overlay,
  plus a transparent `*_target.png` cutout of the original target pixels. Per-instance pixel
  counts and overlay overlap are recorded at every monitor observation. These are audit
  annotations, never inputs to the fitted probes. Instance separation does not guarantee that
  harm labels are statistically separable. Rectangular overlays can also hide distractor pixels;
  their overlap is recorded rather than described as a target-only intervention.
- Collection rejects an initial state with no visible target pixels in either camera. Later
  disappearance remains recorded as unknown visibility. No failed start is silently skipped.
  This dataset schema requires a fresh run folder (default `matched-probes-bowls-002`).
- Clean; target-following rectangle in both cameras; rectangle fixed at initial target position;
  equally sized rectangle avoiding target pixels; external-camera-only screen.
- Rectangles are **flat RGB overlays**, not replicas of the paper's 3D render objects. Segmentation
  measures target pixels before/after the overlay. No target pixels means unknown visibility, not success.
- Peak force at every physics substep for individual robot contacts with movable non-target objects.
  Harm is contact >10 N or non-target bowl lift >5 cm. **Drops are not labeled in this pilot.**
- State / previous executed action; simple 4x4 RGB-grid features from both cameras; 32 fixed coordinates
  from each of three PaliGemma MLPs. These are generic dense activations, not claimed safety features.
- Optional `--checkpoint` adds 32 fixed transcoder coordinates at each of three action-expert layers.
  This is the repository's action-expert transcoder, **not the paper's backbone TopK transcoder**.
- Fresh observations and a separate forward every eight control steps. Its actions are discarded until
  replanning; control still executes 50 actions per chunk. Batched forward latency is saved. Simulation
  pauses during inference, so the study does not yet model real-time sensing/actuation latency.
- Executed actions, forces/lifts, first harm time, completion, image samples, feature tensors and provenance.

The cheap RGB grid is only a starting visual baseline, not a strong image encoder or object detector.
No force, label, condition, task ID, simulator visibility, or future information enters the regression inputs.
Selected feature coordinates depend only on layer identity and width, not test data.

## Fitting and reporting

Default label: harm during the next 40 control steps. Post-first-harm samples are excluded, and a
negative requires a fully observed future window (truncated tails are not silently treated as safe).
The alternative `--target bad_episode` predicts failure OR harm. These are distinct endpoints.

Each episode has equal total training weight. Scaling and logistic regression are fitted only on training
episodes. The threshold uses maximum scores on successful, harm-free **calibration** episodes, targeting
at most 10% false stops empirically. Test episodes are never used to choose thresholds.
No threshold can be calibrated if there are no good calibration episodes; no classifier can be fitted
with only one training class. The report says so rather than fabricating a result.

Reports separate familiar conditions from the unseen screen, show within-condition prediction, raw
numerators/denominators, warning lead time, and delay sensitivity. Bootstrap intervals resample paired
starting-state groups and condition on the fitted model; they do not establish generalization to new tasks.
Only completed condition groups enter partial-run analysis. A time-limited subset can still be biased;
later paper-grade evaluation should finish the fixed plan and avoid repeatedly tuning on its test set.

The sample-count warnings (50 harmful/harm-free training episodes, 30 good calibration episodes,
20 harmful/good test episodes) are pragmatic feasibility checks, **not a power calculation**.
Success depends on event prevalence and episode diversity; no fixed count guarantees an effective monitor.

## Persistence and failures

Store the output directory on mounted Drive or another persistent filesystem. `run_state.json` is
atomically replaced with `phase`, `status`, timestamps, active episode IDs, last control steps,
completed counts, artifact paths, and errors. Phases include model loading, collection, fitting, report.

Each physics-control step appends a partial trajectory event. Each monitor tick saves partial features
and observation metadata. A finished episode gets an atomic `episode.json` marker after its features and
images finish writing. On resume, finished episodes are skipped; an interrupted episode is retried from
its recorded initial state. **This does not pretend to restore the simulator/controller mid-episode.**

Each fitted model is saved atomically and reused on retry for the same data/settings. New completed data
creates a new analysis revision, preserving earlier results. The output contains:

```
run_state.json                  current phase and resumability metadata
manifest.json                   fixed plan, experiment definitions, configuration
episodes/<id>/episode.json      completion marker, observations, physical events
episodes/<id>/features.npz      numerical predictor inputs
episodes/<id>/initial_state.npy exact settled initial simulator state
episodes/<id>/images/           sampled input images
episodes/<id>/progress.json     partial/completed episode status
monitor_future_harm/<revision>/ fitted models, scores, report.json, report.md
budget_summary.json/.md         results or an explicit insufficient-data/error report
collection.log, analysis.log    diagnostic logs
```

## GPU acceptance gate before interpreting results

CPU tests cover masks, split leakage, censored labels, calibration ties, latency accounting, and an
end-to-end synthetic fit/resume. They **do not validate MuJoCo/LeRobot integration**. First inspect
the smoke run's camera images, target masks, force units/contact categories, matched initial hashes,
model load checks and observed actions. In particular, confirm clean task success and real target hiding.
This environment has not executed the GPU integration run.

Suitable wording after an actual small run: "In a preliminary held-out simulation pilot, [N] of [H]
harmful episodes were preceded by an activation-monitor warning, with [F]/[G] false stops on good
episodes." Report the condition split and uncertainty. Do not call this demonstrated prevention.

Sources used for integration: LeRobot v0.4.4 `envs/libero.py` and `scripts/lerobot_eval.py`; LIBERO
`envs/env_wrapper.py`; robosuite v1.4.1 `robot_env.py` camera segmentation convention. Installed-version
differences must pass the smoke gate before larger experiments.

# Pi0.5 DifFRACT Transcoder Plan

This note records the implementation plan for applying DifFRACT-style transcoders to the Pi0.5 LIBERO action expert.

## Goal

Train sparse, timestep-conditioned transcoders that imitate each MLP in the Pi0.5 action expert / diffusion block. During training, the original Pi0.5 model remains frozen and each transcoder learns from the corresponding original MLP input-output behavior. After training, transcoders can be used for probing, replacement, steering, and later circuit tracing.

## Core Concept

For each target MLP layer `l`, train one independent transcoder:

```text
TC_l(x, t) ~= MLP_l(x)
```

where:

```text
x = one token-position input vector to MLP_l, shape [d_model]
y = original MLP output vector MLP_l(x), shape [d_model]
t = diffusion denoising timestep
```

The transformer MLP is position-wise. It receives tensors such as `[batch, tokens, d_model]`, but the MLP function itself maps each token vector independently:

```text
MLP_l: R^d_model -> R^d_model
```

So a full model forward can produce many transcoder training records for each MLP:

```text
records per MLP ~= batch_size * action_tokens * denoise_steps
```

Attention is where token mixing happens. The transcoder only approximates the MLP computation.

## Transcoder Architecture

DifFRACT uses timestep-conditioned transcoders with FiLM modulation:

```text
timestep t
    |
    v
timestep embedding e_t
    |
    v
trainable time MLP
    |
    +--> scale(t)
    +--> shift(t)

x_mod = x * (1 + scale(t)) + shift(t)
z = ReLU(W_enc x_mod + b_enc)
y_hat = W_dec z + b_dec
```

Definitions:

```text
x      : MLP input vector, shape [d_model]
z      : sparse latent features, shape [d_feat]
y_hat  : transcoder prediction of MLP output, shape [d_model]
d_feat : expansion dimension, e.g. 16 * d_model in DifFRACT
```

`scale(t)` and `shift(t)` are produced by a trainable time-conditioning network. They are not fixed vectors. They are learned end to end through the same reconstruction/sparsity loss.

FiLM modulation is per hidden channel:

```text
x_mod[j] = x[j] * (1 + scale(t)[j]) + shift(t)[j]
```

The final scale/shift layer should start near zero so training begins close to an ordinary non-time-conditioned transcoder:

```text
scale(t) = 0, shift(t) = 0 -> x_mod = x
```

## Loss

For each MLP/transcoder `l`, DifFRACT uses a separate objective:

```text
L_l =
E_{x,t} [
  ||MLP_l(x) - TC_l(x,t)||_2^2
  /
  (sum_j Var_{x,t}(MLP_l(x)_j) + eps)
]
+ lambda * E_{x,t}[ ||z_l(x,t)||_1 ]
```

Meaning:

```text
one MLP -> one transcoder -> one loss
```

If there are 60 action-expert MLPs, there are 60 transcoders and 60 conceptual training objectives. In code, these losses may be summed for one PyTorch backward pass because the transcoders have disjoint parameters, but conceptually each transcoder learns only from its corresponding MLP.

The variance denominator is per transcoder/per MLP. It is estimated across that MLP's activation records over examples, token positions, and diffusion timesteps.

Example with `d_model = 4` and five records:

```text
y_1 = [1, 10,  0, 5]
y_2 = [2, 12,  1, 5]
y_3 = [3, 14,  0, 7]
y_4 = [4, 16, -1, 7]
y_5 = [5, 18,  0, 9]
```

Compute variance per coordinate:

```text
Var_1 = Var([1, 2, 3, 4, 5])
Var_2 = Var([10, 12, 14, 16, 18])
Var_3 = Var([0, 1, 0, -1, 0])
Var_4 = Var([5, 5, 7, 7, 9])
```

Then:

```text
denominator = Var_1 + Var_2 + Var_3 + Var_4
```

This normalizes reconstruction error by the natural output scale of that MLP.

## Training Records

A record is one local supervised example for one transcoder:

```text
record = (x, y, t)
```

where:

```text
x = input vector to original MLP_l
y = output vector from original MLP_l
t = diffusion denoising timestep
```

If a target MLP sees:

```text
X shape = [batch, tokens, d_model]
Y shape = [batch, tokens, d_model]
```

then DifFRACT-style records are:

```text
(X[b, p, :], Y[b, p, :], t)
```

for each batch item `b` and token/action position `p`.

This is efficient because one expensive frozen model forward produces many cheap supervised records for each small transcoder.

## Modes

Training mode:

```text
x -> original MLP -> y_teacher -> returned to main model
x,t -> transcoder -> y_hat,z -> used for transcoder loss
```

The original Pi0.5 forward behavior is unchanged. Only transcoder parameters train.

Probe mode:

```text
x -> original MLP -> returned to main model
x,t -> transcoder -> log y_hat, z, MSE, sparsity
```

No optimizer step is required. This checks fidelity without changing policy behavior.

Replacement mode:

```text
x,t -> trained transcoder -> y_hat -> returned to main model
```

The original MLP is skipped or used only for diagnostics.

## Stage 1: Pi0.5 Inspection

Before implementing training, inspect the installed LeRobot Pi0.5 model in the `pi0.5` venv:

```text
/Users/akhidre/pubgit/pi05-run/pi0.5/bin/python
```

Stage 1 deliverables:

```text
1. Confirm import path and version for LeRobot.
2. Load or instantiate the Pi0.5 LIBERO policy.
3. Print all module names matching action expert / diffusion block MLPs.
4. Confirm each target MLP input and output shape.
5. Confirm how many times each MLP is called during action denoising.
6. Locate the diffusion timestep tensor or scheduler value available at each MLP call.
7. Save a short inspection report with names, shapes, and timestep source.
```

Status: completed in `docs/pi05_stage1_inspection.md`.

Expected target family based on existing activation capture:

```text
paligemma_with_expert.gemma_expert.model.layers.<idx>.mlp
```

This must be verified against the installed `lerobot==0.6.1` package.

## Stage 2: Transcoder Module

Implement:

```text
src/pi05_mi/transcoders.py
```

Required pieces:

```text
TimeConditionedTranscoder
FiLM scale/shift time MLP
ReLU sparse latent z
linear decoder
decoder column renormalization after optimizer step
checkpoint save/load helpers
```

Status: core module completed in `src/pi05_mi/transcoders.py`.

## Stage 3: Hooks and Wrappers

Implement wrappers around target MLPs:

```text
src/pi05_mi/patch_pi05.py
```

Wrapper responsibilities:

```text
1. Preserve original MLP.
2. Freeze original model parameters.
3. Capture x, y, t records during training/probe.
4. Route output according to mode: train, probe, replace.
5. Expose per-layer losses and metrics.
```

Status: core wrapper completed in `src/pi05_mi/patch_pi05.py`.

## Stage 4: Buffers and Variance Stats

Implement per-transcoder activation buffers:

```text
src/pi05_mi/buffers.py
```

Required pieces:

```text
bounded per-layer record buffers
online variance estimate for y
minibatch sampling of records
device/dtype management
```

The variance denominator does not require storing every output forever. It can be estimated with streaming statistics while buffers keep a bounded training set.

Status: core buffer module completed in `src/pi05_mi/buffers.py`.

## Stage 5: Training Loop

Implement:

```text
scripts/train_pi05_transcoders.py
```

Training loop:

```text
1. Load frozen Pi0.5 policy.
2. Patch action-expert MLPs with wrappers.
3. Run policy on LIBERO-compatible observations/prompts/state with sampled action noise and a sampled flow time.
4. Collect records into per-layer buffers.
5. Sample buffer minibatches.
6. Compute each transcoder's normalized MSE + lambda L1.
7. Optimizer step on transcoder parameters only.
8. Renormalize decoder columns.
9. Log per-layer reconstruction, sparsity, variance denominator, and buffer fill.
10. Save transcoder checkpoints.
```

Status: initial single-process trainer completed in `scripts/train_pi05_transcoders.py`.

Default trainer behavior:

```text
frozen Pi0.5 cheap denoiser call
    -> prompt/image/state enter the normal Pi0.5 prefix
    -> action-side input is sampled noise
    -> one random flow time t is sampled
    -> Pi0.5 runs one denoise_step, not iterative inference
    -> wrappers capture x, original MLP y, raw flow time t
    -> records are drained into per-layer buffers
    -> each ready transcoder samples its own buffer
    -> normalized MSE + lambda * L1 is optimized
    -> decoder columns are renormalized
    -> checkpoints are saved under outputs/transcoders/pi05_libero/
```

The default is `--collection-mode random-timestep`. It does not use ground-truth action chunks and does not run iterative inference denoising.

The script also has optional diagnostic modes:

```text
--collection-mode inference
    runs normal iterative action generation with scheduled denoising times

--collection-mode training-forward
    uses LeRobot's normal Pi0.5 training forward, needs dataset action chunks,
    and samples one random flow time per example
```

Example short run:

```bash
/Users/akhidre/pubgit/pi05-run/pi0.5/bin/python scripts/train_pi05_transcoders.py \
  --policy-path lerobot/pi05_libero_finetuned \
  --num-feed-forwards 100 \
  --batch-size 1 \
  --episodes 0 \
  --device cuda \
  --collection-mode random-timestep \
  --lambda-l1 1e-4 \
  --expansion-factor 16
```

This should be run on an NVIDIA GPU machine with the Pi0.5 checkpoint and `HuggingFaceVLA/libero` dataset available. Local Mac CPU/MPS is useful for code checks, not real Pi0.5 transcoder training.

Mac/M1 smoke attempt:

First check local cache state:

```bash
/Users/akhidre/pubgit/pi05-run/pi0.5/bin/python scripts/check_pi05_cache.py
```

If this reports missing model or dataset files, remove `--local-files-only` from the smoke command so Hugging Face can download the missing files once.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 \
/Users/akhidre/pubgit/pi05-run/pi0.5/bin/python scripts/train_pi05_transcoders.py \
  --policy-path lerobot/pi05_libero_finetuned \
  --local-files-only \
  --num-feed-forwards 1 \
  --batch-size 1 \
  --episodes 0 \
  --device auto \
  --policy-dtype auto \
  --collection-mode random-timestep \
  --expansion-factor 1 \
  --buffer-capacity 500 \
  --min-buffer-records 50 \
  --transcoder-batch-size 16 \
  --transcoder-epochs-per-ff 1 \
  --save-every 1 \
  --log-every 1
```

Parameter meanings:

```text
--num-feed-forwards
    number of frozen Pi0.5 record-collection calls

--batch-size
    number of prompt/image/state examples per feed-forward

--episodes
    optional comma-separated dataset episode ids to load; useful for smoke runs

--transcoder-epochs-per-ff
    number of full passes through each ready transcoder buffer after each feed-forward

--transcoder-batch-size
    minibatch size for splitting each ready transcoder buffer during an epoch
```

Example: if one transcoder buffer has 50 records and `--transcoder-batch-size 5`, one transcoder epoch has 10 minibatches. With `--transcoder-epochs-per-ff 100`, that transcoder trains for 100 full passes through those buffered records.

On an M1, `--device auto` selects `mps` when available and `--policy-dtype auto` selects `float32`. This command is only a plumbing test. It may still fail if the full Pi0.5 weights or LIBERO dataset are not cached, or if local unified memory is too small.

## Stage 6: Probe and Replacement

Implement:

```text
scripts/probe_pi05_transcoders.py
scripts/eval_pi05_transcoders.py
```

Probe:

```text
load original Pi0.5 + trained transcoders
run original policy behavior
log transcoder fidelity and sparse features
```

Replacement:

```text
load original Pi0.5 + trained transcoders
replace action-expert MLP outputs with transcoder outputs
evaluate action quality / LIBERO rollout behavior
```

## Open Questions To Resolve In Stage 1

```text
1. Exact Pi0.5 action-expert MLP module names in LeRobot 0.6.1.
2. Whether target MLPs receive only x or also explicit conditioning arguments.
3. Where diffusion timestep t is available during each MLP call.
4. Actual action token count and hidden dimension.
5. Whether LIBERO training batches or simulator rollouts are the first practical data source.
6. Memory cost of storing x/y records for all target MLPs.
```

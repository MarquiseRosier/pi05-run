# Pi0.5 Stage 1 Inspection

This report inspects the LeRobot Pi0.5 LIBERO action-expert MLPs targeted by DifFRACT-style transcoders.
It uses the Hugging Face config and a meta-device model instantiation; checkpoint weights are not loaded.

## Source

- HF policy repo: `lerobot/pi05_libero_finetuned`
- Config path: `/Users/akhidre/.cache/huggingface/hub/models--lerobot--pi05_libero_finetuned/snapshots/8e174154ef5f6c60a8da12ae99c303d8963138c1/config.json`
- Train config path: `/Users/akhidre/.cache/huggingface/hub/models--lerobot--pi05_libero_finetuned/snapshots/8e174154ef5f6c60a8da12ae99c303d8963138c1/train_config.json`
- Training dataset listed by config: `HuggingFaceVLA/libero`

## Policy Settings

- `paligemma_variant`: `gemma_2b`
- `action_expert_variant`: `gemma_300m`
- `chunk_size`: `50` action tokens
- `n_action_steps`: `50`
- `max_action_dim`: `32` internal padded action dim
- `num_inference_steps`: `10`
- timestep embedding periods: `0.004` to `4.0`

The LIBERO checkpoint outputs 7 action dimensions, but Pi0.5 pads actions internally to 32 dimensions before projecting them into the 1024-wide action expert.

## Target MLPs

- Target count: `18` action-expert MLPs
- Target name pattern: `paligemma_with_expert.gemma_expert.model.layers.<idx>.mlp`
- Module type: `GemmaMLP`
- Per-token function shape: `R^1024 -> R^1024`
- Internal MLP expansion: `1024 -> 4096 -> 1024`

| idx | module | up_proj | gate_proj | down_proj |
| ---: | --- | --- | --- | --- |
| 0 | `paligemma_with_expert.gemma_expert.model.layers.0.mlp` | `[4096, 1024]` | `[4096, 1024]` | `[1024, 4096]` |
| 1 | `paligemma_with_expert.gemma_expert.model.layers.1.mlp` | `[4096, 1024]` | `[4096, 1024]` | `[1024, 4096]` |
| 2 | `paligemma_with_expert.gemma_expert.model.layers.2.mlp` | `[4096, 1024]` | `[4096, 1024]` | `[1024, 4096]` |
| 3 | `paligemma_with_expert.gemma_expert.model.layers.3.mlp` | `[4096, 1024]` | `[4096, 1024]` | `[1024, 4096]` |
| 4 | `paligemma_with_expert.gemma_expert.model.layers.4.mlp` | `[4096, 1024]` | `[4096, 1024]` | `[1024, 4096]` |
| 5 | `paligemma_with_expert.gemma_expert.model.layers.5.mlp` | `[4096, 1024]` | `[4096, 1024]` | `[1024, 4096]` |
| 6 | `paligemma_with_expert.gemma_expert.model.layers.6.mlp` | `[4096, 1024]` | `[4096, 1024]` | `[1024, 4096]` |
| 7 | `paligemma_with_expert.gemma_expert.model.layers.7.mlp` | `[4096, 1024]` | `[4096, 1024]` | `[1024, 4096]` |
| 8 | `paligemma_with_expert.gemma_expert.model.layers.8.mlp` | `[4096, 1024]` | `[4096, 1024]` | `[1024, 4096]` |
| 9 | `paligemma_with_expert.gemma_expert.model.layers.9.mlp` | `[4096, 1024]` | `[4096, 1024]` | `[1024, 4096]` |
| 10 | `paligemma_with_expert.gemma_expert.model.layers.10.mlp` | `[4096, 1024]` | `[4096, 1024]` | `[1024, 4096]` |
| 11 | `paligemma_with_expert.gemma_expert.model.layers.11.mlp` | `[4096, 1024]` | `[4096, 1024]` | `[1024, 4096]` |
| 12 | `paligemma_with_expert.gemma_expert.model.layers.12.mlp` | `[4096, 1024]` | `[4096, 1024]` | `[1024, 4096]` |
| 13 | `paligemma_with_expert.gemma_expert.model.layers.13.mlp` | `[4096, 1024]` | `[4096, 1024]` | `[1024, 4096]` |
| 14 | `paligemma_with_expert.gemma_expert.model.layers.14.mlp` | `[4096, 1024]` | `[4096, 1024]` | `[1024, 4096]` |
| 15 | `paligemma_with_expert.gemma_expert.model.layers.15.mlp` | `[4096, 1024]` | `[4096, 1024]` | `[1024, 4096]` |
| 16 | `paligemma_with_expert.gemma_expert.model.layers.16.mlp` | `[4096, 1024]` | `[4096, 1024]` | `[1024, 4096]` |
| 17 | `paligemma_with_expert.gemma_expert.model.layers.17.mlp` | `[4096, 1024]` | `[4096, 1024]` | `[1024, 4096]` |

## Call Pattern

Training forward:

- `PI05Policy.forward` samples one continuous flow-matching time per batch item.
- `PI05Pytorch.forward` builds noisy actions `x_t = t * noise + (1 - t) * actions`.
- The action suffix has `chunk_size` action tokens.
- Each target MLP is called once for that sampled time.

Inference forward:

- `sample_actions` runs Euler integration for `10` denoise steps.
- At step `s`, LeRobot uses `time = 1.0 + s * (-1 / num_steps)`.
- With `10` steps, times are `1.0, 0.9, ..., 0.1`.
- Each denoise step calls every action-expert MLP once.

Therefore, for one inference action chunk:

- MLP calls: `18 layers * 10 timesteps = 180` expert MLP calls.
- Per target MLP, DifFRACT-style records: `batch_size * 50 action tokens * 10 timesteps`.

## Timestep Source

- Raw timestep is the flow-matching scalar `t` passed to `embed_suffix(noisy_actions, timestep)`.
- `embed_suffix` creates a sinusoidal time embedding, then applies `time_mlp_in -> SiLU -> time_mlp_out -> SiLU`.
- The resulting `adarms_cond` conditions every action-expert layer through adaptive RMSNorm.
- The actual `GemmaMLP.forward` still receives only the post-attention normalized hidden states `x`; the wrapper must provide raw `t` to the transcoder through side context.

## Stage 2 Implications

- Train one timestep-conditioned transcoder per listed MLP.
- Transcoder input record: one token vector `x` with shape `[1024]` plus raw flow timestep `t`.
- Transcoder target: original MLP output vector `y` with shape `[1024]`.
- Capture tensors before/after each listed `GemmaMLP`.
- Capture `t` by wrapping `forward`/`denoise_step` or `embed_suffix`, because the MLP module itself does not receive `t` as an argument.

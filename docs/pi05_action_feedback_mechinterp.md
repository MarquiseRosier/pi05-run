# Pi0.5 Action Feedback Mechanistic Interpretability

This branch adds tracing for the question:

```text
What does Pi0.5 receive at each replanning turn, how are action chunks rebuilt,
and what changes if we inject visible action/state feedback into the prompt?
```

## Ground Truth Loop

The Colab runner uses LeRobot eval with:

```text
LIBERO simulator observation
-> preprocess images + robot state + task text
-> Pi0.5 predicts [50, 7] action chunk
-> LeRobot executes the first 10 actions one at a time
-> LIBERO returns fresh observations after each action
-> Pi0.5 replans when the 10-action queue is empty
```

By default, prior actions are not fed back as narrative text or as action tokens
on the next policy call. They affect the next call indirectly through the
simulator state: new camera images plus a new robot-state vector.

Pi0.5 preprocessing turns the current normalized robot state into text:

```text
Task: <LIBERO task>, State: <256-bin discretized state values>;
Action:
```

For LIBERO, the state is an 8D vector:

```text
eef_pos(3), eef_axis_angle(3), gripper_qpos(2)
```

## New Capture Events

When `CAPTURE_ACTIVATIONS=1`, this branch now also records:

- `policy_config`: chunk size, executed action horizon, action dimension.
- `policy_step_start`: every `select_action` call, including queue length and whether a new chunk will be generated.
- `chunk_start`: policy-call prompt, token count, token IDs, normalized state summary, and tensor summaries.
- `denoise_start`, `denoise_step`, `denoise_end`: flow-matching trajectory from initial action noise to final predicted chunk.
- `action_chunk`: the normalized `[50, 7]` chunk before the output postprocessor.
- `policy_selected_action`: the normalized single action popped from the queue for this step.
- `env_reset`: raw simulator observation summary at episode reset.
- `env_step`: exact postprocessed action sent to LIBERO, reward/success flags, and the fresh simulator observation returned after the action.
- `prompt_feedback`: visible feedback inserted into the next Pi0.5 prompt, when enabled.

The raw event stream is:

```text
outputs/eval/pi05_libero/<run>/activation_capture/events.jsonl
```

## Notebook Controls

Useful defaults:

```text
CAPTURE_ACTIVATIONS = True
CAPTURE_FEEDBACK_TRACE = True
CAPTURE_ENV_STEPS = True
CAPTURE_ENV_STEP_IMAGES = False
CAPTURE_TOKEN_IDS = True
CAPTURE_DECODE_LANGUAGE = False
CAPTURE_DENOISE_TRACE = True
CAPTURE_MAX_TENSOR_VALUES = 64
PI05_PROMPT_FEEDBACK_MODE = "off"
```

`CAPTURE_DECODE_LANGUAGE=True` asks the capture shim to decode token IDs back
through the PaliGemma tokenizer. Keep it off unless needed because it can touch
the tokenizer cache and adds noise to the trace.

`CAPTURE_ENV_STEP_IMAGES=True` stores raw post-step simulator images every
`CAPTURE_ENV_STEP_IMAGE_EVERY_N` steps. Leave it off for normal runs because
the chunk camera tensors and rollout video are usually enough.

## Visible Feedback Ablations

Use paired runs with the same suite, task ID, episode count, and seed.

Baseline:

```text
PI05_PROMPT_FEEDBACK_MODE = "off"
```

Last-action feedback:

```text
PI05_PROMPT_FEEDBACK_MODE = "last_action"
```

Recent action-window feedback:

```text
PI05_PROMPT_FEEDBACK_MODE = "chunk_summary"
```

These modes inject explicit visible text into the Pi0.5 prompt before
tokenization. This is not hidden reasoning. It is a controlled prompt
intervention for testing whether the action expert/prefix stack changes when
the model receives a textual statement of what was just applied.

Example injected string:

```text
Feedback: last applied action dx=+0.012, dy=-0.004, dz=+0.021, dRx=+0.000, dRy=+0.002, dRz=-0.001, grip=+0.433;
```

## Rollout Ablation

For the question "does visible feedback accomplish the task faster?", use the
rollout ablation runner rather than the one-chunk prompt probe:

```bash
CAPTURE_ACTIVATIONS=1 TASK_IDS='[0]' SEED=1000 ./scripts/run_pi05_feedback_ablation.sh libero_spatial 5
```

For Docker, add `PI05_EVAL_RUNNER=./scripts/run_pi05_libero_docker.sh`.

By default this runs:

```text
off,last_action,chunk_summary
```

and writes:

```text
outputs/eval/pi05_libero/<ablation-id>_analysis/feedback_ablation_comparison.md
outputs/eval/pi05_libero/<ablation-id>_analysis/feedback_ablation_comparison.csv
outputs/eval/pi05_libero/<ablation-id>_analysis/feedback_ablation_episodes.csv
outputs/eval/pi05_libero/<ablation-id>_analysis/feedback_ablation_comparison.json
```

The comparison table reports official success rate, average reward, total
environment steps, total chunk calls, and steps/chunks to the first official
success signal recorded in the feedback trace.

Keep the suite, task IDs, episode count, and seed fixed across modes. One
episode is useful for debugging the trace; use several episodes before treating
speed differences as meaningful.

## Trace Summary Script

After a captured run:

```bash
python scripts/summarize_pi05_feedback_trace.py --run latest --max-steps 80
```

Outputs:

```text
analysis/feedback_trace_summary.md
analysis/feedback_env_steps.csv
analysis/feedback_chunks.csv
analysis/feedback_denoise_steps.csv
analysis/feedback_trace_summary.json
```

The notebook display cell runs this automatically when activation capture is
enabled. Inspect `feedback_trace_summary.md` first. It answers:

- how many actions each chunk contributed to the simulator;
- how the flow-matching sampler moved from noise to action chunk over denoise steps;
- what prompt/token/state the model received at chunk generation time;
- which normalized action was selected from the queue at each step;
- which postprocessed action was actually sent to LIBERO;
- what fresh robot state came back from the simulator after each action;
- whether prompt feedback injection was active.

## Suggested Analysis

After the ablation runner finishes, compare:

- chunk action deltas between baseline and feedback runs;
- token count changes and whether feedback caused prompt truncation;
- action-expert layer deltas for chunks after the first feedback-bearing prompt;
- success, steps-to-success, and trajectory divergence in rollout videos;
- applied env-action differences versus normalized selected-action differences.

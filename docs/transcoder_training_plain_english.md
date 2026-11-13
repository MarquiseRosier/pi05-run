# What we are training, in plain English

**We are teaching small models to copy parts of an existing robot model. We hope their simpler internal signals will help explain the robot later. Copying well is the first test; explaining correctly is a separate test.**

[Open the training notebook in Colab](https://colab.research.google.com/github/MarquiseRosier/pi05-run/blob/feature/transcoder-trajectory-training/notebooks/pi05_transcoder_trajectory_training.ipynb) · [Read the technical audit](transcoder_training_audit.md)

## 1. The teacher and the student

The **policy** is the existing Pi0.5 robot model. It reads camera pictures, robot state, and an instruction, then predicts actions. Think of it as the teacher.

A **layer** is one stage of its calculation. An **MLP** is a small neural network inside that stage. We target the MLPs in the 18 action-expert layers—not every part of the policy.

A **transcoder** is our student: it receives the same input as one MLP and tries to reproduce that MLP's output. Each layer gets its own student. The teacher's weights stay **frozen**, meaning we never train or change them.

Example: if a teacher MLP turns `[2, 1]` into `[4, 3]`, the student's job is to produce something close to `[4, 3]` from `[2, 1]`. The actual vectors contain 1,024 numbers. These numbers are **activations**: internal signals, not bowl coordinates or named concepts.

## 2. Where the examples come from

We read saved LIBERO observations from different moments throughout recorded robot episodes. An **episode** is one recorded attempt at a task.

For each observation, the frozen policy starts with random action noise and refines it through 10 **denoising steps**. Think of a rough sketch becoming a clearer drawing. This is an internal action prediction; we do not move a robot or execute simulator actions here.

At every step we save three things for each target layer:

| Saved item | Meaning |
|---|---|
| `x` | What entered the original MLP |
| `y` | What the original MLP produced—the answer to copy |
| `t` | Where the policy was in denoising |

**The key correction:** later steps receive the policy's already-refined action state. We do not substitute fresh random noise at every later timestep.

A **trajectory** here means this sequence of internal denoising states. It does **not** mean a robot moving through the room.

## 3. How the student copies the teacher

```text
MLP input + denoising time
          ↓
Encoder: turn the input into feature strengths
          ↓
16,384 candidate features; encourage most to be off
          ↓
Decoder: combine active features into a predicted MLP output
          ↓
Compare prediction with the original MLP output
```

The **encoder** detects patterns. The **decoder** combines their contributions. A **feature** is a learned numerical pattern; it is not automatically “the bowl feature.” No one supplies bowl-feature labels during training.

**ReLU** sets negative feature strengths to zero. **Sparse** means relatively few features are active for a particular input. For example, 500 active features out of 16,384 is about 3%. This is an illustration, not a promised outcome.

The time network adjusts how the student reads its input at different denoising steps. The same student handles all 10 steps at its layer.

## 4. What training rewards

The **loss** is a penalty the optimizer tries to reduce:

**Loss = copying error + penalty for large feature activity.**

- **MSE:** squared differences between the student's answer and the teacher's answer.
- **Variance:** how much the teacher's answers vary. Dividing by variance makes the copying error less dependent on their numerical size.
- **L1 penalty:** add up feature strengths and multiply by `λ = 0.0001`. This encourages economical use of features; it does not enforce an exact active-feature count.
- **Optimizer / AdamW:** the rule that adjusts the student's weights to reduce the loss.
- **Learning rate:** how large those adjustments are. It starts at `0.0002` and gradually decreases along a cosine curve.

We also keep each decoder feature direction at length one. Otherwise, the student could shrink a feature strength and enlarge its decoder contribution to evade the activity penalty. **Gradient clipping** limits unusually large update signals.

## 5. Exactly how much training?

Defaults, assuming the policy's 50-position action chunk:

| Item | Default |
|---|---:|
| Independent students / target layers | 18 |
| Candidate features per student | 16,384 |
| Collection-and-training cycles | 256 |
| Observation frames per cycle | 64 |
| Denoising steps per observation | 10 |
| Records per layer per cycle: `64 × 10 × 50` | 32,000 |
| Records sampled per optimizer update (**batch size**) | 1,024 |
| Updates per layer per cycle | 32 |
| Total updates per layer | 8,192 |
| Total sampled training records per layer | 8,388,608 |
| Scheduled observation frames | 16,384 |

A **record** is one action-token position at one denoising step, for one observation. Fifty positions describe a proposed action chunk—not 50 separately observed scenes.

We collect one shared set of teacher examples, then train one student at a time to save GPU memory. Batches sample records **with replacement**, so some records repeat and others may be skipped. Millions of records are not millions of independent scenes. Frames repeat only after the training-frame pool is exhausted.

## 6. How we check progress

Episodes are split approximately **80% training / 10% validation / 10% test**. Training adjusts weights. Validation checks progress and chooses saved weights. Test checks those selected weights only after training finishes.

| Dashboard metric | How to read it |
|---|---|
| Reconstruction nMSE | Smaller means closer copying; **not percent accuracy** |
| Active features/token, or **L0** | How many features are on; read together with copying error |
| Error by denoising call | Shows whether early or late steps are harder |
| Predicted-action error | How much predictions change when one MLP is replaced; **not movement or collision rate** |
| Updates, records/second, learning rate | Progress, speed, and update schedule |
| Checkpoint status | Whether training state has been saved to Drive |

The training loss uses each batch's variance. Comparable chart errors use one frozen reference variance per layer, preserved across restarts. Those are different measurements and are labelled separately.

Validation uses 24 fixed frames; action checks use 2 of them. Final test uses 16 frames. These are small checks, not broad proof of reliability. Held-out episodes can contain familiar tasks and may have been seen during policy pretraining.

A **checkpoint** saves weights and optimizer state so training can resume. The final export selects each layer's best measured validation nMSE. Keep the same settings and run name when resuming. This can take multiple Colab sessions.

## 7. What success would—and would not—mean

This is a **DifFRACT-informed VLA adaptation**, with a smaller budget and robot-specific choices. It is not an exact paper reproduction. [Paper, Appendix E](https://arxiv.org/html/2606.15796v1#A5)

Good copying would give us better candidates for the next test: **attribution**, meaning a claim about which internal signal contributed to an output. We must predict what changing a feature should do, change it, and compare with what actually happens. Connecting such contributions across layers is **circuit tracing**.

Training alone does not prove that features have clear meanings, that attribution is correct, or that steering works. Our earlier pilot improved copying under matched short training; it did not establish improved action attribution.

## Our first discussion question

**If the transcoder copies an MLP almost perfectly, have we proved that feature 717 means “move toward the bowl”? Why or why not?**

We can work through that one question first, then discuss denoising, sparsity, and the train/validation/test split.

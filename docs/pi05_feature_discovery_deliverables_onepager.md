# Pi0.5 Transcoder Feature Discovery: Deliverables #4-#6

## Goal

We run the frozen Pi0.5 LIBERO policy with trained action-expert transcoders in
probe mode. The pass records sparse transcoder activations and turns them into
feature-level evidence for interpretation.

## Data Shape

For each LIBERO observation, each action-expert layer, and each flow timestep,
the transcoder latent has action-position and feature axes:

```text
z[l, tau, p, i]
```

where:

- `l`: action-expert layer, 18 total
- `tau`: flow timestep, 10 inference steps in the pilot report
- `p`: action-chunk position, usually 0..49
- `i`: sparse feature, 16,384 for expansion 16

We collapse action position but keep flow time:

```text
score[l, tau, i] = max_p z[l, tau, p, i]
```

The collector also stores `p*`, the action position that produced the maximum.

## Pipeline

```text
LIBERO observations
        |
        v
Frozen Pi0.5 + trained transcoders in probe mode
        |
        v
Sparse activations z[l, tau, p, i]
        |
        v
Collapse action position: score[l, tau, i] = max_p z[l, tau, p, i]
        |
        +--> #4 Top-K observations per (layer, tau, feature)
        |
        +--> #5 Global statistics per (layer, tau, feature)
        |
        v
#6 Feature browser for ranking and visual inspection
```

## Pseudo Algorithm

```text
for each LIBERO observation:
    run Pi0.5 inference with transcoders in probe mode
    for each flow timestep tau:
        for each action-expert layer l:
            read transcoder latent z[l, tau, p, i]
            score[l, tau, i] = max over action position p
            update Top-K observations for every feature i
            update mean, std, firing frequency, and top-M frequency

after collection:
    rank candidate (layer, tau, feature) cells
    write feature_candidates.csv and feature_candidates.json
    render feature_report.html and feature_report_with_images.html
```

## Deliverables

### #4 Top-K Observations

`feature_topk.pt` stores the strongest observations for every feature-time cell:

```text
TopK[l, tau, i, k]
```

Each Top-K entry stores the activation score, observation id, episode index,
frame index, task text, flow timestep, and `p*`.

### #5 Global Statistics

`feature_stats.pt` stores statistics over all collected observations:

```text
mean[l, tau, i]
std[l, tau, i]
firing_frequency[l, tau, i]
top_m_frequency[l, tau, i]
```

Firing frequency means:

```text
P(score[l, tau, i] > epsilon)
```

### #6 Feature Inspection

The dashboard reports one row per selected `(layer, tau, feature)` cell. The
pilot report selects the top 200 candidates by:

```text
interesting = TopK mean / global std
```

Clicking a row shows the Top-20 observations that activated that feature most
strongly, including images in `feature_report_with_images.html`.

## Pilot Output

The first pilot used episodes `0..49`:

```text
observations: 13,835
layers: 18
flow timesteps: 10
features per layer: 16,384
Top-K: 20
candidate rows in report: 200
```

Example interpretation:

```text
L12:tau1:F7584
Top-20 observations mostly share a cup task.
Hypothesis: candidate cup-task or cup-phase feature.
Images and later interventions are needed before claiming causality.
```

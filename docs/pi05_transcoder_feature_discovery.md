# Pi0.5 Transcoder Feature Discovery

This stage implements the DifFRACT-style feature-discovery pass for the trained
Pi0.5 action-expert transcoders. It does not replace the model MLPs. It runs the
original Pi0.5 policy and uses the trained transcoders in probe mode to expose
sparse latent activations.

## Indices

The sparse activation is

```text
z[e, k, tau, p, l, i]
```

where:

- `e`: LIBERO episode
- `k`: robot observation timestep inside the episode
- `tau`: flow/denoising timestep
- `p`: action-chunk position, usually 0..49
- `l`: action-expert layer/transcoder, 18 total
- `i`: sparse feature, 16,384 for expansion factor 16 and d_model 1024

For one observation, one flow timestep, and one layer, the latent tensor is:

```text
Z_l[e, k, tau] in R[50, 16384]
```

## Deliverable 4: Top-K Observations

For initial discovery we collapse action position only and keep flow time as
an explicit axis:

```text
v_l[e, k, tau, i] = max_p z[e, k, tau, p, l, i]
```

For every `(layer, tau, feature)`, the collector keeps the top `K` robot
observations with largest score:

```text
top_scores in R[18, num_flow_steps, 16384, K]
```

It also stores the metadata needed to recover where the activation happened:

```text
observation_id, episode_index, frame_index, p*, tau
```

## Deliverable 5: Global Statistics

Independently of Top-K, the collector computes streaming statistics over all
observations:

```text
mean_l,tau,i = E[v_l[e,k,tau,i]]
std_l,tau,i  = Std[v_l[e,k,tau,i]]
freq_l,tau,i = P(v_l[e,k,tau,i] > epsilon)
```

These are all shaped:

```text
R[18, num_flow_steps, 16384]
```

The script can also track `top_m_frequency`, which measures how often a feature
is among the top `M` active features for an observation.

## Deliverable 6: Feature Inspection

The feature-inspection stage turns Deliverables 4 and 5 into a practical
DifFRACT-style browsing workflow. It is a selection and interpretation tool,
not an exhaustive manual inspection of all 295k features.

The report ranks `(layer, tau, feature)` triples. This matches the DifFRACT
feature-discovery convention: position is collapsed by `max_p`, while denoising
time remains meaningful.

Ranking options:

- `interesting`: Top-K mean divided by global std
- `max`: strongest observed activation
- `topk_mean`: mean of stored Top-K activations
- `mean`
- `std`
- `frequency`
- `top_m_frequency`

The report generator writes three artifacts:

```text
feature_candidates.csv
feature_candidates.json
feature_report.html
```

`feature_candidates.csv/json` are the machine-readable candidate list. Each row
contains:

```text
layer, timestep, feature, rank_score, max, topk_mean, mean, std,
frequency, top_m_frequency, active_count, top_m_count
```

`feature_report.html` is the browser. It supports filtering by layer/timestep,
searching by task text or feature id, sorting by metrics, and clicking a feature
to view its Top-K observations with `episode_index`, `frame_index`, `p*`, `tau`,
activation score, and instruction text.

Thumbnails are optional. When `--save-thumbnails` is set, the script reloads the
selected LIBERO observations and saves small images next to the report. Without
that flag, the report remains lightweight and uses only the saved metadata.

## Output Files

The collector writes:

```text
feature_topk.pt
feature_stats.pt
observations.jsonl
collection_metrics.jsonl
config.json
```

The report script writes:

```text
feature_report.html
feature_candidates.csv
feature_candidates.json
feature_report_thumbnails/
```

Thumbnails are optional. Without thumbnails, the report still contains scores,
episode/frame ids, instruction text, `p*`, and `tau`.

## Smoke Collection

Use a small cap first:

```bash
/Users/akhidre/pubgit/pi05-run/pi0.5/bin/python scripts/collect_pi05_transcoder_features.py \
  --checkpoint /path/to/step_027233.pt \
  --output-dir outputs/features/pi05_libero/smoke_top5 \
  --episodes 0 \
  --max-batches 2 \
  --batch-size 1 \
  --top-k 5 \
  --device auto
```

Then render a small report:

```bash
/Users/akhidre/pubgit/pi05-run/pi0.5/bin/python scripts/make_pi05_feature_report.py \
  --feature-dir outputs/features/pi05_libero/smoke_top5 \
  --max-features 20 \
  --top-examples 5 \
  --sort-by interesting
```

## Full Collection

For the same 80/10/10 train split used in transcoder training:

```bash
/Users/akhidre/pubgit/pi05-run/pi0.5/bin/python scripts/collect_pi05_transcoder_features.py \
  --checkpoint /path/to/step_027233.pt \
  --output-dir outputs/features/pi05_libero/train_80_top20_exp16_lambda1e-4 \
  --episode-split 80,10,10 \
  --split train \
  --episode-split-seed 0 \
  --batch-size 8 \
  --collection-mode inference \
  --num-inference-steps 10 \
  --top-k 20 \
  --firing-threshold 1e-6 \
  --top-m-active 100 \
  --device cuda
```

For an initial ranked report:

```bash
/Users/akhidre/pubgit/pi05-run/pi0.5/bin/python scripts/make_pi05_feature_report.py \
  --feature-dir outputs/features/pi05_libero/train_80_top20_exp16_lambda1e-4 \
  --max-features 200 \
  --top-examples 20 \
  --sort-by interesting \
  --min-frequency 0.001 \
  --max-frequency 0.2
```

For large Modal feature artifacts, render the report on the Modal volume instead
of downloading `feature_topk.pt` locally:

```bash
/Users/akhidre/pubgit/pi05-run/pi0.5/bin/modal run scripts/modal_make_pi05_feature_report.py \
  --feature-dir /vol/outputs/features/pi05_libero/pilot_ep0-49_inference10_top20 \
  --max-features 200 \
  --top-examples 20 \
  --sort-by interesting
```

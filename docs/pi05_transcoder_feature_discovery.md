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

## Layer Flow Report

`scripts/make_pi05_transcoder_flow_report.py` renders the aggregate layer-to-layer
view used for the activation-flow probe:

```bash
python scripts/make_pi05_transcoder_flow_report.py \
  --feature-dir outputs/features/pi05_libero/transcoder-probe \
  --checkpoint /path/to/step_027233.pt \
  --top-features-per-layer 6
```

It writes:

```text
transcoder_flow_report.html
transcoder_flow_chart.svg
transcoder_flow_summary.json
transcoder_flow_layers.csv
```

Each node is one Pi0.5 action-expert MLP transcoder layer. Node intensity and
edge width summarize sparse latent activity accumulated across the collected
LIBERO observations and denoising passes. When the checkpoint is available, the
default attribution proxy is `E[z_i] * ||decoder_i||`; otherwise the report falls
back to `E[z_i]`. Edges are adjacent-layer aggregate statistics, not causal proof
of connectivity.

## Langfuse Tracing

Langfuse tracing is opt-in via environment variables and never requires secrets
inside notebooks or committed files:

```bash
export PI05_LANGFUSE_TRACE=1
export LANGFUSE_PUBLIC_KEY=...
export LANGFUSE_SECRET_KEY=...
export LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
```

The Colab notebook reads `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` from
private Colab Secrets. The runtime writes one trace per rollout policy chunk:

```text
pi05-transcoder-rollout-chunk
  capture-observation inputs: task text, tensor shapes, optional camera media
  trace-diffusion-trajectory: initial noise, denoise x_t, denoise velocity summaries
  decode-action-chunk: final 50-step action chunk and first 10 executed actions
  trace-transcoder-layer-activation: one span per action-expert layer with top sparse features
```

The offline feature pass writes one root trace for the aggregate collection and
child spans per dataset batch. The flow report logs its SVG/HTML/JSON/CSV output
paths and attaches the SVG chart when media upload is enabled.

Use `PI05_LANGFUSE_MAX_IMAGES`, `PI05_LANGFUSE_MAX_MEDIA_BYTES`,
`PI05_LANGFUSE_MAX_DIFFUSION_EVENTS`, and `PI05_LANGFUSE_MAX_LAYER_SPANS` to cap
trace size.

## Replacement Equivalence Metrics

Use the sanity comparison script after running both the original/probe and
replace sanity notebooks:

```bash
python scripts/compare_pi05_transcoder_sanity_runs.py \
  --base-dir outputs/eval/pi05_libero \
  --base-dir /content/drive/MyDrive/groot-run-shared-programmer908/outputs/eval/pi05_libero \
  --make-video
```

It writes:

```text
sanity_comparison_report.html
sanity_comparison_summary.json
action_chunk_comparison.csv
layer_l1_comparison.csv
```

This report compares closed-loop success, runtime, success step, trace
completeness, action chunks by rollout chunk index, and aggregate success-rate
statistics. Treat the success-rate statistics as descriptive unless the run has
enough paired episodes/tasks for meaningful inference.

For a stricter estimate of the action error added by replacement, run the paired
same-observation probe:

```bash
python scripts/eval_pi05_transcoder_action_equivalence.py \
  --checkpoint /path/to/step_027233.pt \
  --episodes 0,1,2,3,4,5,6,7,8,9 \
  --batch-size 2 \
  --max-batches 50 \
  --device cuda \
  --policy-dtype bfloat16
```

It writes:

```text
paired_action_metrics.csv
action_equivalence_summary.json
```

These metrics compare original/probe and replace action chunks on identical
dataset observations with the diffusion RNG restored between modes. This is the
preferred statistic for quantifying replacement error before simulator feedback
causes the two closed-loop trajectories to diverge.

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
transcoder_flow_report.html
transcoder_flow_chart.svg
transcoder_flow_summary.json
transcoder_flow_layers.csv
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

## GCP Transcoder Probe

The GCP runner builds the requested artifact folder and archive on the L4 VM:

```bash
LOCAL_TRANSCODER_CHECKPOINT=/path/to/step_027233.pt \
PI05_LANGFUSE_TRACE=1 \
./scripts/gcp_run_transcoder_probe.sh
```

Default remote outputs:

```text
~/groot-run/outputs/features/pi05_libero/transcoder-probe/
~/groot-run/transcoder-probe/transcoder-probe.tar.gz
```

The archive is fetched locally to:

```text
outputs/transcoder-probe/transcoder-probe.tar.gz
```

Set `GCS_URI=gs://bucket/prefix` to also upload the archive to Cloud Storage.

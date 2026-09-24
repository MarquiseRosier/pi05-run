# Pi0.5 Transcoder Circuit Tracing

This document summarizes what we implemented for circuit tracing after feature
discovery. The implementation follows the DifFRACT-style idea: replace target
MLPs with trained transcoders, use sparse features as computational nodes, and
trace feature-to-feature influence using local attribution.

## Starting Point

Feature discovery gives candidate sparse features such as:

```text
L12:tau1:F7584
```

This means:

```text
layer = 12
flow timestep tau = 1.0
feature id = 7584
```

Feature discovery also stores the Top-K activating LIBERO observations and the
winning action-token position for each observation:

```text
p*_e = argmax_p z_{layer,tau,p,feature}(e)
```

Circuit tracing starts from one discovered feature and asks:

```text
Which earlier sparse features causally influence this target feature
inside the transcoder replacement model?
```

## Why Use The Replacement Model

Feature discovery runs the transcoders in probe mode:

```text
original MLP output drives Pi0.5
transcoder runs beside the MLP and records sparse latent z
```

That is correct for finding interpretable features, but it is not enough for
circuit tracing. In probe mode, an earlier transcoder latent is only a side
measurement, not a causal input to later model layers.

For circuit tracing, we use replacement mode:

```text
transcoder output replaces each action-expert MLP output
sparse latent z is now on the computational path
```

Now gradients from a later target feature back to earlier sparse latents are
meaningful in the local replacement model.

## Transcoder Block

For action-expert MLP layer `l`, the trained transcoder approximates:

```text
TC_l(x_l, tau) ~= MLP_l(x_l)
```

where:

```text
x_l     = MLP input hidden state, shape [B, P, d_model]
P       = action-token positions, usually 50
d_model = 1024 for Pi0.5 action expert
tau     = flow timestep
```

The time-conditioned transcoder is:

```text
e_tau = sinusoidal_embedding(tau)
[scale(tau), shift(tau)] = TimeMLP_l(e_tau)

x_mod = x_l * (1 + scale(tau)[:, None, :]) + shift(tau)[:, None, :]

pre_l  = W_enc,l x_mod + b_enc,l
z_l    = ReLU(pre_l)
yhat_l = W_dec,l z_l + b_dec,l
```

For tracing, the important tensors are:

```text
pre_l[p, i] = feature preactivation
z_l[p, i]   = sparse feature activation
```

## Replacement Model Formula

The original action-expert block is approximately:

```text
h_l'     = h_l + attention_l(h_l, prefix, tau)
u_l      = norm_l(h_l', tau)
h_{l+1}  = h_l' + MLP_l(u_l)
```

The replacement model is:

```text
h_l'          = h_l + attention_l(h_l, prefix, tau)
u_l           = norm_l(h_l', tau)
pre_l, z_l    = encoder_l(FiLM_l(u_l, tau))
h_{l+1}^{LRM} = h_l' + decoder_l(z_l)
```

So the sparse latent `z_l` is not just recorded. It drives later layers through
the decoder.

## Target Scalar

For a target feature:

```text
T = (l_t, tau, i_t)
```

and one Top-K observation `e`, feature discovery provides:

```text
p*_e = argmax_p z_{l_t,tau,p,i_t}(e)
```

The scalar target for attribution is the target preactivation:

```text
target_e = pre_{l_t,tau,p*_e,i_t}
```

We use preactivation instead of post-ReLU activation because it gives a cleaner
local scalar for gradient attribution. The ReLU already tells us whether the
feature is active.

## Edge Attribution

For an earlier source feature:

```text
S = (l_s, tau, j), where l_s < l_t
```

the per-position local attribution is:

```text
C_{e,p,j -> T} =
    z_{l_s,tau,p,j}(e) *
    d target_e / d z_{l_s,tau,p,j}(e)
```

We collapse over action position by signed summation:

```text
C_{e,j -> T} = sum_p C_{e,p,j -> T}
```

For inspection metadata only, we also keep:

```text
p*_s = argmax_p |C_{e,p,j -> T}|
```

The edge mass used for ranking and pruning is the mean absolute attribution over
the target Top-K observations:

```text
edge_mass(S -> T) = mean_e |C_{e,j -> T}|
```

We also store the mean signed attribution:

```text
edge_signed(S -> T) = mean_e C_{e,j -> T}
```

## Default Algorithm: DifFRACT-Frontier Trace

The default implemented mode is:

```text
--trace-mode diffract-frontier
```

The algorithm is:

```text
Input:
  target feature T_root
  Top-K examples for T_root from feature discovery
  trained transcoder checkpoint
  frozen Pi0.5 policy

Initialize:
  graph nodes = {T_root}
  graph edges = {}
  frontier = {T_root}

Repeat until max_nodes expanded:
  1. Pick the highest-influence discovered node that has not been expanded.

  2. For each Top-K observation:
       run Pi0.5 in replacement mode at the target tau
       record pre_l and z_l for every action-expert transcoder

  3. For the selected target node:
       compute one VJP:
         d target_e / d z_{l_s,tau,p,j}
       for all earlier source layers, positions, and features.

  4. Convert gradients to feature attributions:
       C = z * grad
       collapse positions by sum_p
       aggregate over Top-K observations

  5. Add candidate parent edges whose:
       mean absolute attribution >= min_attribution

  6. Recompute node influence in the current graph.

After construction:
  7. Prune nodes by cumulative indirect influence.

  8. Prune edges by cumulative edge influence.

Output:
  JSON, CSV, HTML table, and SVG graph.
```

## Node Influence And Pruning

After every expansion, the graph has nonnegative edge masses:

```text
A[source, target] = edge_mass(source -> target)
```

We column-normalize incoming attribution:

```text
A_norm[:, target] = A[:, target] / sum_source A[source, target]
```

Then compute indirect influence:

```text
B = (I - A_norm)^-1 - I
node_influence = B @ one_hot(root_target)
```

The root target is assigned influence `1.0`.

Node pruning keeps the highest-influence nodes until they explain the requested
cumulative fraction:

```text
--node-cumulative-threshold 0.8
```

means keep enough non-root nodes to account for 80% of cumulative node influence.

Edge pruning uses:

```text
edge_influence(source -> target) =
    A_norm[source, target] * node_influence[target]
```

Then keeps the strongest edges until:

```text
--edge-cumulative-threshold 0.98
```

or whatever threshold was requested.

## Important Interpretation Point

`max_nodes` controls how many nodes are expanded, not the final graph size.

For example:

```text
--max-nodes 5
```

means:

```text
perform at most 5 expensive VJP expansions
```

It does not mean:

```text
return a graph with 5 nodes
```

Each expansion can discover thousands of parent features if:

```text
--source-policy all-earlier
--min-attribution is small
```

So a full scientific trace can be large. A compact presentation graph should use
stricter attribution/pruning thresholds.

## Fixed-K Debug Mode

We also kept a simpler debug mode:

```text
--trace-mode fixed-k
--parents-per-node 2
--max-depth 3
--source-policy previous-layer
```

This mode traces only the previous layer and keeps exactly `parents_per_node`
parents for each expanded node. It is useful for sanity checks, but the
DifFRACT-frontier mode is the preferred construction.

## Output Files

Each run writes:

```text
trace_config.json      run settings and target feature
nodes_summary.csv      one row per graph node
edges_summary.csv      one row per traced edge
graph.json             full graph with node and edge metadata
circuit_report.html    table-based browser report
circuit_graph.svg      square-node circuit diagram
circuit_graph.html     browser view of the SVG diagram
```

## Implemented Files

Core implementation:

```text
src/pi05_mi/circuit_tracing.py
```

Pi0.5 wrapper changes:

```text
src/pi05_mi/transcoders.py
src/pi05_mi/patch_pi05.py
```

Local tracing CLI:

```text
scripts/trace_pi05_transcoder_circuit.py
```

Modal launcher:

```text
scripts/modal_trace_pi05_transcoder_circuit.py
```

Synthetic smoke test:

```text
scripts/smoke_test_circuit_tracing.py
```

## Modal Smoke Runs

We tested target:

```text
L12:tau1:F7584
```

using the pilot feature discovery folder:

```text
/vol/outputs/features/pi05_libero/pilot_ep0-49_inference10_top20
```

and transcoder checkpoint:

```text
/vol/outputs/transcoders/pi05_libero/allframes_80-10-10_epoch1_b8_exp16_latest_lambda1e-4/step_027233.pt
```

### Full Smoke Trace

Settings:

```text
--trace-mode diffract-frontier
--max-nodes 5
--min-attribution 0.0001
--node-cumulative-threshold 0.8
--edge-cumulative-threshold 0.98
--source-policy all-earlier
```

Result:

```text
raw graph:    10,728 nodes, 20,188 edges
pruned graph: 1,858 nodes, 2,021 edges
```

Output:

```text
/vol/outputs/circuits/pi05_libero/L12_tau1_F7584
```

This run confirmed that the full remote workflow works:

```text
load Pi0.5
load trained transcoders
replace action-expert MLPs
run frontier VJP tracing
prune graph
write artifacts
```

The graph is too large for manual reading, but useful as a full attribution
artifact.

### Compact Presentation Trace

Settings:

```text
--trace-mode diffract-frontier
--max-nodes 5
--min-attribution 0.005
--node-cumulative-threshold 0.3
--edge-cumulative-threshold 0.7
--source-policy all-earlier
```

Result:

```text
raw graph:    43 nodes, 46 edges
pruned graph: 5 nodes, 4 edges
```

Output:

```text
/vol/outputs/circuits/pi05_libero_compact_min005_node03_edge07/L12_tau1_F7584
```

This is the readable graph we should inspect first.

## How To Interpret A Compact Circuit

A compact circuit is not the complete circuit. It is a thresholded view of the
dominant attribution paths.

The right phrasing is:

```text
Under these attribution and pruning thresholds, the dominant upstream circuit
for L12:tau1:F7584 contains these nodes and edges.
```

not:

```text
This is the entire causal circuit.
```

Next interpretation steps:

```text
1. Inspect the target feature Top-K examples.
2. Inspect Top-K examples for each parent node in the compact circuit.
3. Decide whether each parent looks object-specific, phase-specific,
   prompt-specific, or action-specific.
4. Validate important claims with steering or ablation.
```

## Tau Interpretation

The current circuit trace holds tau fixed. For target:

```text
L12:tau1:F7584
```

all traced nodes are also evaluated at:

```text
tau = 1.0
```

So the graph answers:

```text
At tau=1, which earlier sparse features influence this target?
```

It does not answer:

```text
How does the circuit evolve across tau?
```

To study tau, run the same feature id at multiple timesteps:

```text
L12:tau1:F7584
L12:tau0.9:F7584
L12:tau0.7:F7584
L12:tau0.5:F7584
L12:tau0.3:F7584
L12:tau0.1:F7584
```

Even if later tau values are not globally top-ranked in the feature dashboard,
they can still be useful for understanding how this feature changes over the
flow trajectory.

## Validation

Before committing, we verified:

```text
python -m py_compile src/pi05_mi/circuit_tracing.py \
  scripts/trace_pi05_transcoder_circuit.py \
  scripts/modal_trace_pi05_transcoder_circuit.py \
  scripts/smoke_test_circuit_tracing.py

PYTHONPATH=src python scripts/smoke_test_circuit_tracing.py

git diff --check
```

The smoke test checks:

```text
source * gradient attribution
multi-layer VJP attribution
cumulative node pruning
cumulative edge pruning
trace output writing
```

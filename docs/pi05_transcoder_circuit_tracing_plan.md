# Pi0.5 Transcoder Circuit Tracing Plan

This stage starts after feature discovery. Feature discovery gives candidate nodes such as

```text
L12:tau1:F7584
```

meaning layer 12, flow timestep 1.0, sparse feature 7584. The feature browser ranks these candidates and stores their Top-K activating LIBERO observations plus the winning action-token position `p*`.

## Why Circuit Tracing Needs The Replacement Model

Feature discovery used the frozen original Pi0.5 model in **probe mode**:

```text
original MLP output drives the model
transcoder runs beside it and records z
```

That is correct for finding features because it preserves the original model behavior.

Circuit tracing is different. We need gradients from a target sparse feature back to earlier sparse features. If the original MLPs still drive the model, an earlier transcoder latent is only an observation branch and is not causally upstream of the target. So for tracing we use the **local replacement model**, following the DifFRACT idea:

```text
transcoder output replaces each target MLP output
sparse latent z is now on the computational path
```

This makes feature-to-feature attribution meaningful inside the approximating sparse model.

## Transcoder Formula

For action-expert MLP layer `l`, the trained transcoder approximates the original MLP output:

```text
TC_l(x_l, tau) ~= MLP_l(x_l)
```

where:

```text
x_l   = MLP input hidden state, shape [B, P, d_model]
tau   = flow timestep
P     = Pi0.5 action-token positions, usually 50
d_model = 1024 for the Pi0.5 action expert
```

The timestep-conditioned transcoder is:

```text
e_tau = sinusoidal_embedding(tau)
[scale(tau), shift(tau)] = TimeMLP_l(e_tau)

x_mod = x_l * (1 + scale(tau)[:, None, :]) + shift(tau)[:, None, :]

pre_l = W_enc,l x_mod + b_enc,l
z_l   = ReLU(pre_l)
yhat_l = W_dec,l z_l + b_dec,l
```

The sparse feature activation is `z_l[p, i]`. The preactivation is `pre_l[p, i]`.

## Replacement Model Formula

The original action-expert block has the usual residual structure:

```text
h_l'     = h_l + attention_l(h_l, prefix, tau)
u_l      = norm_l(h_l', tau)
h_{l+1}  = h_l' + MLP_l(u_l)
```

In the replacement model, every selected action-expert MLP is replaced:

```text
h_l'         = h_l + attention_l(h_l, prefix, tau)
u_l          = norm_l(h_l', tau)
pre_l, z_l   = encoder_l(FiLM_l(u_l, tau))
h_{l+1}^{LRM} = h_l' + decoder_l(z_l)
```

So `z_l` is no longer just logged. It actually drives later layers through the decoder output.

## Target Node

For a discovered target feature

```text
T = (l_t, tau, i_t)
```

we use its Top-K observations from feature discovery. For each observation `e`, feature discovery already stored:

```text
p*_e = argmax_p z_{l_t,tau,p,i_t}
```

The scalar target for attribution is the target **preactivation**:

```text
target_e = pre_{l_t,tau,p*_e,i_t}
```

We use preactivation instead of post-ReLU activation because it gives a cleaner local linear target. The ReLU gate has already selected whether this feature is active.

## Edge Attribution

For a source feature in an earlier layer,

```text
S = (l_s, tau, j)
```

the per-position contribution to the target is:

```text
C_{e,p,j -> T} = z_{l_s,tau,p,j} * d target_e / d z_{l_s,tau,p,j}
```

Then we collapse over source action position by summing the signed contributions:

```text
C_{e,j -> T} = sum_p C_{e,p,j -> T}
```

For inspection only, we also retain the position with the largest absolute local contribution:

```text
p*_s = argmax_p |C_{e,p,j -> T}|
```

So `p*_s` helps us inspect where the strongest local evidence occurred, but it is not the collapse rule for the edge score.

Across the target Top-K observations, the fixed-K debug trace aggregates:

```text
mean_abs      = mean_e |C_{e,j -> T}|
mean_signed   = mean_e C_{e,j -> T}
std_signed    = std_e C_{e,j -> T}
freq_top      = fraction of examples where j is among the top local parents
edge_score    = mean_abs      # default ranking
```

The highest-scoring source features are the candidate parent nodes.

## DifFRACT-Frontier Trace

The default trace mode is the DifFRACT-style frontier algorithm.

For one expanded target node `T`, we run one vector-Jacobian product per Top-K
observation:

```text
d target_e / d z_{l_s,tau,p,j}
```

for all earlier source layers `l_s < l_t`, all action positions `p`, and all
features `j` recorded by the replacement model. The edge attribution is still:

```text
C_{e,j -> T} = sum_p z_{l_s,tau,p,j} * d target_e / d z_{l_s,tau,p,j}
```

The script discovers candidate parents whose mean absolute attribution across
the Top-K observations is at least:

```text
min_attribution
```

After each expansion, the graph recomputes indirect node influence using a
normalized attribution adjacency matrix:

```text
A[source, target] = absolute edge attribution mass
A_norm[:, target] = A[:, target] / sum_source A[source, target]
B = (I - A_norm)^-1 - I
node_influence = B @ one_hot(root_target)
```

Then the next frontier nodes are the highest-influence discovered nodes that
have not yet been expanded. `max_nodes` controls how many nodes are expanded,
which is also the approximate number of expensive VJP rounds.

After construction, the graph is pruned in two stages:

```text
node_cumulative_threshold = 0.80
```

keeps the highest-influence nodes accounting for 80% of cumulative node
influence, including the root target.

```text
edge_cumulative_threshold = 0.98
```

keeps the highest-influence edges accounting for 98% of cumulative edge
influence. The edge influence score is:

```text
edge_influence(source -> target) =
    A_norm[source, target] * node_influence[target]
```

This follows the spirit of DifFRACT: discover a broader sparse-feature graph
with VJPs, then prune by cumulative node and edge influence.

## Fixed-K Debug Trace

The older debug mode remains available:

```text
--trace-mode fixed-k
--parents-per-node 2
--max-depth 3
--source-policy previous-layer
```

This traces only the previous layer and keeps exactly `parents_per_node` parents
per expanded node. It is useful for smoke tests and intuition, but it is not the
preferred DifFRACT-style construction.

## Outputs

The tracing script writes:

```text
trace_config.json      run settings and target feature
nodes_summary.csv      one row per graph node
edges_summary.csv      one row per traced parent edge
graph.json             full graph with node and edge metadata
circuit_report.html    compact browser for inspecting the trace
circuit_graph.svg      simple square-node circuit diagram
circuit_graph.html     browser view of the SVG diagram
```

## First Recommended Run

Use the cup candidate from the feature browser:

```text
target = L12:tau1:F7584
top_examples = 20
trace_mode = diffract-frontier
max_nodes = 100
min_attribution = 1e-4
node_cumulative_threshold = 0.8
edge_cumulative_threshold = 0.98
num_inference_steps = 10
```

This should answer:

```text
Which sparse features across earlier action-expert layers influence this target?
Which discovered parent features have enough indirect influence to remain after pruning?
What compact directed graph summarizes the strongest feature-to-feature circuit?
```

Only after this produces a coherent trace should we run larger candidate sets or steering/ablation validation.

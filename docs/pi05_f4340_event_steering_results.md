# F4340 Event-Matched Steering Pilot

## Question

Can the speed-correlated transcoder feature `L15:tau0.3:F4340`, or its traced
upstream circuit, steer Pi0.5 toward faster drawer-closing actions?

This is an open-loop pilot on the LIBERO task:

```text
put the black bowl in the bottom drawer of the cabinet and close it
```

Episode 16 supplies the fast positive examples. Episodes 14, 15, 31, 32, and
36 supply slower negative examples. All observations are in the broad late
phase and are matched within one of four normalized drawer-closing event bins.
Each bin uses 10 positive observations and 20 negative observations, balanced
as four negatives from each slow episode.

## Circuits

| Setup | Max depth | Max expanded nodes | Min attribution | Node mass | Edge mass | Final nodes | Final edges |
|---|---:|---:|---:|---:|---:|---:|---:|
| Compact | 3 | 8 | 0.003 | 0.60 | 0.90 | 169 | 172 |
| Relaxed | 15 | 100 | 0.003 | 0.90 | 0.90 | 582 | 3,376 |

Both circuits use DifFRACT-frontier attribution with position collapsed by
signed summation, 20 top examples, and two per-example attribution candidates.

## Steering

The frozen Pi0.5 policy runs in transcoder replacement mode for 10 flow steps.
At `tau=0.3`, `set-max` steering raises selected sparse activations to the mean
maximum activation measured on the matched episode-16 positive examples.

Each run evaluates:

- target feature steering;
- parent-node steering;
- full traced-circuit steering;
- three matched random-feature controls;
- three matched random-circuit controls.

The same sampled noise is reused between baseline and intervention for each
observation.

## Metrics

For action chunk `a` with 50 positions, the open-loop speed score is

```text
v(a) = mean_t ||a[t, 0:3]||_2.
```

The primary steering effect is

```text
speed_delta = v(steered) - v(baseline).
```

Positive values indicate steering toward faster predicted motion. We also
report:

- relative speed change: `speed_delta / v(baseline)`;
- positive-speed gap closed:
  `(|v(positive)-v(baseline)| - |v(positive)-v(steered)|) / |v(positive)-v(baseline)|`;
- speed-increase rate: fraction of examples with positive `speed_delta`;
- action-change L2: intervention magnitude over the full action chunk;
- distance-to-positive-action improvement as a secondary, stricter metric;
- matched-control-adjusted speed delta.

## Per-Bin Results

| Circuit | Bin | Target speed delta | Target increase rate | Circuit speed delta | Circuit increase rate | Random-circuit mean | Circuit minus random |
|---|---|---:|---:|---:|---:|---:|---:|
| Compact | q1 | +0.000839 | 100% | +0.001401 | 60% | -0.003339 | +0.004740 |
| Compact | q2 | +0.000857 | 100% | +0.005021 | 90% | -0.002464 | +0.007485 |
| Compact | q3 | +0.000531 | 100% | +0.005175 | 100% | -0.001139 | +0.006314 |
| Compact | q4 | +0.000620 | 100% | +0.005089 | 90% | -0.000525 | +0.005614 |
| Relaxed | q1 | +0.000839 | 100% | -0.014618 | 0% | -0.006396 | -0.008222 |
| Relaxed | q2 | +0.000857 | 100% | -0.013126 | 10% | -0.004999 | -0.008128 |
| Relaxed | q3 | +0.000531 | 100% | -0.012710 | 20% | -0.002276 | -0.010434 |
| Relaxed | q4 | +0.000620 | 100% | -0.013878 | 0% | -0.000357 | -0.013521 |

## Aggregate Results

Results below pool the four event bins. Confidence intervals treat the five
slow episodes as the independent units.

| Intervention | Mean speed delta | Episode-clustered 95% CI | Relative speed change | Gap closed | Increase rate |
|---|---:|---:|---:|---:|---:|
| Target feature | +0.000712 | [+0.000586, +0.000838] | +0.0570% | +0.0600% | 100% |
| Compact parents | +0.003500 | [-0.000853, +0.007852] | +0.3355% | +0.2386% | 82.5% |
| Compact circuit | +0.004171 | [-0.000053, +0.008396] | +0.3889% | +0.2958% | 85.0% |
| Relaxed parents | -0.014270 | [-0.024186, -0.004354] | -1.1336% | -1.2138% | 6.25% |
| Relaxed circuit | -0.013583 | [-0.023401, -0.003766] | -1.0789% | -1.1557% | 7.5% |

Matched-control-adjusted effects are:

| Comparison | Adjusted speed delta | Episode-clustered 95% CI |
|---|---:|---:|
| Target minus random feature | +0.000904 | [+0.000744, +0.001065] |
| Compact circuit minus random circuit | +0.006038 | [+0.003725, +0.008352] |
| Relaxed circuit minus random circuit | -0.010076 | [-0.020370, +0.000217] |

## Interpretation

`F4340` has a consistent causal direction in this pilot: increasing it raises
the open-loop speed score on every evaluated observation and exceeds matched
random-feature controls. The magnitude is nevertheless very small and closes
only about 0.06% of the gap to the positive speed reference.

The compact traced circuit strengthens the desired direction, particularly in
event bins q2-q4, and exceeds size-matched random-circuit controls. Its raw
episode-clustered confidence interval narrowly includes zero, so this remains
pilot evidence rather than a definitive behavioral result.

The relaxed circuit is counterproductive. Steering hundreds of retained nodes
slows the action consistently, showing that a denser attribution graph is not
automatically a better steering set. It likely includes mixed-sign or
compensatory features that should not all be forced to positive-example maxima.

These are open-loop action-score results, not rollout success rates. Episode 16
is also the only fast positive trajectory, so speed remains partially
confounded with episode and scene identity. Closed-loop evaluation and more
independent fast trajectories are required for a publishable behavioral claim.

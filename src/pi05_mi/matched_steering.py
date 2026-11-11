"""Matched dense-probe and sparse-transcoder steering; no simulator imports."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class SteeringConfig:
    train_tasks: tuple[int, ...] = (0, 1, 2, 3, 4)
    validation_tasks: tuple[int, ...] = (5, 6)
    test_tasks: tuple[int, ...] = (7, 8, 9)
    train_starts: int = 10
    validation_starts: int = 3
    test_starts: int = 5
    layers: tuple[int, ...] = (0, 8, 17)
    sparse_k: int = 16
    budget: float = 0.1
    action_steps: int = 50
    inference_steps: int = 10
    seed: int = 20261111
    bootstrap_draws: int = 2000
    min_train_gap: float = .05

    def __post_init__(self):
        tasks = [*self.train_tasks, *self.validation_tasks, *self.test_tasks]
        if len(tasks) != len(set(tasks)) or any(t not in range(10) for t in tasks):
            raise ValueError('Task splits must be disjoint LIBERO-Spatial task IDs')
        if not all((self.train_tasks, self.validation_tasks, self.test_tasks, self.layers)):
            raise ValueError('Every split and the layer list must be nonempty')
        if len(set(self.layers)) != len(self.layers) or any(l not in range(18) for l in self.layers):
            raise ValueError('Layers must be unique action-expert layers in [0, 17]')
        if min(self.train_starts, self.validation_starts, self.test_starts, self.sparse_k,
               self.inference_steps, self.bootstrap_draws) < 1 or not 1 <= self.action_steps <= 50:
            raise ValueError('Invalid sample counts or action horizon')
        if not 0 < self.budget <= 1:
            raise ValueError('Intervention budget must be in (0, 1] of native output RMS')
        if not 0 <= self.min_train_gap < 2:
            raise ValueError('Invalid training contrast threshold')

    def cases(self):
        return [dict(case_id=f'task{task}_init{init}', task_id=task, init_state_id=init,
                     split=split, seed=self.seed + 1009 * task + init)
                for split, tasks, count in (
                    ('train', self.train_tasks, self.train_starts),
                    ('validation', self.validation_tasks, self.validation_starts),
                    ('test', self.test_tasks, self.test_starts))
                for task in tasks for init in range(count)]


def preference(eef, target, distractor):
    """[-1, 1]: relative proximity to fixed initial bowl positions, in world metres."""
    eef, target, distractor = map(lambda x: np.asarray(x, dtype=float), (eef, target, distractor))
    separation = float(np.linalg.norm(target - distractor))
    if separation < .01:
        raise ValueError('Bowl centres must be at least 1 cm apart')
    return float((np.linalg.norm(eef - distractor) - np.linalg.norm(eef - target)) / separation)


def steering_score(baseline, intervened, desired_sign):
    if desired_sign not in (-1, 1):
        raise ValueError('Desired sign must be -1 or +1')
    return 50.0 * desired_sign * (intervened['final_preference'] - baseline['final_preference'])


def _probe(x, labels):
    from sklearn.linear_model import LogisticRegression
    scale = x.std(0).clip(1e-6)
    mean = x.mean(0)
    model = LogisticRegression(C=.1, max_iter=2000, solver='liblinear', random_state=0)
    model.fit((x - mean) / scale, labels)
    weight = model.coef_[0] / scale
    return weight, float(model.intercept_[0] - mean @ weight), mean


def fit_layer_handles(dense, sparse, labels, *, layer, k, seed):
    """Train-only readouts and intervention directions. Sparse units remain native."""
    dense, sparse, labels = np.asarray(dense), np.asarray(sparse), np.asarray(labels)
    if set(labels.tolist()) != {-1, 1} or min(np.sum(labels == s) for s in (-1, 1)) < 2:
        raise ValueError('Need at least two training examples of each polarity')
    if dense.ndim != 2 or sparse.ndim != 2 or len(dense) != len(labels) or len(sparse) != len(labels):
        raise ValueError('Expected matched example-by-feature matrices')
    if not np.isfinite(dense).all() or not np.isfinite(sparse).all():
        raise ValueError('Activations must be finite')
    weight, intercept, mean = _probe(dense, labels)
    if np.linalg.norm(weight) < 1e-10:
        raise ValueError(f'Layer {layer}: dense probe has no direction')
    contrast = sparse[labels == 1].mean(0) - sparse[labels == -1].mean(0)
    sd = sparse.std(0)
    active = np.flatnonzero((sd > 1e-6) & (sparse.max(0) > 0))
    if len(active) < k:
        raise ValueError(f'Layer {layer}: only {len(active)} varying active sparse features, need {k}')
    order = active[np.argsort(-np.abs(contrast[active]) / sd[active], kind='stable')]
    indices = order[:k]
    if np.linalg.norm(contrast[indices]) < 1e-10:
        raise ValueError(f'Layer {layer}: sparse features have no polarity contrast')
    sw, si, _ = _probe(sparse[:, indices], labels)
    # Random sparse controls match train activity-frequency bins, not mostly-dead units.
    rng = np.random.default_rng(seed + layer)
    frequency = (sparse > 0).mean(0)
    unused = set(active.tolist()) - set(indices.tolist())
    random_indices = []
    for feature in indices:
        if not unused:
            raise ValueError('Not enough active features for a disjoint random control')
        candidates = np.array(sorted(unused))
        distance = np.abs(frequency[candidates] - frequency[feature])
        close = candidates[distance == distance.min()]
        chosen = int(rng.choice(close)); random_indices.append(chosen); unused.remove(chosen)
    random_direction = rng.normal(size=dense.shape[1]); random_direction /= np.linalg.norm(random_direction)
    return dict(layer=layer, dense=dict(direction=(weight / np.linalg.norm(weight)).tolist(),
                center=mean.tolist(), probe_weight=weight.tolist(), probe_intercept=intercept,
                random_direction=random_direction.tolist()),
                sparse=dict(indices=indices.tolist(), contrast=contrast[indices].tolist(),
                probe_weight=sw.tolist(), probe_intercept=si, random_indices=random_indices,
                random_contrast=(np.abs(contrast[indices]) * rng.choice([-1, 1], size=k)).tolist(),
                activity_frequency=frequency[indices].tolist(),
                random_activity_frequency=frequency[random_indices].tolist()))


def probe_auc(handle, family, values, labels):
    from sklearn.metrics import roc_auc_score
    arm = handle[family]
    x = np.asarray(values)
    if family == 'sparse':
        x = x[:, arm['indices']]
    return float(roc_auc_score(np.asarray(labels) == 1, x @ np.array(arm['probe_weight']) + arm['probe_intercept']))


def intervention_delta(y, z, decoder_weight, arm, *, family, mode, sign, budget):
    """Native y + delta, never a wholesale transcoder reconstruction replacement.

    Both arms have the same per-token output-RMS cap. Sparse changes are clipped
    to preserve nonnegative codes, then scaled down if needed; realized norms
    are returned so saturation is visible. Ablation removes the discovered
    component subject to that same cap.
    """
    if mode not in ('steer', 'random', 'ablate', 'random_ablate', 'zero'):
        raise ValueError(mode)
    if budget < 0 or sign not in (-1, 1) or family not in ('dense', 'sparse'):
        raise ValueError('Invalid intervention specification')
    yf = y.float()
    if mode == 'zero' or budget == 0:
        return torch.zeros_like(y), dict(mean_relative_rms=0.0, max_relative_rms=0.0)
    limit = budget * yf.square().mean(-1, keepdim=True).sqrt()
    tensor = lambda value: torch.as_tensor(value, device=y.device, dtype=torch.float32)
    random = mode in ('random', 'random_ablate')
    if family == 'dense':
        direction = tensor(arm['random_direction'] if random else arm['direction'])
        if mode in ('ablate', 'random_ablate'):
            delta = -((yf - tensor(arm['center'])) * direction).sum(-1, keepdim=True) * direction
        else:
            delta = sign * direction.expand_as(yf)
            delta = delta * limit / delta.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-12)
    else:
        indices = arm['random_indices'] if random else arm['indices']
        columns = decoder_weight[:, indices].float().to(y.device)
        current = z[..., indices].float()
        if mode in ('ablate', 'random_ablate'):
            dz = -current
        else:
            dz = sign * tensor(arm['random_contrast'] if random else arm['contrast']).expand_as(current)
            raw_delta = dz @ columns.T
            dz = dz * limit / raw_delta.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-12)
            dz = torch.maximum(dz, -current)
        delta = dz @ columns.T  # decoder bias cancels in decode(z + dz) - decode(z)
    rms = delta.square().mean(-1, keepdim=True).sqrt()
    delta = delta * torch.minimum(torch.ones_like(rms), limit / rms.clamp_min(1e-12))
    # Enforce the cap after model-dtype quantization as well as in float32.
    # A requested edit below one bfloat16 quantum may become an exact no-op.
    for _ in range(4):
        changed = (yf + delta).to(y.dtype)
        realized_rms = (changed.float() - yf).square().mean(-1, keepdim=True).sqrt()
        over = realized_rms > limit
        if not bool(over.any()):
            break
        delta = torch.where(over, delta * (.95 * limit / realized_rms.clamp_min(1e-12)), delta)
    changed = (yf + delta).to(y.dtype)
    over = (changed.float() - yf).square().mean(-1, keepdim=True).sqrt() > limit
    changed = torch.where(over, y, changed)
    actual = changed.float() - yf
    relative = actual.square().mean(-1).sqrt() / yf.square().mean(-1).sqrt().clamp_min(1e-12)
    fraction = actual.square().mean(-1, keepdim=True).sqrt() / rms.clamp_min(1e-12)
    return changed - y, dict(mean_relative_rms=float(relative.mean()), max_relative_rms=float(relative.max()),
                            mean_uncapped_delta_fraction=float(fraction.mean()))


class SteeringHooks:
    """Observe matched native MLP outputs/codes or intervene at one chosen layer."""
    def __init__(self, targets, transcoders, context):
        self.context, self.capture, self.spec = context, False, None
        self.values, self.norms, self.handles = {}, [], []
        for target in targets:
            if target.name in transcoders:
                self.handles.append(target.module.register_forward_hook(
                    self._hook(target.layer_index, transcoders[target.name])))

    def _hook(self, layer, transcoder):
        def hook(module, inputs, output):
            relevant = self.spec is not None and self.spec['handle']['layer'] == layer
            if not self.capture and not relevant:
                return output
            _, z = transcoder.encode(inputs[0], self.context.timestep_for(inputs[0]))
            if self.capture:
                for family, value in (('dense', output), ('sparse', z)):
                    self.values.setdefault(f'{family}_{layer}', []).append(value.detach().float().mean((0, 1)).cpu().numpy())
            if not relevant:
                return output
            spec = self.spec
            delta, norms = intervention_delta(output, z, transcoder.decoder.weight,
                spec['handle'][spec['family']], family=spec['family'], mode=spec['mode'],
                sign=spec['sign'], budget=spec['budget'])
            self.norms.append(norms)
            return output + delta
        return hook

    def reset(self, *, capture=False, spec=None):
        self.capture, self.spec = capture, spec
        self.values, self.norms = {}, []

    def arrays(self):
        return {key: np.mean(values, axis=0) for key, values in self.values.items()}

    def close(self):
        for handle in self.handles:
            handle.remove()


def select_layers(validation_rows, handles):
    """Choose ONE shared layer by both arms' mean validation movement effect."""
    if any(row['split'] != 'validation' for row in validation_rows):
        raise ValueError('Layer selection may only use validation cases')
    selected, scores = {}, {}
    for family in ('dense', 'sparse'):
        scores[family] = {}
        for handle in handles:
            rows = [r for r in validation_rows if r['family'] == family and r['layer'] == handle['layer']]
            if not rows:
                raise ValueError('Missing validation steering results')
            scores[family][str(handle['layer'])] = float(np.mean([r['score'] for r in rows]))
    best = max(handles, key=lambda h: (np.mean([scores[f][str(h['layer'])] for f in ('dense', 'sparse')]), -h['layer']))
    selected = {family: best for family in ('dense', 'sparse')}
    return selected, scores


def paired_report(rows, *, seed=0, draws=2000):
    """Pair both directions within each start; bootstrap held-out tasks together."""
    if not rows or any(r['split'] != 'test' for r in rows):
        raise ValueError('Final score requires held-out test results')
    primary = [r for r in rows if r['mode'] == 'steer' and r['dose'] == 1.0]
    if not primary:
        raise ValueError('No primary-dose paired test trials')
    cases = {}
    for row in primary:
        key = (row['case_id'], row['desired_sign'])
        cell = cases.setdefault(key, {'task_id': row['task_id']})
        if row['family'] in cell:
            raise ValueError('Duplicate primary trial')
        cell[row['family']] = row['score']
    if any(set(c) != {'task_id', 'dense', 'sparse'} for c in cases.values()):
        raise ValueError('Dense and sparse primary scores must be paired')
    starts = {}
    for (case, sign), cell in cases.items():
        starts.setdefault(case, {})[sign] = cell
    if any(set(pair) != {-1, 1} for pair in starts.values()):
        raise ValueError('Both steering directions are required for every test start')
    grouped = {}
    for pair in starts.values():
        task = pair[1]['task_id']
        grouped.setdefault(task, []).append(np.mean([pair[s]['sparse'] - pair[s]['dense'] for s in (-1, 1)]))
    task_deltas = np.array([np.mean(grouped[t]) for t in sorted(grouped)])
    interval = None
    if len(task_deltas) >= 3:
        rng = np.random.default_rng(seed)
        boot = rng.choice(task_deltas, size=(draws, len(task_deltas)), replace=True).mean(1)
        interval = np.quantile(boot, [.025, .975]).tolist()
    means = {family: float(np.mean([np.mean([c[family] for c in cases.values() if c['task_id'] == task])
                                    for task in sorted(grouped)])) for family in ('dense', 'sparse')}
    return dict(primary_score='50 * desired_sign * (intervened - baseline final relative bowl proximity)',
                dense_score=means['dense'], sparse_score=means['sparse'],
                sparse_minus_dense=float(task_deltas.mean()), task_bootstrap_95ci=interval,
                test_starts=len(starts), test_tasks=len(grouped),
                task_differences={str(t): float(np.mean(v)) for t, v in grouped.items()},
                interpretation='Task-balanced first-chunk movement effect; not task-success or safety certification. '
                'Few task clusters give imprecise intervals. Negative scores mean movement against the requested change.')


def protocol_hash(config, checkpoint_digest, source_digest):
    return hashlib.sha256(json.dumps(dict(config=asdict(config), checkpoint=checkpoint_digest,
        source=source_digest), sort_keys=True).encode()).hexdigest()

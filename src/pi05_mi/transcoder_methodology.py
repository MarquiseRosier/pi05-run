"""Controlled, small-budget training-method comparison; no steering claims."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from pi05_mi.transcoders import TimeConditionedTranscoder, TimeConditionedTranscoderConfig
from pi05_mi.transcoder_audit import audit_json, file_sha256, task_interval


ARMS = ('A_noise_sampled', 'B_noise_grid', 'C_trajectory', 'D_paper_adapted')


@dataclass(frozen=True)
class MethodConfig:
    layer: int = 8
    train_tasks: tuple[int, ...] = (0, 1, 2, 3, 4)
    validation_tasks: tuple[int, ...] = (5, 6)
    test_tasks: tuple[int, ...] = (7, 8, 9)
    train_starts: tuple[int, ...] = (0, 1, 2, 3)
    eval_starts: tuple[int, ...] = (0, 1, 2)
    noise_draws: int = 2
    fit_seeds: tuple[int, ...] = (41, 42, 43)
    inference_steps: int = 10
    action_prefix: int = 10
    updates: int = 1000
    batch_size: int = 512
    checkpoint_every: int = 250
    lambda_l1: float = 1e-4
    seed: int = 20261112

    def cases(self):
        groups = (self.train_tasks, self.validation_tasks, self.test_tasks)
        if any(not g or len(set(g)) != len(g) for g in groups):
            raise ValueError('Each task split must be nonempty and unique')
        if any(set(groups[i]) & set(groups[j]) for i in range(3) for j in range(i)):
            raise ValueError('Task splits overlap')
        if (self.layer < 0 or self.lambda_l1 < 0 or
            not self.fit_seeds or len(set(self.fit_seeds)) != len(self.fit_seeds) or
            min(self.noise_draws, self.inference_steps, self.action_prefix, self.updates,
                self.batch_size, self.checkpoint_every) < 1 or self.batch_size < 2):
            raise ValueError('Invalid experiment counts or regularization')
        result = []
        for split, tasks in zip(('train', 'validation', 'test'), groups):
            starts = self.train_starts if split == 'train' else self.eval_starts
            if not starts or min((*starts, *tasks)) < 0 or len(set(starts)) != len(starts):
                raise ValueError('Invalid initial states or task IDs')
            for task in tasks:
                for start in starts:
                    for noise in range(self.noise_draws):
                        result.append(dict(case_id=f'task{task}_init{start}_noise{noise}', split=split,
                            task_id=task, init_state_id=start, noise_index=noise,
                            scene_seed=self.seed + 1009*task + start,
                            noise_seed=self.seed + 100003*task + 101*start + noise))
        return result


class PaperAdaptedTranscoder(TimeConditionedTranscoder):
    """Appendix E initialization/time network; VLA embedding and lambda remain adaptations."""
    def __init__(self, config):
        super().__init__(config)
        d = config.time_dim
        self.time_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d),
                                      nn.SiLU(), nn.Linear(d, 2*config.d_model))
        for module in self.time_mlp:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.time_mlp[-1].weight)
        with torch.no_grad():
            self.encoder.weight.copy_(self.decoder.weight.T)
        self.renormalize_decoder_columns()


def method_model(base_config, arm, seed):
    if arm not in ARMS:
        raise ValueError(arm)
    cfg = dict(base_config)
    if arm == ARMS[3]:
        cfg.update(time_embedding_dim=256, time_hidden_dim=256, eps=1e-6)
    # CPU construction is paired and does not change teacher CUDA RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        cls = PaperAdaptedTranscoder if arm == ARMS[3] else TimeConditionedTranscoder
        return cls(TimeConditionedTranscoderConfig(**cfg))


def method_loss(model, x, t, y, common_variance, arm, lambda_l1):
    prediction, latent = model(x, t)
    if arm == ARMS[3]:
        denominator = y.float().var(0, unbiased=False).sum() + 1e-6
    else:
        denominator = torch.as_tensor(common_variance, device=y.device).clamp_min(1e-8)
    mse = (prediction.float()-y.float()).square().sum(-1).mean()/denominator
    l1 = latent.float().abs().sum(-1).mean()
    return mse + lambda_l1*l1, mse, (latent > 0).float().sum(-1).mean()


def atomic_torch_save(payload, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.partial')
    torch.save(payload, tmp); tmp.replace(path)


def records_for(root, cases, split, mode):
    """All raw records, float32 on CPU; no ring truncation or stochastic selection."""
    chunks = []
    for case in cases:
        if case['split'] != split:
            continue
        folder = Path(root)/'cases'/case['case_id']
        receipt = json.loads((folder/'collected.json').read_text())
        path = folder/(mode+'.npz')
        if file_sha256(path) != receipt['files'][path.name]:
            raise ValueError(f'Activation file changed: {path}')
        with np.load(path) as data:
            chunks.append({k: torch.from_numpy(data[k].copy()) for k in ('x', 'y', 't')})
    if not chunks:
        raise ValueError(f'No {split} records')
    return {k: torch.cat([c[k] for c in chunks]) for k in chunks[0]}


@torch.no_grad()
def reconstruction_metrics(model, records, denominator, batch_size=512):
    device = next(model.parameters()).device
    sse = l0 = 0.; n = len(records['t']); per_time = {}
    for offset in range(0, n, batch_size):
        x, y, t = (records[k][offset:offset+batch_size].to(device) for k in ('x', 'y', 't'))
        prediction, z = model(x, t)
        if not torch.isfinite(prediction).all() or not torch.isfinite(z).all():
            raise ValueError('Nonfinite dictionary evaluation')
        errors = (prediction.double()-y.double()).square().sum(-1)
        sse += float(errors.sum()); l0 += float((z > 0).sum())
        for value in t.unique():
            key = f'{float(value):.6f}'; mask = t == value
            bucket = per_time.setdefault(key, [0., 0])
            bucket[0] += float(errors[mask].sum()); bucket[1] += int(mask.sum())
    return dict(nmse=sse/n/denominator, mean_l0=l0/n, records=n,
                per_time={k: dict(nmse=v[0]/v[1]/denominator, records=v[1]) for k,v in per_time.items()})


def fit_method(root, config, base_config, arm, seed, train, validation, denominator,
               deadline, device='cuda'):
    """Atomic checkpoint boundaries; resume RNG/Adam state, fixed final-step comparison."""
    folder = Path(root)/'fits'/f'{arm}_seed{seed}'; folder.mkdir(parents=True, exist_ok=True)
    done = folder/'complete.json'
    if done.exists():
        receipt = json.loads(done.read_text())
        if file_sha256(folder/'model.pt') != receipt['sha256']:
            raise ValueError('Completed model checksum changed')
        return True
    model = method_model(base_config, arm, seed).to(device)
    paper = arm == ARMS[3]
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4 if paper else 1e-4,
                                 weight_decay=0. if paper else .01)
    generator = torch.Generator().manual_seed(seed + 90000)
    state_path = folder/'resume.pt'; start = 0; history = []
    if state_path.exists():
        saved = torch.load(state_path, map_location='cpu', weights_only=True)
        if saved['arm'] != arm or saved['seed'] != seed:
            raise ValueError('Wrong fit resume identity')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        generator.set_state(saved['sampler_state']); start = saved['step']; history = saved['history']
    elif time.monotonic() < deadline:
        history.append(dict(step=0, validation=reconstruction_metrics(model, validation, denominator)))
    def save_progress(step):
        atomic_torch_save(dict(arm=arm, seed=seed, step=step, model=model.state_dict(),
            optimizer=optimizer.state_dict(), sampler_state=generator.get_state(), history=history), state_path)
        audit_json(folder/'learning_curve.json', history)
    step = start
    try:
        while step < config.updates:
            if time.monotonic() >= deadline:
                save_progress(step); return False
            indices = torch.randint(len(train['t']), (config.batch_size,), generator=generator)
            x, y, t = (train[k][indices].to(device) for k in ('x', 'y', 't'))
            if paper:
                optimizer.param_groups[0]['lr'] = 2e-4*.5*(1+math.cos(math.pi*step/config.updates))
            optimizer.zero_grad(set_to_none=True)
            loss, mse, l0 = method_loss(model, x, t, y, denominator, arm, config.lambda_l1)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); model.renormalize_decoder_columns(); step += 1
            if step % config.checkpoint_every == 0 or step == config.updates:
                history.append(dict(step=step, train_loss=float(loss.detach()), train_nmse=float(mse.detach()),
                    train_l0=float(l0), grad_norm=float(grad_norm), lr=optimizer.param_groups[0]['lr'],
                    validation=reconstruction_metrics(model, validation, denominator)))
                save_progress(step)
                print(f'{arm} seed={seed}: {step}/{config.updates}, validation nMSE={history[-1]["validation"]["nmse"]:.5f}', flush=True)
        atomic_torch_save(dict(config=asdict(model.config), state_dict=model.state_dict(), arm=arm,
                              seed=seed, step=step), folder/'model.pt')
        audit_json(done, dict(sha256=file_sha256(folder/'model.pt'), updates=step,
            records_presented=step*config.batch_size, unique_available_records=len(train['t']), history=history))
        state_path.unlink(missing_ok=True)
        return True
    finally:
        del model, optimizer
        if str(device).startswith('cuda'): torch.cuda.empty_cache()


def paired_method_interval(rows, left, right, metric, fit_seeds):
    """Average paired fit/noise repeats within task; bootstrap tasks, not tokens."""
    index = {(r['model'], r['case']['case_id']): r for r in rows}
    case_ids = sorted({r['case']['case_id'] for r in rows})
    pairs = []
    for case_id in case_ids:
        for seed in fit_seeds:
            l, r = index[(f'{left}_seed{seed}', case_id)], index[(f'{right}_seed{seed}', case_id)]
            if l[metric] is not None and r[metric] is not None:
                pairs.append(dict(task_id=l['case']['task_id'], seed=seed, delta=l[metric]-r[metric]))
    result = task_interval(pairs, 'delta')
    result['excluded_undefined_pairs'] = len(case_ids)*len(fit_seeds)-len(pairs)
    result['per_seed_mean'] = {str(s): task_interval([p for p in pairs if p['seed']==s], 'delta')['task_balanced_mean'] for s in fit_seeds}
    return result


def method_report(root, config):
    root = Path(root); cases = config.cases(); test = [c for c in cases if c['split']=='test']
    names = ['existing']+[f'{a}_seed{s}' for s in config.fit_seeds for a in ARMS]
    rows = []
    for case in test:
        for name in names:
            path = root/'evaluation'/case['case_id']/(name+'.json')
            if path.exists(): rows.append(json.loads(path.read_text()))
    complete = len(rows) == len(test)*len(names)
    report = dict(status='complete' if complete else 'partial', completed_model_cases=len(rows),
        planned_model_cases=len(test)*len(names), test_tasks=len(config.test_tasks),
        test_starts=len(config.test_tasks)*len(config.eval_starts), noise_draws_per_start=config.noise_draws,
        fit_seeds=list(config.fit_seeds), training_updates=config.updates,
        records_presented_per_fit=config.updates*config.batch_size, comparisons={}, models={}, controls={}, training_progress={},
        limitations=[
            'Small-budget layer-specific adaptation, not a full DifFRACT reproduction or historical training replay.',
            'A/B/C share the original architecture/init/Adam settings but use a fixed shared normalizer and cached uniform batches.',
            'All new activation records use float32, unlike the original trainer buffer default float16; new-arm contrasts hold this fixed.',
            'D changes a bundle of time architecture, initialization, optimization and normalization; individual causes are not isolated.',
            'D keeps VLA-specific sinusoidal periods, lambda and clipping; smaller batches/data/budget than the paper.',
            'Original checkpoint has a different training history; fresh-arm contrasts isolate methodology, old/new contrasts do not.',
            'Reported nMSE uses the shared training-trajectory variance; do not compare numerically with an earlier audit using a different denominator.',
            'Only initial simulator observations; no actions executed, attribution validation, steering or safety claims.',
            'Task bootstrap is conditional on these fit seeds and observations; three test task clusters give weak population inference.',
            'States are held out from these new fits, not necessarily from policy or old transcoder pretraining.',
            'A null result at this budget does not establish that retraining is unnecessary; inspect learning curves and L0.',
        ])
    controls = []
    for case in test:
        path = root/'evaluation'/case['case_id']/'controls.json'
        if path.exists():
            control = json.loads(path.read_text())
            controls.append(dict(task_id=case['task_id'], **{k:control[k]['prefix']['relative_l2'] for k in ('repeat','zero','mean')}))
    report['completed_control_cases'] = len(controls)
    report['controls'] = {k:task_interval(controls,k) for k in ('repeat','zero','mean')}
    if complete and len(controls) != len(test):
        raise ValueError('Model results exist without complete sensitivity controls')
    if complete:
        for name in names[1:]:
            path = root/'fits'/name/'complete.json'
            saved = json.loads(path.read_text()); history = saved['history']
            val = [r['validation']['nmse'] for r in history]
            report['training_progress'][name] = dict(updates=saved['updates'],
                unique_available_records=saved['unique_available_records'], records_presented=saved['records_presented'],
                initial_validation_nmse=val[0],final_validation_nmse=val[-1],
                last_validation_change=val[-1]-val[-2] if len(val)>1 else None)
        for name in names:
            subset = [dict(r, task_id=r['case']['task_id']) for r in rows if r['model']==name]
            report['models'][name] = {m: task_interval(subset, m) for m in ('nmse','mean_l0','action_relative_l2')}
        for left, right in ((ARMS[1],ARMS[0]),(ARMS[2],ARMS[1]),(ARMS[3],ARMS[2])):
            report['comparisons'][left+'_minus_'+right] = {
                m: paired_method_interval(rows,left,right,m,config.fit_seeds) for m in ('nmse','action_relative_l2','mean_l0')}
        report['interpretation'] = ('Negative nMSE/action differences favor the left arm. Read together with L0, '
            'per-seed consistency, learning curves, repeat floor and zero/mean controls. '
            'Improvement supports further retraining at this layer/budget; no automatic deployment or attribution verdict.')
    else:
        report['interpretation'] = 'Incomplete matched experiment: no primary arm comparison released. Resume the same run.'
    audit_json(root/'report.json', report)
    lines = ['# Training-method decision pilot', '', report['interpretation'], '',
             f'Model-cases: {len(rows)}/{len(test)*len(names)}. Test: {report["test_starts"]} starts, '
             f'{len(config.test_tasks)} tasks, {config.noise_draws} noise draws/start.', '',
             '| Model | Inference nMSE | Active features/token | Relative action error |', '|---|---:|---:|---:|']
    fmt = lambda v: 'undefined' if v is None else f'{v:.5f}'
    for name, value in report['models'].items():
        lines.append(f'| {name} | {fmt(value["nmse"]["task_balanced_mean"])} | {fmt(value["mean_l0"]["task_balanced_mean"])} | {fmt(value["action_relative_l2"]["task_balanced_mean"])} |')
    lines += ['', '## Paired comparisons (negative error differences favor left arm)', '',
              '| Comparison | Δ nMSE | Task-bootstrap 95% interval | Δ action error |', '|---|---:|---|---:|']
    for name, values in report['comparisons'].items():
        lines.append(f'| {name} | {fmt(values["nmse"]["task_balanced_mean"])} | {values["nmse"]["task_bootstrap_95ci"]} | {fmt(values["action_relative_l2"]["task_balanced_mean"])} |')
    lines += ['', '## Layer sensitivity controls', '', '| Control | Relative action error | Cases |', '|---|---:|---:|']
    for name,value in report['controls'].items():
        lines.append(f'| {name} | {fmt(value["task_balanced_mean"])} | {value["observations"]} |')
    lines += ['', *['- '+s for s in report['limitations']]]
    (root/'summary.md').write_text('\n'.join(lines)+'\n')
    return report

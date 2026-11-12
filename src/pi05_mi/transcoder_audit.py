"""Checkpoint and streaming fidelity diagnostics; no LeRobot or simulator imports."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from pi05_mi.transcoders import TimeConditionedTranscoder, TimeConditionedTranscoderConfig


@dataclass(frozen=True)
class AuditConfig:
    tasks: tuple[int, ...] = (0, 7, 8, 9)
    starts: tuple[int, ...] = (0, 1, 2)
    seeds_per_start: int = 1
    inference_steps: int = 10
    replacement_layers: tuple[int, ...] = (0, 8, 17)
    replace_all: bool = True
    action_prefix: int = 10
    seed: int = 20261112

    def cases(self):
        if (not self.tasks or not self.starts or not self.replacement_layers or
            len(set(self.tasks)) != len(self.tasks) or len(set(self.starts)) != len(self.starts) or
            len(set(self.replacement_layers)) != len(self.replacement_layers) or
            min((*self.tasks, *self.starts, *self.replacement_layers)) < 0 or
            min(self.seeds_per_start, self.inference_steps, self.action_prefix) < 1):
            raise ValueError('Audit IDs must be unique/nonnegative; counts must be positive')
        return [dict(case_id=f'task{t}_init{i}_noise{n}', task_id=t, init_state_id=i,
                     noise_index=n, scene_seed=self.seed+1009*t+i,
                     noise_seed=self.seed+100003*t+101*i+n)
                for t in self.tasks for i in self.starts for n in range(self.seeds_per_start)]


def audit_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(encoded+'\n')
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value):
    a = value.detach().float().cpu().contiguous().numpy()
    return hashlib.sha256(str(a.shape).encode()+a.tobytes()).hexdigest()


def batch_sha256(batch):
    digest=hashlib.sha256()
    def visit(value):
        if isinstance(value,torch.Tensor):
            digest.update(str(value.dtype).encode()); digest.update(tensor_sha256(value).encode())
        elif isinstance(value,dict):
            for key in sorted(value):digest.update(str(key).encode());visit(value[key])
        elif isinstance(value,(list,tuple)):
            digest.update(str(len(value)).encode())
            for item in value:visit(item)
        elif isinstance(value,np.ndarray):
            digest.update(str((value.shape,value.dtype)).encode());digest.update(value.tobytes())
        else:digest.update(repr(value).encode())
    visit(batch)
    return digest.hexdigest()


def checkpoint_evidence(checkpoint):
    """Inspect optimizer evidence without treating checkpoint names as updates."""
    configs, states = checkpoint['configs'], checkpoint['state_dicts']
    if set(configs) != set(states) or not configs:
        raise ValueError('Checkpoint configurations and dictionaries do not match')
    layers = []
    for name, cfg in configs.items():
        weights = states[name]
        bad = [key for key, value in weights.items() if not bool(torch.isfinite(value).all())]
        decoder = weights['decoder.weight'].float()
        norms = decoder.norm(dim=0)
        finite_number=lambda v:float(v) if bool(torch.isfinite(v)) else None
        layers.append(dict(name=name, layer=int(name.split('.')[-2]), config=cfg,
            finite=not bad, nonfinite_tensors=bad, parameters=sum(v.numel() for v in weights.values()),
            decoder_norm_min=finite_number(norms.min()), decoder_norm_mean=finite_number(norms.mean()),
            decoder_norm_max=finite_number(norms.max())))
    optimizer = checkpoint.get('optimizer_state_dict')
    parameter_rows = []
    groups = optimizer.get('param_groups', []) if optimizer else []
    ids = [p for group in groups for p in group['params']]
    for pid in ids:
        state = optimizer.get('state', {}).get(pid, {})
        step = state.get('step')
        step = float(step) if step is not None else None
        if step is not None and not np.isfinite(step):
            step = None
        parameter_rows.append(dict(parameter_id=pid, step=step))
    steps = [r['step'] for r in parameter_rows if r['step'] is not None]
    if optimizer is None:
        update_status = 'unknown_optimizer_not_saved'
    elif not optimizer.get('state'):
        update_status = 'no_optimizer_updates_recorded'
    elif ids and len(steps) == len(ids) and min(steps) > 0:
        update_status = 'positive_steps_for_all_saved_optimizer_parameters'
    else:
        update_status = 'incomplete_or_zero_step_evidence'
    history = checkpoint.get('metrics_history', [])
    train_rows = [r for r in history if 'event' not in r and 'minibatches' in r]
    bad_metrics=[dict(row=i,key=k,value=str(v)) for i,row in enumerate(history)
                 for k,v in row.items() if isinstance(v,float) and not np.isfinite(v)]
    last_metrics=[{k:(str(v) if isinstance(v,float) and not np.isfinite(v) else v)
                  for k,v in row.items()} for row in history[-5:]]
    return dict(collection_mode=checkpoint.get('collection_mode'),
        collection_feed_forwards=checkpoint.get('feed_forward'),
        policy_path=checkpoint.get('policy_path'), num_inference_steps=checkpoint.get('num_inference_steps'),
        lambda_l1=checkpoint.get('lambda_l1'), layers=layers,
        all_dictionary_weights_finite=all(r['finite'] for r in layers),
        optimizer_update_status=update_status, optimizer_parameter_count=len(ids),
        optimizer_positive_step_parameters=sum(s is not None and s > 0 for s in [r['step'] for r in parameter_rows]),
        optimizer_step_min=min(steps) if steps else None, optimizer_step_max=max(steps) if steps else None,
        optimizer_steps=parameter_rows,
        history_rows=len(history), training_rows=len(train_rows),
        history_total_layer_minibatches=sum(r.get('minibatches') or 0 for r in train_rows
                                           if r.get('minibatches') is None or np.isfinite(r['minibatches'])),
        history_rows_without_updates=sum(not r.get('minibatches') for r in train_rows),
        last_metrics=last_metrics, nonfinite_history_metrics=bad_metrics,
        limitations=['Optimizer evidence is not convergence or semantic validity.',
                    'Optimizer IDs are not assigned to layer names without an original parameter-order manifest.',
                    'Collection step numbers are not optimizer step counts.',
                    'No training split, teacher weight hash or runtime version is inferred from the filename.'])


class FitBucket:
    """Mergeable sufficient statistics: all records counted, no activation ring."""
    def __init__(self):
        self.n = 0
        self.mean = None
        self.m2 = self.sse = self.energy = self.l0 = 0.0
        self.x_energy = self.max_abs_x = self.max_abs_y = 0.0
        self.fp16_overflow_elements = 0
        self.calls = 0
        self.t_min, self.t_max = None, None

    def update(self, x, y, predicted, z, timestep):
        arrays = [v.detach().float().reshape(-1, v.shape[-1]) for v in (x, y, predicted, z)]
        if any(not bool(torch.isfinite(v).all()) for v in arrays):
            raise ValueError('Non-finite activation, reconstruction or sparse code; refusing a fidelity score')
        xx, yy, pp, zz = arrays
        if xx.shape != yy.shape or yy.shape != pp.shape or zz.shape[0] != yy.shape[0]:
            raise ValueError('Unmatched activation records')
        yd = yy.double()
        count, mean = len(yy), yd.mean(0).cpu().numpy()
        batch_m2 = float(((yd-yd.mean(0))**2).sum())
        self.merge_moments(count, mean, batch_m2)
        self.sse += float(((pp.double()-yd)**2).sum())
        self.energy += float(yd.square().sum())
        self.x_energy += float(xx.double().square().sum())
        self.l0 += float((zz > 0).sum())
        self.max_abs_x = max(self.max_abs_x, float(xx.abs().max()))
        self.max_abs_y = max(self.max_abs_y, float(yy.abs().max()))
        self.fp16_overflow_elements += int((xx.abs()>65504).sum())+int((yy.abs()>65504).sum())
        self.calls += 1
        self.t_min = timestep if self.t_min is None else min(self.t_min, timestep)
        self.t_max = timestep if self.t_max is None else max(self.t_max, timestep)
        return (zz > 0).sum(0).cpu().numpy().astype(np.int64)

    def merge_moments(self, n, mean, m2):
        if not self.n:
            self.n, self.mean, self.m2 = n, mean.copy(), m2
            return
        delta = mean-self.mean
        total = self.n+n
        self.m2 += m2+float(delta @ delta)*self.n*n/total
        self.mean += delta*n/total
        self.n = total

    def merge(self, other):
        if not other.n:
            return
        self.merge_moments(other.n, other.mean, other.m2)
        for key in ('sse', 'energy', 'l0', 'x_energy', 'fp16_overflow_elements', 'calls'):
            setattr(self, key, getattr(self,key)+getattr(other,key))
        for key in ('max_abs_x', 'max_abs_y'):
            setattr(self,key,max(getattr(self,key),getattr(other,key)))
        self.t_min = other.t_min if self.t_min is None else min(self.t_min,other.t_min)
        self.t_max = other.t_max if self.t_max is None else max(self.t_max,other.t_max)

    def row(self, common_variance=None):
        variance = self.m2/self.n if self.n else 0
        return dict(records=self.n, calls=self.calls, sse=self.sse, target_energy=self.energy,
            mse_per_vector=self.sse/self.n if self.n else None,
            own_variance_nmse=self.sse/self.m2 if self.m2>1e-12 else None,
            common_variance_nmse=self.sse/self.n/common_variance if self.n and common_variance and common_variance>1e-12 else None,
            variance_sum=variance, relative_rmse=(self.sse/self.energy)**.5 if self.energy>1e-12 else None,
            l0_mean=self.l0/self.n if self.n else None,
            input_rms=(self.x_energy/self.n/len(self.mean))**.5 if self.n else None,
            target_rms=(self.energy/self.n/len(self.mean))**.5 if self.n else None,
            max_abs_x=self.max_abs_x, max_abs_y=self.max_abs_y,
            fp16_overflow_elements=self.fp16_overflow_elements, t_min=self.t_min,t_max=self.t_max)


class AuditAccumulator:
    def __init__(self):
        self.buckets, self.occupancy, self.first_output_hashes = {}, {}, {}

    def update(self, mode, layer, x, y, predicted, z, t, call):
        # Fixed width flow-time bins, independent of sampled/trajectory call order.
        tau_bin = min(9,max(0,int(np.ceil(float(t)*10-1e-6))-1))
        key = f'{mode}|{layer}|{tau_bin}'
        bucket = self.buckets.setdefault(key,FitBucket())
        counts = bucket.update(x,y,predicted,z,float(t))
        occupancy_key = f'{mode}|{layer}'
        self.occupancy[occupancy_key] = self.occupancy.get(occupancy_key,np.zeros_like(counts))+counts
        if call == 0:
            self.first_output_hashes[occupancy_key] = tensor_sha256(y)

    def save(self, path):
        arrays = {f'occupancy::{k}':v for k,v in self.occupancy.items()}
        scalars = {}
        for key,b in self.buckets.items():
            arrays[f'mean::{key}'] = b.mean
            scalars[key] = {k:v for k,v in vars(b).items() if k!='mean'}
        arrays['metadata'] = np.array(json.dumps(dict(buckets=scalars,hashes=self.first_output_hashes)))
        temporary = Path(path).with_suffix('.tmp')
        with temporary.open('wb') as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        result = cls()
        with np.load(path,allow_pickle=False) as data:
            metadata=json.loads(str(data['metadata']))
            result.first_output_hashes=metadata['hashes']
            for key,values in metadata['buckets'].items():
                b=FitBucket(); b.__dict__.update(values); b.mean=data[f'mean::{key}'].copy()
                result.buckets[key]=b
            result.occupancy={k.removeprefix('occupancy::'):data[k].copy() for k in data.files if k.startswith('occupancy::')}
        return result

    def merge(self, other):
        for key,b in other.buckets.items():
            self.buckets.setdefault(key,FitBucket()).merge(b)
        for key,counts in other.occupancy.items():
            self.occupancy[key]=self.occupancy.get(key,np.zeros_like(counts))+counts


def action_error(reference, other, prefix):
    a,b=np.asarray(reference,dtype=float),np.asarray(other,dtype=float)
    if a.shape!=b.shape or a.ndim!=2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid or mismatched action chunks')
    def stats(a,b):
        difference=b-a; norm=np.linalg.norm(a)
        return dict(rmse=float(np.sqrt(np.mean(difference**2))),
            relative_l2=float(np.linalg.norm(difference)/norm) if norm>1e-12 else None,
            max_abs=float(np.max(np.abs(difference))))
    return dict(full=stats(a,b),prefix=stats(a[:prefix],b[:prefix]),
                prefix_steps=min(prefix,len(a)),exactly_equal=bool(np.array_equal(a,b)))


def task_interval(rows, field, *, seed=0):
    by_task={}
    for row in rows:
        value=row.get(field)
        if value is not None and np.isfinite(value):
            by_task.setdefault(row['task_id'],[]).append(value)
    means=[float(np.mean(v)) for v in by_task.values()]
    ci=None
    if len(means)>=3:
        boot=np.random.default_rng(seed).choice(means,size=(4000,len(means)),replace=True).mean(1)
        ci=np.quantile(boot,[.025,.975]).tolist()
    return dict(task_balanced_mean=float(np.mean(means)) if means else None,
                task_bootstrap_95ci=ci,tasks=len(means),observations=sum(map(len,by_task.values())))


def summarize_audit(root, config, checkpoint_report):
    root=Path(root); merged=AuditAccumulator(); cases=[]; action_rows=[]; case_fit=[]
    planned=config.cases()
    for case in planned:
        folder=root/'cases'/case['case_id']
        if not (folder/'result.json').exists():
            continue
        row=json.loads((folder/'result.json').read_text())
        if row['case']!=case:
            raise ValueError('Saved case does not match protocol')
        current=AuditAccumulator.load(folder/'fit_stats.npz')
        merged.merge(current); cases.append(row)
        per_case={}
        for key,b in current.buckets.items():
            mode,layer,_=key.split('|'); per_case.setdefault((mode,layer),FitBucket()).merge(b)
        for (mode,layer),b in per_case.items():
            if mode!='trajectory':
                native=per_case[('trajectory',layer)]
                case_fit.append(dict(task_id=case['task_id'],case_id=case['case_id'],layer=int(layer),mode=mode,
                                     mse_difference=native.sse/native.n-b.sse/b.n))
        for mode,metrics in row['actions'].items():
            action_rows.append(dict(**case,mode=mode,rmse=metrics['prefix']['rmse'],
                                    relative_l2=metrics['prefix']['relative_l2']))
    pooled={}
    for key,b in merged.buckets.items():
        mode,layer,_=key.split('|'); pooled.setdefault((mode,layer),FitBucket()).merge(b)
    fit=[]
    for (mode,layer),b in sorted(pooled.items()):
        native=pooled[('trajectory',layer)]
        counts=merged.occupancy[f'{mode}|{layer}']
        fit.append(dict(mode=mode,layer=int(layer),**b.row(native.m2/native.n),
            dictionary_width=len(counts),never_fired_in_sample=int((counts==0).sum()),
            never_fired_fraction=float((counts==0).mean()),
            occupancy_scope='Observed sample only; never-fired does not establish a dead feature'))
    per_time=[]
    for key,b in sorted(merged.buckets.items()):
        mode,layer,tau_bin=key.split('|'); native=pooled[('trajectory',layer)]
        per_time.append(dict(mode=mode,layer=int(layer),tau_bin=int(tau_bin),
                             tau_low=int(tau_bin)/10,tau_high=(int(tau_bin)+1)/10,**b.row(native.m2/native.n)))
    actions={mode:dict(relative_l2=task_interval([r for r in action_rows if r['mode']==mode],'relative_l2'),
                       rmse=task_interval([r for r in action_rows if r['mode']==mode],'rmse'))
             for mode in sorted({r['mode'] for r in action_rows})}
    contrasts=[]
    for layer in sorted({r['layer'] for r in case_fit}):
        native=pooled[('trajectory',str(layer))]; denominator=native.m2/native.n
        for mode in ('noise_grid','noise_sampled'):
            rows=[dict(r,delta=r['mse_difference']/denominator if denominator>1e-12 else None)
                  for r in case_fit if r['layer']==layer and r['mode']==mode]
            contrasts.append(dict(layer=layer,comparison='trajectory_minus_'+mode,
                                  **task_interval(rows,'delta')))
    report=dict(planned_cases=len(planned),completed_cases=len(cases),
        completed_starts=len({(r['case']['task_id'],r['case']['init_state_id']) for r in cases}),
        completed_tasks=len({r['case']['task_id'] for r in cases}),
        status='complete' if len(cases)==len(planned) else 'partial',fit=fit,per_time=per_time,actions=actions,
        reconstruction_contrasts=contrasts,
        first_step_control_failures=sum(not c['equal_within_repeat_floor'] for row in cases for c in row['first_step_controls']),
        checkpoint_update_status=checkpoint_report['optimizer_update_status'],
        limitations=['Diagnostic cohort uses initial simulator observations, not full rollouts or occlusions.',
            'States are not established as held out from policy/transcoder pretraining.',
            'Current runtime fidelity does not establish fidelity under the original training runtime.',
            'All modes use the same trajectory-derived layer variance denominator for common_variance_nmse.',
            'Relative action errors are in normalized policy action coordinates; actions are not executed.',
            'Few task clusters make bootstrap intervals imprecise; no universal fidelity threshold is assumed.',
            'No feature attribution, steerability, collision prevention or task success is certified.'])
    audit_json(root/'report.json',report)
    lines=['# Transcoder diagnostic evidence','',f"Completed **{len(cases)}/{len(planned)} cases**, {report['completed_starts']} starts, {report['completed_tasks']} tasks.",'',
           '| Layer | Inference nMSE | Fresh-noise/grid nMSE | Random-time nMSE | Inference active features/token |',
           '|---|---:|---:|---:|---:|']
    show=lambda value:'n/a' if value is None else f'{value:.5g}'
    lookup={(r['mode'],r['layer']):r for r in fit}
    for layer in sorted({r['layer'] for r in fit}):
        values=[lookup[(mode,layer)]['common_variance_nmse'] for mode in ('trajectory','noise_grid','noise_sampled')]
        lines.append('| '+str(layer)+' | '+' | '.join(map(show,values))+' | '+show(lookup[('trajectory',layer)]['l0_mean'])+' |')
    lines += ['', 'nMSE uses a common denominator from original inference activations. Lower reconstruction error is better.',
        '', '| Replacement | Prefix relative action error | 95% task-bootstrap interval | Cases |', '|---|---:|---|---:|']
    for mode,metrics in actions.items():
        value=metrics['relative_l2']; interval=value['task_bootstrap_95ci']
        lines.append(f"| {mode} | {show(value['task_balanced_mean'])} | {interval if interval else 'insufficient tasks'} | {value['observations']} |")
    lines += ['','Optimizer evidence: '+checkpoint_report['optimizer_update_status'],
              f"First-step noise-grid controls outside repeat floor: {report['first_step_control_failures']}",
              'Paired reconstruction differences and task-bootstrap intervals: report.json → reconstruction_contrasts.',
              '',*['- '+v for v in report['limitations']]]
    if report['first_step_control_failures']:
        lines += ['', '**First-step pairing control failed: treat the distribution comparison as inconclusive until resolved.**']
    (root/'summary.md').write_text('\n'.join(lines)+'\n')
    return report

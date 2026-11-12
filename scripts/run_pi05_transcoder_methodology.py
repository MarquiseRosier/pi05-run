"""Notebook-thread runner for a controlled layer-level retraining decision."""
from __future__ import annotations

from dataclasses import asdict
import gc
import hashlib
import inspect
import json
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
import torch

from pi05_mi.transcoder_methodology import (
    ARMS, MethodConfig, method_model, atomic_torch_save, records_for, fit_method,
    reconstruction_metrics, method_report,
)
from pi05_mi.transcoder_audit import audit_json, file_sha256, batch_sha256, action_error
from pi05_mi.transcoders import TimeConditionedTranscoder, TimeConditionedTranscoderConfig
from pi05_mi.pi05_weights import assert_weights_loaded
from pi05_mi.patch_pi05 import list_pi05_action_expert_mlp_targets
from run_pi05_transcoder_audit import stage_audit_checkpoint
from probe_pi05_transcoder_counterfactual import build_harness, _add_batch_axis, _infer_prompt


def cpu_tree(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu_tree(v) for k,v in value.items()}
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    return value


def device_tree(value, device):
    if isinstance(value, torch.Tensor): return value.to(device)
    if isinstance(value, dict): return {k: device_tree(v,device) for k,v in value.items()}
    if isinstance(value, list): return [device_tree(v,device) for v in value]
    if isinstance(value, tuple): return tuple(device_tree(v,device) for v in value)
    return value


def snapshot_evidence(policy_path):
    path = Path(policy_path)
    if not path.is_dir():
        raise ValueError('Resolve a local policy snapshot before starting the experiment')
    files = {str(p.relative_to(path)): file_sha256(p) for p in sorted(path.rglob('*'))
             if p.is_file() and (p.suffix in ('.json','.safetensors','.model') or p.name.endswith('.bin'))}
    if not files or not any(n.endswith(('.safetensors','.bin')) for n in files):
        raise ValueError('No teacher weight files in policy snapshot')
    return dict(path=str(path.resolve()), files=files,
        sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest())


class MethodHooks:
    """One layer only. Capture native outputs; substitute only during fidelity evaluation."""
    def __init__(self, policy, layer):
        self.policy = policy
        self.target = next(t for t in list_pi05_action_expert_mlp_targets(policy) if t.layer_index == layer)
        self.original = policy.model.denoise_step
        self.signature = inspect.signature(self.original)
        self.active = False
        self.handle = self.target.module.register_forward_hook(self.hook)
        def denoise(*args, **kwargs):
            bound = self.signature.bind(*args, **kwargs)
            if self.mode in ('A_noise_sampled','B_noise_grid'):
                bound.arguments['x_t'] = self.noise.clone()
            if self.mode == 'A_noise_sampled':
                bound.arguments['timestep'] = self.draw_times[self.call:self.call+1]
            self.t = float(bound.arguments['timestep'].detach().float().reshape(-1)[0])
            self.active = True
            try: return self.original(*bound.args, **bound.kwargs)
            finally:
                self.active = False
                self.call += 1
        policy.model.denoise_step = denoise

    def hook(self, module, inputs, output):
        if not self.active: raise RuntimeError('MLP called outside denoising')
        self.hook_calls += 1
        if self.capture:
            x = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1]).cpu().clone()
            y = output.detach().float().reshape_as(x).cpu().clone()
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                raise ValueError('Nonfinite teacher activation')
            self.records.append(dict(x=x, y=y, t=torch.full((len(x),), self.t)))
        if self.replacement == 'zero': return torch.zeros_like(output)
        if self.replacement == 'mean': return self.mean.to(output).expand_as(output)
        if self.transcoder is not None:
            times = torch.full((inputs[0].shape[0],),self.t,device=inputs[0].device)
            predicted, _ = self.transcoder(inputs[0], times)
            if predicted.shape != output.shape or not torch.isfinite(predicted).all():
                raise ValueError('Invalid replacement output')
            return predicted.to(output.dtype)
        return output

    def predict(self, batch, noise, steps, *, mode='C_trajectory', draw_times=None,
                capture=False, transcoder=None, replacement=None, mean=None):
        before = batch_sha256(batch)
        self.mode, self.noise, self.draw_times = mode, noise, draw_times
        self.capture, self.transcoder, self.replacement, self.mean = capture, transcoder, replacement, mean
        self.records = []; self.call = self.hook_calls = 0
        self.policy.reset()
        try:
            with torch.inference_mode():
                actions = self.policy.predict_action_chunk(batch, noise=noise.clone(),num_steps=steps)
            if self.call != steps or self.hook_calls != steps:
                raise RuntimeError('Incomplete denoising or layer hook coverage')
            if before != batch_sha256(batch): raise RuntimeError('Observation batch mutated')
            records = {k:torch.cat([r[k] for r in self.records]) for k in ('x','y','t')} if capture else None
            if capture and len(records['t']) != steps*noise.shape[1]:
                raise RuntimeError('Activation record count mismatch')
            return actions[0].detach().float().cpu().numpy(), records
        finally:
            self.transcoder = None
            self.records = []

    def close(self):
        self.policy.model.denoise_step = self.original
        self.handle.remove()


def collect_method_case(root, case, config, base, hooks, args, deadline, device):
    folder = root/'cases'/case['case_id']; receipt = folder/'collected.json'
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved['case'] != case: raise ValueError('Case identity changed')
        for name, digest in saved['files'].items():
            if file_sha256(folder/name) != digest: raise ValueError(f'Corrupt case file: {name}')
        return True
    if time.monotonic() >= deadline: return False
    random.seed(case['scene_seed']); np.random.seed(case['scene_seed'])
    args.task_id = case['task_id']; active = build_harness(args, load_policy=False)
    try:
        inner = active.inner_env
        if inner._init_states is None or case['init_state_id'] >= len(inner._init_states):
            raise ValueError('Initial state unavailable; no modulo fallback')
        inner.init_state_id = case['init_state_id']
        observation, _ = inner.reset(seed=case['scene_seed'])
        prompt = _infer_prompt(active, observation)
        if not prompt: raise ValueError('Missing original task instruction')
        obs = base.preprocess_observation(_add_batch_axis(observation)); obs['task'] = [prompt]
        batch = base.preprocessor(base.env_preprocessor(obs))
        generator = torch.Generator(device=device).manual_seed(case['noise_seed'])
        shape = (1,base.policy.model.config.chunk_size,base.policy.model.config.max_action_dim)
        noise = torch.randn(shape, generator=generator,device=device,dtype=torch.float32)
        with torch.random.fork_rng(devices=[torch.device(device).index or 0] if str(device).startswith('cuda') else []):
            torch.manual_seed(case['noise_seed']+1)
            times = base.policy.model.sample_time(config.inference_steps, torch.device(device))
        folder.mkdir(parents=True,exist_ok=True)
        native, native_records = hooks.predict(batch,noise,config.inference_steps,capture=True)
        repeat, _ = hooks.predict(batch,noise,config.inference_steps)
        repeat_error = action_error(native,repeat,config.action_prefix)
        if not repeat_error['exactly_equal']:
            raise RuntimeError('Native repeat differs; resolve nondeterminism before matched training comparisons')
        np.savez(folder/'C_trajectory.npz',**{k:v.numpy() for k,v in native_records.items()})
        files = ['C_trajectory.npz']
        if case['split'] == 'train':
            for mode in ARMS[:2]:
                _, records = hooks.predict(batch,noise,config.inference_steps,mode=mode,draw_times=times,capture=True)
                if mode == ARMS[1] and not torch.equal(records['y'][:shape[1]],native_records['y'][:shape[1]]):
                    raise RuntimeError('First-step noise-grid/trajectory identity failed')
                np.savez(folder/(mode+'.npz'), **{k:v.numpy() for k,v in records.items()})
                files.append(mode+'.npz')
        atomic_torch_save(dict(batch=cpu_tree(batch), noise=noise.cpu(), native=torch.from_numpy(native)),folder/'input.pt')
        files.append('input.pt')
        audit_json(receipt,dict(case=case,prompt=prompt,batch_sha256=batch_sha256(batch),
            files={name:file_sha256(folder/name) for name in files},records_per_mode=len(native_records['t']),
            repeat=repeat_error))
        print('Collected',case['case_id'],case['split'],flush=True)
        return True
    finally:
        active.vec_env.close()


def evaluate_method_case(root, case, config, hooks, models, denominator, mean, deadline, device):
    folder = root/'cases'/case['case_id']; out = root/'evaluation'/case['case_id']
    out.mkdir(parents=True,exist_ok=True)
    receipt = json.loads((folder/'collected.json').read_text())
    if receipt['case'] != case: raise ValueError('Evaluation case identity changed')
    for name in ('input.pt','C_trajectory.npz'):
        if file_sha256(folder/name) != receipt['files'][name]:
            raise ValueError(f'Evaluation input changed: {name}')
    saved = torch.load(folder/'input.pt',map_location='cpu',weights_only=True)
    batch, noise = device_tree(saved['batch'],device), saved['noise'].to(device)
    if batch_sha256(batch) != receipt['batch_sha256']:
        raise ValueError('Saved preprocessed observation hash changed')
    native = saved['native'].numpy()
    with np.load(folder/'C_trajectory.npz') as arrays:
        records = {k:torch.from_numpy(arrays[k].copy()) for k in ('x','y','t')}
    if not (out/'controls.json').exists():
        if time.monotonic() >= deadline: return False
        controls = {}; predictions = dict(native=native)
        for mode in ('repeat','zero','mean'):
            predicted, _ = hooks.predict(batch,noise,config.inference_steps,
                replacement=None if mode=='repeat' else mode,mean=mean)
            controls[mode] = action_error(native,predicted,config.action_prefix)
            predictions[mode] = predicted
        if not controls['repeat']['exactly_equal']:
            raise RuntimeError('Saved observation/noise no longer reproduces original action; teacher/runtime drift')
        np.savez(out/'controls_actions.npz',**predictions)
        audit_json(out/'controls.json',dict(case=case,**controls))
    for name, loader in models:
        if (out/(name+'.json')).exists(): continue
        if time.monotonic() >= deadline: return False
        tc = loader().to(device).eval()
        try:
            fit = reconstruction_metrics(tc,records,denominator)
            predicted, _ = hooks.predict(batch,noise,config.inference_steps,transcoder=tc)
            errors = action_error(native,predicted,config.action_prefix)
            np.savez(out/(name+'_actions.npz'),native=native,replacement=predicted)
            audit_json(out/(name+'.json'),dict(case=case,model=name,**fit,
                action_relative_l2=errors['prefix']['relative_l2'],action_errors=errors))
        finally:
            del tc
            if str(device).startswith('cuda'): torch.cuda.empty_cache()
        print('Evaluated',case['case_id'],name,flush=True)
    return True


def run_methodology(config, checkpoint_path, root, *, source_digest, runtime_report, policy_path,
                    minutes=120, phase='all', device='cuda', cache_dir='/content/pi05-audit-checkpoint'):
    """Budget checked at safe boundaries; setup/loading and an in-flight operation may overrun."""
    if phase not in ('collect','train','evaluate','all') or minutes <= 0:
        raise ValueError('Invalid phase or budget')
    started = time.monotonic(); deadline = started+minutes*60
    cases = config.cases(); root = Path(root); root.mkdir(parents=True,exist_ok=True)
    local, digest = stage_audit_checkpoint(checkpoint_path,cache_dir)
    checkpoint = torch.load(local,map_location='cpu',mmap=True,weights_only=True)
    names = [n for n in checkpoint['configs'] if int(n.split('.')[-2]) == config.layer]
    if len(names) != 1: raise ValueError('Checkpoint layer missing or ambiguous')
    name = names[0]; base_config = checkpoint['configs'][name]
    def old_model():
        tc = TimeConditionedTranscoder(TimeConditionedTranscoderConfig(**base_config))
        tc.load_state_dict(checkpoint['state_dicts'][name],strict=True)
        if not all(torch.isfinite(p).all() for p in tc.parameters()): raise ValueError('Nonfinite old checkpoint')
        return tc
    manifest = dict(config=asdict(config),checkpoint_sha256=digest,source_digest=source_digest,
        runtime=runtime_report,teacher=snapshot_evidence(policy_path),base_config=base_config,
        protocol='methodology-v1; no interventions executed; fixed final update, no test-based selection')
    manifest = json.loads(json.dumps(manifest))
    path = root/'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Protocol/teacher/runtime changed. Use a new RUN_NAME; old evidence is preserved.')
    audit_json(path,manifest)
    base = hooks = None
    args = SimpleNamespace(suite='libero_spatial',task_id=config.train_tasks[0],prompt='',
                           policy_path=str(policy_path),device=device,policy_dtype='bfloat16')
    def load_teacher():
        nonlocal base, hooks
        if base is None:
            base = build_harness(args)
            audit_json(root/'teacher_load.json',assert_weights_loaded(base.policy,source=str(policy_path)))
            base.vec_env.close()
            hooks = MethodHooks(base.policy,config.layer)
    try:
        collected = all((root/'cases'/c['case_id']/'collected.json').exists() for c in cases)
        if phase in ('collect','all'):
            if not collected: load_teacher()
            for case in cases:
                if not collect_method_case(root,case,config,base,hooks,args,deadline,device):
                    return method_report(root,config)
            collected = True
        if phase == 'collect': return method_report(root,config)
        if not collected: raise RuntimeError('Complete collection before training or evaluation')
        training = records_for(root,cases,'train','C_trajectory')
        # Frozen train-only denominator shared by A/B/C and all reported evaluation metrics.
        denominator = float(training['y'].double().var(0,unbiased=False).sum())
        if not np.isfinite(denominator) or denominator <= 1e-12: raise ValueError('Degenerate training variance')
        mean = training['y'].mean(0)
        audit_json(root/'normalizer.json',dict(variance=denominator,records=len(training['t']),source='train C_trajectory'))
        validation = records_for(root,cases,'validation','C_trajectory')
        if phase in ('train','all'):
            # Free the teacher before fitting. Reload from the identical hashed snapshot for evaluation.
            if hooks is not None: hooks.close(); hooks = None
            base = None; gc.collect()
            if str(device).startswith('cuda'): torch.cuda.empty_cache()
            for arm in ARMS:
                data = training if arm in ARMS[2:] else records_for(root,cases,'train',arm)
                if len(data['t']) != len(training['t']): raise ValueError('Unequal training record budgets')
                for seed in config.fit_seeds:
                    if not fit_method(root,config,base_config,arm,seed,data,validation,denominator,deadline,device):
                        return method_report(root,config)
                if arm not in ARMS[2:]: del data
        if phase == 'train': return method_report(root,config)
        models = [('existing',old_model)]
        for seed in config.fit_seeds:
            for arm in ARMS:
                folder = root/'fits'/f'{arm}_seed{seed}'
                if not (folder/'complete.json').exists(): raise RuntimeError('Finish all matched fits before test evaluation')
                receipt = json.loads((folder/'complete.json').read_text())
                if file_sha256(folder/'model.pt') != receipt['sha256']: raise ValueError('Model checksum changed')
                def loader(path=folder/'model.pt',arm=arm,seed=seed):
                    saved = torch.load(path,map_location='cpu',weights_only=True)
                    tc = method_model(base_config,arm,seed)
                    tc.load_state_dict(saved['state_dict'],strict=True)
                    return tc
                models.append((f'{arm}_seed{seed}',loader))
        del training, validation
        if all((root/'evaluation'/c['case_id']/(n+'.json')).exists()
               for c in cases if c['split']=='test' for n,_ in models):
            return method_report(root,config)
        if time.monotonic() >= deadline: return method_report(root,config)
        load_teacher()
        # Keep one dictionary resident across cases; avoid re-reading multi-GB weights from Drive per scene.
        for model_name, loader in models:
            if time.monotonic() >= deadline: break
            tc = loader().to(device).eval()
            try:
                for case in cases:
                    if case['split']=='test' and not evaluate_method_case(root,case,config,hooks,
                            [(model_name,lambda:tc)],denominator,mean,deadline,device):
                        break
            finally:
                del tc
                if str(device).startswith('cuda'): torch.cuda.empty_cache()
        return method_report(root,config)
    except BaseException as exc:
        audit_json(root/'failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
    finally:
        if hooks is not None: hooks.close()
        if base is not None: base.vec_env.close()
        audit_json(root/'last_session.json',dict(elapsed_seconds=time.monotonic()-started,requested_minutes=minutes))

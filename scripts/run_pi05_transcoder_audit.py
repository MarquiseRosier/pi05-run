"""Main-thread checkpoint audit. Embedded literally in the diagnostic notebook."""
from __future__ import annotations
import inspect
import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

from pi05_mi.transcoder_audit import (
    AuditConfig, AuditAccumulator, action_error, audit_json, checkpoint_evidence,
    file_sha256, summarize_audit, tensor_sha256, batch_sha256,
)
from pi05_mi.transcoders import TimeConditionedTranscoder, TimeConditionedTranscoderConfig
from pi05_mi.patch_pi05 import list_pi05_action_expert_mlp_targets
from pi05_mi.pi05_weights import assert_weights_loaded


def stage_audit_checkpoint(source, cache_dir):
    """Copy once to local disk for mmap; validate the cached file on every run."""
    source, cache_dir=Path(source),Path(cache_dir)
    if not source.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {source}')
    cache_dir.mkdir(parents=True,exist_ok=True)
    signature=dict(path=str(source.resolve()),size=source.stat().st_size,mtime_ns=source.stat().st_mtime_ns)
    receipt=cache_dir/'receipt.json'; local=cache_dir/'checkpoint.pt'
    old=json.loads(receipt.read_text()) if receipt.exists() else None
    if old and old['source']==signature and local.is_file() and local.stat().st_size==signature['size']:
        digest=file_sha256(local)
        if digest==old['sha256']:
            print('Reusing verified local checkpoint:',local,flush=True)
            return local,digest
    import hashlib
    digest=hashlib.sha256(); total=0; temporary=local.with_suffix('.partial')
    with source.open('rb') as reader,temporary.open('wb') as writer:
        for block in iter(lambda:reader.read(16*1024*1024),b''):
            writer.write(block); digest.update(block); total+=len(block)
            if total%(256*1024*1024)==0:
                print(f'Checkpoint copied: {total/2**30:.2f}/{signature["size"]/2**30:.2f} GiB',flush=True)
    if total!=signature['size'] or source.stat().st_mtime_ns!=signature['mtime_ns']:
        raise RuntimeError('Checkpoint changed during copy; rerun after the writer finishes')
    temporary.replace(local)
    digest=digest.hexdigest(); audit_json(receipt,dict(source=signature,sha256=digest))
    return local,digest


class AuditHooks:
    """Observe native MLPs and optionally replace them; retain exact time/context."""
    def __init__(self, policy, transcoders):
        self.policy,self.transcoders=policy,transcoders
        self.targets=list_pi05_action_expert_mlp_targets(policy)
        if set(transcoders)!={t.name for t in self.targets}:
            raise ValueError('Policy MLP names and checkpoint layers must match exactly')
        self.original=policy.model.denoise_step
        self.signature=inspect.signature(self.original)
        self.handles=[]; self.mode=None; self.acc=None; self.index=0; self.replacements=set()
        self.active=False; self.replace_calls=0; self.first_y={}; self.times=[]
        for target in self.targets:
            self.handles.append(target.module.register_forward_hook(self._hook(target)))
        def denoise(*args,**kwargs):
            bound=self.signature.bind(*args,**kwargs)
            if self.mode=='noise_grid':
                bound.arguments['x_t']=self.noise.clone()
            elif self.mode=='noise_sampled':
                bound.arguments['x_t']=self.draw_noise[self.index].clone()
                bound.arguments['timestep']=self.draw_times[self.index:self.index+1]
            self.t=float(bound.arguments['timestep'].detach().float().reshape(-1)[0])
            self.times.append(self.t); self.active=True
            try:
                return self.original(*bound.args,**bound.kwargs)
            finally:
                self.active=False; self.index+=1
        policy.model.denoise_step=denoise

    def _hook(self,target):
        def hook(module,inputs,output):
            if not self.active:
                raise RuntimeError('Action MLP called outside denoise_step; unsupported policy path')
            if self.index==0:
                self.first_y[target.layer_index]=output.detach().float().cpu().clone()
            measure=self.mode in ('trajectory','noise_grid','noise_sampled')
            replace=target.layer_index in self.replacements
            if not measure and not replace:
                return output
            tc=self.transcoders[target.name]
            t=torch.full((inputs[0].shape[0],),self.t,device=inputs[0].device,dtype=torch.float32)
            predicted,z=tc(inputs[0],t)
            if predicted.shape!=output.shape:
                raise ValueError('Transcoder output shape does not match native MLP')
            if measure:
                self.acc.update(self.mode,target.layer_index,inputs[0],output,predicted,z,self.t,self.index)
            if replace:
                self.replace_calls+=1
                if not bool(torch.isfinite(predicted).all()):
                    raise ValueError('Non-finite replacement output')
                return predicted.to(dtype=output.dtype)
            return output
        return hook

    def predict(self,batch,noise,steps,*,mode,acc,replacements=(),draw_times=None,draw_noise=None):
        before=batch_sha256(batch)
        self.mode,self.acc,self.noise=mode,acc,noise
        self.replacements=set(replacements); self.index=0; self.replace_calls=0
        self.first_y={}; self.times=[]; self.draw_times,self.draw_noise=draw_times,draw_noise
        self.policy.reset()
        with torch.inference_mode():
            action=self.policy.predict_action_chunk(batch,noise=noise.clone(),num_steps=steps)
        if self.index!=steps or self.replace_calls!=steps*len(self.replacements):
            raise RuntimeError('Denoising/replacement call count mismatch; refusing incomplete evidence')
        if batch_sha256(batch)!=before:
            raise RuntimeError('Policy mutated the shared observation batch; refusing unmatched comparisons')
        return action[0].detach().float().cpu().numpy()

    def close(self):
        self.policy.model.denoise_step=self.original
        for handle in self.handles:
            handle.remove()


def run_transcoder_audit(config,checkpoint_path,root,*,source_digest,runtime_report,
                         policy_path='lerobot/pi05_libero_finetuned',minutes=30,phase='all',
                         cache_dir='/content/pi05-audit-checkpoint',device='cuda'):
    """Save each complete case, resume exact protocols, and preserve native errors."""
    if phase not in ('metadata','all') or minutes<=0:
        raise ValueError('Use phase metadata/all and a positive collection budget')
    planned=config.cases(); root=Path(root); root.mkdir(parents=True,exist_ok=True)
    local,digest=stage_audit_checkpoint(checkpoint_path,cache_dir)
    checkpoint=torch.load(local,map_location='cpu',mmap=True,weights_only=True)
    old_manifest=root/'manifest.json'
    if old_manifest.exists():
        old=json.loads(old_manifest.read_text())
        if old['checkpoint_sha256']!=digest or old['source_digest']!=source_digest:
            raise ValueError('Checkpoint or notebook definitions changed; choose a new RUN_NAME')
    evidence=checkpoint_evidence(checkpoint)
    evidence.update(checkpoint_sha256=digest,source_path=str(checkpoint_path),local_path=str(local),
                    source_digest=source_digest)
    audit_json(root/'checkpoint_audit.json',evidence)
    print(json.dumps({k:evidence[k] for k in ('collection_mode','collection_feed_forwards',
        'optimizer_update_status','optimizer_step_min','optimizer_step_max','all_dictionary_weights_finite')},indent=2),flush=True)
    if phase=='metadata':
        return evidence
    if not evidence['all_dictionary_weights_finite']:
        raise ValueError('Checkpoint contains non-finite dictionary weights; see checkpoint_audit.json')
    layer_ids={r['layer'] for r in evidence['layers']}
    if not set(config.replacement_layers)<=layer_ids:
        raise ValueError('Requested replacement layer is absent from checkpoint')
    manifest=dict(config=asdict(config),source_digest=source_digest,checkpoint_sha256=digest,
                  runtime=runtime_report,policy_path=str(policy_path),
                  scope='Initial simulator observations; action predictions only; no executed interventions')
    manifest=json.loads(json.dumps(manifest))
    manifest_path=root/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:
        raise ValueError('Diagnostic protocol/runtime/checkpoint changed. Choose a new RUN_NAME; saved cases are preserved.')
    audit_json(manifest_path,manifest)
    if all((root/'cases'/c['case_id']/'result.json').exists() for c in planned):
        return summarize_audit(root,config,evidence)
    from probe_pi05_transcoder_counterfactual import build_harness, _add_batch_axis, _infer_prompt
    base=None; active=None; hooks=None
    args=SimpleNamespace(suite='libero_spatial',task_id=config.tasks[0],prompt='',
                         policy_path=str(policy_path),device=device,policy_dtype='bfloat16')
    try:
        base=build_harness(args); policy=base.policy
        teacher_report=assert_weights_loaded(policy,source=str(policy_path))
        audit_json(root/'teacher_load.json',teacher_report)
        base.vec_env.close()
        transcoders={}
        for name,cfg in checkpoint['configs'].items():
            tc=TimeConditionedTranscoder(TimeConditionedTranscoderConfig(**cfg))
            tc.load_state_dict(checkpoint['state_dicts'][name],strict=True)
            tc.to(device=device,dtype=torch.float32).eval()
            for parameter in tc.parameters(): parameter.requires_grad_(False)
            transcoders[name]=tc
        del checkpoint
        hooks=AuditHooks(policy,transcoders)
        deadline=time.monotonic()+minutes*60
        for case in planned:
            folder=root/'cases'/case['case_id']
            if (folder/'result.json').exists():
                print('Reusing',case['case_id'],flush=True); continue
            if time.monotonic()>=deadline:
                print('Collection budget reached between cases; rerun to continue.',flush=True); break
            print('Auditing',case['case_id'],flush=True)
            random.seed(case['scene_seed']); np.random.seed(case['scene_seed'])
            args.task_id=case['task_id']; active=build_harness(args,load_policy=False)
            inner=active.inner_env
            if inner._init_states is None or case['init_state_id']>=len(inner._init_states):
                raise ValueError('Requested initial state unavailable; no modulo wrapping')
            inner.init_state_id=case['init_state_id']
            observation,_=inner.reset(seed=case['scene_seed'])
            prompt=_infer_prompt(active,observation)
            if not prompt: raise ValueError('No task instruction found')
            obs=base.preprocess_observation(_add_batch_axis(observation)); obs['task']=[prompt]
            batch=base.preprocessor(base.env_preprocessor(obs))
            generator=torch.Generator(device=device).manual_seed(case['noise_seed'])
            shape=(1,policy.model.config.chunk_size,policy.model.config.max_action_dim)
            noise=torch.randn(shape,generator=generator,device=device,dtype=torch.float32)
            draw_noise=torch.randn((config.inference_steps,*shape),generator=generator,device=device,dtype=torch.float32)
            # Model.sample_time does not accept a generator; isolate its global RNG use.
            with torch.random.fork_rng(devices=[torch.device(device).index or 0] if str(device).startswith('cuda') else []):
                torch.manual_seed(case['noise_seed']+1)
                draw_times=policy.model.sample_time(config.inference_steps,torch.device(device))
            acc=AuditAccumulator(); actions={}; arrays={}; first={}; time_draws={}
            for mode in ('trajectory','baseline_repeat','noise_grid','noise_sampled'):
                arrays[mode]=hooks.predict(batch,noise,config.inference_steps,mode=mode,acc=acc,
                                          draw_times=draw_times,draw_noise=draw_noise)
                first[mode]=dict(hooks.first_y); time_draws[mode]=list(hooks.times)
            actions['baseline_repeat']=action_error(arrays['trajectory'],arrays['baseline_repeat'],config.action_prefix)
            controls=[]
            for layer in sorted(layer_ids):
                native=first['trajectory'][layer]
                repeat=float((native-first['baseline_repeat'][layer]).abs().max())
                grid=float((native-first['noise_grid'][layer]).abs().max())
                controls.append(dict(layer=layer,repeat_max_abs=repeat,noise_grid_first_max_abs=grid,
                                     equal_within_repeat_floor=grid<=max(1e-6,repeat*5)))
            modes={f'layer_{layer}':[layer] for layer in config.replacement_layers}
            if config.replace_all: modes['all_layers']=sorted(layer_ids)
            for mode,layers in modes.items():
                arrays[mode]=hooks.predict(batch,noise,config.inference_steps,mode='replacement',acc=acc,replacements=layers)
                actions[mode]=action_error(arrays['trajectory'],arrays[mode],config.action_prefix)
            for layer in layer_ids:
                for mode in ('trajectory','noise_grid','noise_sampled'):
                    buckets=[b for k,b in acc.buckets.items() if k.startswith(f'{mode}|{layer}|')]
                    if sum(b.calls for b in buckets)!=config.inference_steps or sum(b.n for b in buckets)!=shape[1]*config.inference_steps:
                        raise RuntimeError('Missing activation records; refusing truncated statistics')
            folder.mkdir(parents=True,exist_ok=True)
            acc.save(folder/'fit_stats.npz')
            np.savez_compressed(folder/'actions.npz',**arrays)
            np.savez_compressed(folder/'initial_images.npz',**observation['pixels'])
            image_hashes={k:hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for k,v in observation['pixels'].items()}
            # Readback verification precedes the result.json completion marker.
            loaded=AuditAccumulator.load(folder/'fit_stats.npz')
            if sum(b.n for b in loaded.buckets.values())!=sum(b.n for b in acc.buckets.values()):
                raise RuntimeError('Saved activation counts differ')
            audit_json(folder/'result.json',dict(case=case,prompt=prompt,initial_image_sha256=image_hashes,
                noise_sha256=tensor_sha256(noise),batch_sha256=batch_sha256(batch),time_draws=time_draws,actions=actions,
                first_step_controls=controls,grid_initial_time=time_draws['noise_grid'][0],
                native_initial_time=time_draws['trajectory'][0]))
            active.vec_env.close(); active=None
            completed=sum((root/'cases'/c['case_id']/'result.json').exists() for c in planned)
            audit_json(root/'state.json',dict(status='collecting',completed=completed,planned=len(planned)))
            print(f'Saved {completed}/{len(planned)} cases',flush=True)
        report=summarize_audit(root,config,evidence)
        audit_json(root/'state.json',dict(status=report['status'],completed=report['completed_cases'],planned=len(planned)))
        if (root/'failure.json').exists(): (root/'failure.json').unlink()
        return report
    except BaseException as exc:
        audit_json(root/'failure.json',dict(error_type=type(exc).__name__,message=str(exc)))
        raise
    finally:
        if hooks is not None: hooks.close()
        if active is not None:
            try: active.vec_env.close()
            except Exception: pass
        if base is not None and hooks is None:
            try: base.vec_env.close()
            except Exception: pass

"""Full-dataset observation collection, frozen-teacher training, and fidelity evaluation."""
from __future__ import annotations

from dataclasses import asdict
import gc
import hashlib
import inspect
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from pi05_mi.trajectory_training import (
    TrajectoryTrainConfig, training_plan, new_trajectory_model, TrainingDashboard,
    publish_training_checkpoint, restore_training_checkpoint, optimize_trajectory_layer,
    validation_with_occupancy,
)
from pi05_mi.transcoder_methodology import atomic_torch_save, reconstruction_metrics, PaperAdaptedTranscoder
from pi05_mi.transcoders import TimeConditionedTranscoderConfig
from pi05_mi.transcoder_audit import audit_json, file_sha256, action_error
from pi05_mi.pi05_weights import assert_weights_loaded
from pi05_mi.patch_pi05 import list_pi05_action_expert_mlp_targets
from run_pi05_transcoder_methodology import snapshot_evidence
from probe_pi05_transcoder_counterfactual import build_harness


class DatasetTrajectorySource:
    """Uses whole recorded episodes as observations; policy actions are predicted, never executed."""
    def __init__(self,dataset,policy_path,config,device='cuda'):
        self.dataset,self.config,self.device=dataset,config,device
        args=SimpleNamespace(suite='libero_spatial',task_id=0,prompt='',policy_path=str(policy_path),
                             device=device,policy_dtype='bfloat16')
        self.base=build_harness(args);self.policy=self.base.policy;self.base.vec_env.close()
        self.teacher_load=assert_weights_loaded(self.policy,source=str(policy_path))
        if self.policy.training or any(p.requires_grad for p in self.policy.parameters()):
            raise ValueError('Teacher must be in evaluation mode with all weights frozen')
        targets=list_pi05_action_expert_mlp_targets(self.policy)
        self.targets={t.layer_index:t for t in targets if t.layer_index in config.layers}
        if set(self.targets)!=set(config.layers): raise ValueError('Requested policy layers missing')
        self.widths={l:t.d_model for l,t in self.targets.items()}
        self.names={l:t.name for l,t in self.targets.items()}
        self.chunk_size=int(self.policy.model.config.chunk_size)
        self.handles=[];self.capture=None;self.offsets={};self.replacements={};self.active=False
        self.original=self.policy.model.denoise_step;signature=inspect.signature(self.original)
        def denoise(*args,**kwargs):
            bound=signature.bind(*args,**kwargs)
            self.t=float(bound.arguments['timestep'].float().reshape(-1)[0]);self.active=True
            try: return self.original(*bound.args,**bound.kwargs)
            finally: self.active=False;self.calls+=1
        self.policy.model.denoise_step=denoise
        for layer,target in self.targets.items(): self.handles.append(target.module.register_forward_hook(self.hook(layer)))

    def hook(self,layer):
        def observe(module,inputs,output):
            if not self.active: raise RuntimeError('MLP called outside denoising')
            if self.capture is not None:
                x=inputs[0].detach().reshape(-1,inputs[0].shape[-1]);y=output.detach().reshape_as(x)
                if not torch.isfinite(x).all() or not torch.isfinite(y).all(): raise ValueError('Nonfinite teacher activation')
                offset=self.offsets[layer];end=offset+len(x)
                if end>len(self.capture[layer]['t']): raise ValueError('Activation buffer exceeded expected call count')
                self.capture[layer]['x'][offset:end].copy_(x)
                self.capture[layer]['y'][offset:end].copy_(y)
                self.capture[layer]['t'][offset:end]=self.t;self.offsets[layer]=end
            if layer in self.replacements:
                t=torch.full((inputs[0].shape[0],),self.t,device=inputs[0].device)
                y,_=self.replacements[layer](inputs[0],t)
                if not torch.isfinite(y).all(): raise ValueError('Nonfinite replacement prediction')
                return y.to(output.dtype)
            return output
        return observe

    def batch(self,index):
        from torch.utils.data import default_collate
        row=self.dataset[int(index)]
        if int(row['index'])!=int(index): raise ValueError('Dataset row indices do not match frozen frame plan')
        obs={k:v for k,v in row.items() if k.startswith('observation.')}
        task=row.get('task')
        if not isinstance(task,str) or not task.strip(): raise ValueError('Dataset frame has no task instruction')
        for key in self.dataset.meta.camera_keys:
            if key not in obs: raise ValueError(f'Dataset camera absent: {key}')
            if obs[key].dtype==torch.uint8: obs[key]=obs[key].float()/255.
        obs['task']=task
        return self.base.preprocessor(default_collate([obs]))

    def predict(self,index,seed,replacements=None):
        batch=self.batch(index);self.policy.reset();self.calls=0;self.replacements=replacements or {}
        g=torch.Generator(device=self.device).manual_seed(int(seed))
        noise=torch.randn((1,self.chunk_size,self.policy.model.config.max_action_dim),generator=g,device=self.device,dtype=torch.float32)
        try:
            with torch.inference_mode(): actions=self.policy.predict_action_chunk(batch,noise=noise,num_steps=self.config.inference_steps)
            if self.calls!=self.config.inference_steps: raise ValueError('Wrong denoising call count')
            return actions[0].float().cpu().numpy()
        finally: self.replacements={}

    def collect(self,indices,seed,dashboard,deadline):
        n=len(indices)*self.chunk_size*self.config.inference_steps
        self.capture={l:dict(x=torch.empty(n,d,dtype=torch.float32,device='cpu'),
            y=torch.empty(n,d,dtype=torch.float32,device='cpu'),t=torch.empty(n,dtype=torch.float32,device='cpu')) for l,d in self.widths.items()}
        self.offsets={l:0 for l in self.targets}
        started=time.monotonic()
        try:
            for i,index in enumerate(indices):
                if time.monotonic()>=deadline: raise CollectionBudgetExpired()
                self.predict(index,seed+i)
                dashboard.emit('collecting native trajectories',frames_in_buffer=f'{i+1}/{len(indices)}',
                    layer_count=len(self.targets),collection_frames_per_second=round((i+1)/max(time.monotonic()-started,1e-6),3))
            if any(v!=n for v in self.offsets.values()): raise ValueError('Missing layer activation records')
            result=self.capture;self.capture=None
            return result
        finally: self.capture=None

    def action_metrics(self,indices,seed,replacements):
        rows=[]
        for i,index in enumerate(indices):
            baseline=self.predict(index,seed+i);repeat=self.predict(index,seed+i)
            if not np.array_equal(baseline,repeat): raise ValueError('Native repeat drift: fidelity comparison refused')
            modified=self.predict(index,seed+i,replacements)
            metrics=action_error(baseline,modified,10)
            rows.append(dict(frame=int(index),seed=seed+i,**metrics))
        values=[r['prefix']['relative_l2'] for r in rows]
        return dict(relative_l2=float(np.mean(values)) if values and all(v is not None for v in values) else None,
                    cases=len(rows),repeat_floor=0.,rows=rows)

    def close(self):
        self.policy.model.denoise_step=self.original
        for handle in self.handles: handle.remove()


class CollectionBudgetExpired(Exception): pass


def dataset_episodes(dataset):
    table=dataset.meta.episodes
    if hasattr(table,'to_pandas'): rows=table.to_pandas().to_dict('records')
    elif hasattr(table,'to_dict'): rows=table.to_dict('records')
    else: rows=list(table)
    return [{k:int(row[k]) for k in ('episode_index','dataset_from_index','dataset_to_index')} for row in rows]


def fingerprint_dataset(dataset, callback=print):
    root=Path(dataset.root);files={}
    selected=sorted(p for p in root.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.parquet','.mp4'))
    if not selected: raise ValueError('Dataset has no hashable metadata/data/video files')
    total=sum(p.stat().st_size for p in selected);done=0;last=0.
    for path in selected:
        files[str(path.relative_to(root))]=file_sha256(path);done+=path.stat().st_size
        if time.monotonic()-last>5:
            callback(f'Dataset identity: hashed {done/2**30:.2f}/{total/2**30:.2f} GiB');last=time.monotonic()
    return dict(repo_id=dataset.repo_id,files=files,bytes=total,
                sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest())


def layer_progress(state_dir,layer):
    path=Path(state_dir)/f'layer_{layer}.pt'
    if not path.exists(): return None
    return torch.load(path,map_location='cpu',mmap=True,weights_only=True)


def prune_uncommitted_metrics(dashboard,state_dir):
    steps={l:(layer_progress(state_dir,l) or {}).get('step',0) for l in dashboard.config.layers}
    unique={}
    for r in dashboard.rows:
        if r.get('step',0)<=steps.get(r.get('layer'),0): unique[(r.get('event'),r.get('layer'),r.get('step'))]=r
    dashboard.rows=list(unique.values())
    dashboard.path.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in dashboard.rows))


def run_trajectory_training(config,source,episodes,root,local,provenance,*,minutes=120,
                            checkpoint_minutes=30,charts=True,device='cuda'):
    """Train one dictionary at a time on shared fresh buffers; snapshots cover all completed updates."""
    config.validate()
    if minutes<=0 or checkpoint_minutes<=0: raise ValueError('Budgets must be positive')
    started=time.monotonic();deadline=started+minutes*60
    root,local=Path(root),Path(local);root.mkdir(parents=True,exist_ok=True);local.mkdir(parents=True,exist_ok=True)
    plan=training_plan(episodes,config)
    protocol=json.loads(json.dumps(dict(config=asdict(config),provenance=provenance,plan=plan,
        architecture='diffract_adapted_time_v1',collection='native inference trajectories on dataset observations')))
    protocol_hash=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    manifest=root/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=protocol:
        raise ValueError('Run protocol/data/teacher changed. Use a new RUN_NAME; existing checkpoints are preserved.')
    audit_json(manifest,protocol)
    finished=root/'training_summary.json'
    if finished.exists() and (root/'trained_transcoders.pt').exists():
        summary=json.loads(finished.read_text())
        if summary.get('status')=='complete' and summary.get('export_sha256')==file_sha256(root/'trained_transcoders.pt'):
            print('This training run and final test are already complete.',flush=True)
            return summary
    if (root/'failure.json').exists():
        history=root/'previous_failures';history.mkdir(exist_ok=True)
        (root/'failure.json').replace(history/f'{time.time_ns()}.json')
    state_dir=local/protocol_hash[:16]/'state';cache=local/protocol_hash[:16]/'cache';cache.mkdir(parents=True,exist_ok=True)
    restore_training_checkpoint(root,state_dir,protocol_hash)
    dashboard=TrainingDashboard(root,state_dir,config,enabled=charts)
    prune_uncommitted_metrics(dashboard,state_dir)
    per_cycle=math.ceil(config.frames_per_cycle*source.chunk_size*config.inference_steps/config.batch_size)
    total_steps=config.cycles*per_cycle
    dashboard.emit('preflight passed',force=True,layers=list(config.layers),updates_per_layer=total_steps,
        samples_per_layer=total_steps*config.batch_size,planned_unique_train_frames=len(set(plan['train_frames'])),
        activation_buffer_GiB=round(config.frames_per_cycle*source.chunk_size*config.inference_steps*
                                   sum(source.widths.values())*8/2**30,2),scope='Fidelity only; attribution untested')
    last_checkpoint=time.monotonic();saved=False
    def checkpoint():
        nonlocal last_checkpoint,saved
        prune_uncommitted_metrics(dashboard,state_dir)
        publish_training_checkpoint(state_dir,root,protocol_hash,dashboard)
        last_checkpoint=time.monotonic();saved=True
    def cached_records(tag,indices,seed):
        path=cache/(tag+'.pt');receipt=cache/(tag+'.json')
        if path.exists() and receipt.exists() and file_sha256(path)==json.loads(receipt.read_text())['sha256']:
            return torch.load(path,map_location='cpu',mmap=True,weights_only=True)
        dashboard.emit('collecting '+tag,force=True,observations=len(indices))
        records=source.collect(indices,seed,dashboard,deadline)
        atomic_torch_save(records,path);audit_json(receipt,dict(sha256=file_sha256(path)))
        return records
    try:
        normalizer_path=state_dir/'normalizers.json'
        if normalizer_path.exists():
            normalizers={int(k):float(v) for k,v in json.loads(normalizer_path.read_text()).items()}
            if set(normalizers)!=set(config.layers): raise ValueError('Saved reference layers changed')
        else:
            reference=cached_records('train_reference',plan['reference_frames'],config.seed+700)
            normalizers={l:float(reference[l]['y'].double().var(0,unbiased=False).sum()) for l in config.layers}
            del reference
        if any(not np.isfinite(v) or v<=1e-12 for v in normalizers.values()): raise ValueError('Degenerate reference variance')
        if not normalizer_path.exists():
            audit_json(normalizer_path,{str(k):v for k,v in normalizers.items()})
        validation=cached_records('validation',plan['validation_frames'],config.seed+800)
        for cycle in range(config.cycles):
            validate_cycle=(cycle+1)%config.validate_every==0 or cycle==0 or cycle+1==config.cycles
            def needs_work(layer):
                trained=(layer_progress(state_dir,layer) or {}).get('step',0)
                marker=state_dir/f'validated_{layer}.json'
                validated=json.loads(marker.read_text())['step'] if marker.exists() else -1
                return trained<(cycle+1)*per_cycle or (validate_cycle and validated<(cycle+1)*per_cycle)
            needed=[l for l in config.layers if needs_work(l)]
            if not needed: continue
            if time.monotonic()>=deadline: checkpoint();return dict(status='paused_budget',root=str(root))
            indices=plan['train_frames'][cycle*config.frames_per_cycle:(cycle+1)*config.frames_per_cycle]
            cycle_started=time.monotonic()
            dashboard.emit('starting cycle',force=True,cycle=f'{cycle+1}/{config.cycles}',
                           unique_train_frames_seen=len(set(plan['train_frames'][:cycle*config.frames_per_cycle])))
            records=cached_records(f'cycle_{cycle}',indices,config.seed+10000+cycle*config.frames_per_cycle)
            for layer in needed:
                if time.monotonic()>=deadline: checkpoint();return dict(status='paused_budget',root=str(root))
                model=new_trajectory_model(source.widths[layer],config,layer).to(device)
                optimizer=torch.optim.AdamW(model.parameters(),lr=config.lr,weight_decay=0.)
                previous=layer_progress(state_dir,layer);cursor=0
                if previous:
                    model.load_state_dict(previous['model'],strict=True);optimizer.load_state_dict(previous['optimizer'])
                    cursor=previous['step']-cycle*per_cycle
                    if not 0<=cursor<=per_cycle: raise ValueError('Layer checkpoint has inconsistent cycle position')
                del previous
                dashboard.emit('training',force=True,layer=layer,cycle=f'{cycle+1}/{config.cycles}',
                               dictionary_features=source.widths[layer]*config.expansion)
                # Only a successfully returned update block is committed. Interrupts discard its uncommitted updates.
                try:
                    if cycle==0 and cursor==0:
                        metrics=validation_with_occupancy(model,validation[layer],normalizers[layer])
                        dashboard.record(dict(event='validation',layer=layer,step=0,cycle=0,**metrics))
                    cursor=optimize_trajectory_layer(model,optimizer,records[layer],normalizers[layer],config,
                        layer,cycle,cursor,per_cycle,dashboard,deadline,device)
                    step=cycle*per_cycle+cursor
                    atomic_torch_save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,
                        config=asdict(model.config),protocol_hash=protocol_hash),state_dir/f'layer_{layer}.pt')
                    if cursor==per_cycle and validate_cycle:
                        metrics=validation_with_occupancy(model,validation[layer],normalizers[layer])
                        dashboard.record(dict(event='validation',layer=layer,step=step,cycle=cycle+1,**metrics))
                        dashboard.emit('validation',validation_nmse=round(metrics['nmse'],5),
                            validation_active_features=round(metrics['mean_l0'],1),
                            never_fired_in_validation_percent=round(100*metrics['observed_never_fired_fraction'],2))
                        best_path=state_dir/f'best_{layer}.pt'
                        best=torch.load(best_path,map_location='cpu',mmap=True,weights_only=True) if best_path.exists() else None
                        if best is None or metrics['nmse']<best['validation_nmse']:
                            atomic_torch_save(dict(model=model.state_dict(),config=asdict(model.config),step=step,
                                validation_nmse=metrics['nmse'],validation_l0=metrics['mean_l0']),best_path)
                        del best
                        if config.action_validation_frames:
                            if time.monotonic()>=deadline: raise CollectionBudgetExpired()
                            action=source.action_metrics(plan['validation_frames'][:config.action_validation_frames],
                                config.seed+800,{layer:model})
                            if action['relative_l2'] is not None:
                                dashboard.record(dict(event='validation_action',layer=layer,step=step,cycle=cycle+1,**action))
                        audit_json(state_dir/f'validated_{layer}.json',dict(step=step))
                finally:
                    del model,optimizer;gc.collect()
                    if str(device).startswith('cuda'): torch.cuda.empty_cache()
                if time.monotonic()-last_checkpoint>=checkpoint_minutes*60:
                    checkpoint()
            del records
            # A completed cycle can be regenerated from the frozen frame/noise plan if needed.
            (cache/f'cycle_{cycle}.pt').unlink(missing_ok=True);(cache/f'cycle_{cycle}.json').unlink(missing_ok=True)
            cycle_seconds=time.monotonic()-cycle_started
            dashboard.emit('cycle complete',force=True,cycle=f'{cycle+1}/{config.cycles}',
                observed_cycle_seconds=round(cycle_seconds,1),
                rough_remaining_hours=round(cycle_seconds*(config.cycles-cycle-1)/3600,2),
                unique_train_frames_seen=len(set(plan['train_frames'][:(cycle+1)*config.frames_per_cycle])))
        checkpoint()
        bundle=dict(format='pi05-trajectory-transcoders-v1',complete=True,configs={},state_dicts={},architectures={},
                    manifest=protocol,protocol_hash=protocol_hash,updates_per_layer=total_steps,
                    selection='best validation nMSE per layer',selected_steps={})
        for layer in config.layers:
            saved_layer=layer_progress(state_dir,layer)
            if saved_layer['step']!=total_steps: raise ValueError('Refusing to export an incomplete layer set')
            saved_layer=torch.load(state_dir/f'best_{layer}.pt',map_location='cpu',mmap=True,weights_only=True)
            name=source.names[layer];bundle['configs'][name]=saved_layer['config']
            bundle['state_dicts'][name]=saved_layer['model'];bundle['architectures'][name]='diffract_adapted_time_v1'
            bundle['selected_steps'][name]=saved_layer['step']
        dashboard.emit('exporting trained dictionaries',force=True)
        atomic_torch_save(bundle,root/'trained_transcoders.pt');del bundle
        test=cached_records('test',plan['test_frames'],config.seed+900)
        results={}
        for layer in config.layers:
            path=root/'test'/f'layer_{layer}.json'
            if path.exists(): results[str(layer)]=json.loads(path.read_text());continue
            if time.monotonic()>=deadline:
                audit_json(root/'training_summary.json',dict(status='training_complete_test_pending',updates_per_layer=total_steps))
                return dict(status='training_complete_test_pending',root=str(root))
            state=torch.load(state_dir/f'best_{layer}.pt',map_location='cpu',mmap=True,weights_only=True)
            model=new_trajectory_model(source.widths[layer],config,layer).to(device)
            model.load_state_dict(state['model']);model.eval();del state
            metrics=validation_with_occupancy(model,test[layer],normalizers[layer])
            action=source.action_metrics(plan['test_frames'],config.seed+900,{layer:model})
            results[str(layer)]=dict(reconstruction=metrics,action=action,frames=len(plan['test_frames']))
            path.parent.mkdir(exist_ok=True);audit_json(path,results[str(layer)]);del model
            dashboard.emit('final test evaluation',force=True,layer=layer,test_layers_complete=len(results))
        summary=dict(status='complete',protocol_hash=protocol_hash,export_sha256=file_sha256(root/'trained_transcoders.pt'),
            layers=list(config.layers),updates_per_layer=total_steps,
            sampled_records_per_layer=total_steps*config.batch_size,unique_train_frames=len(set(plan['train_frames'])),
            episode_split_counts={k:len(v) for k,v in plan['episode_ids'].items()},test=results,
            limitations=['Episode-held-out observations, not unseen tasks or policy-pretraining holdout.',
                'Export selects the best measured validation nMSE separately per layer; layers may come from different updates.',
                'Per-layer normalized-action fidelity only; simultaneous all-layer replacement not measured.',
                'No feature attribution, steering, executed task success or safety claim.',
                'Adapted VLA recipe and finite data budget, not a full DifFRACT reproduction.'])
        audit_json(root/'training_summary.json',summary)
        dashboard.emit('complete',force=True,export=str(root/'trained_transcoders.pt'))
        return summary
    except CollectionBudgetExpired:
        checkpoint();dashboard.emit('paused: session budget reached',force=True)
        return dict(status='paused_budget',root=str(root))
    except BaseException as exc:
        audit_json(root/'failure.json',dict(type=type(exc).__name__,message=str(exc)))
        try: checkpoint();dashboard.emit('interrupted; last committed updates saved',force=True,error=str(exc))
        except Exception as save_error: print('Checkpoint save also failed:',save_error,flush=True)
        raise
    finally:
        audit_json(root/'last_session.json',dict(elapsed_seconds=time.monotonic()-started,requested_minutes=minutes))
        dashboard.render(True)

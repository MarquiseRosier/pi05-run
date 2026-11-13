"""Resumable trajectory-based transcoder training and live notebook observability."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import uuid

import numpy as np
import torch

from pi05_mi.transcoder_methodology import PaperAdaptedTranscoder, atomic_torch_save, reconstruction_metrics
from pi05_mi.transcoders import TimeConditionedTranscoderConfig
from pi05_mi.transcoder_audit import audit_json, file_sha256


@dataclass(frozen=True)
class TrajectoryTrainConfig:
    layers: tuple[int, ...] = tuple(range(18))
    cycles: int = 256
    frames_per_cycle: int = 64
    batch_size: int = 1024
    inference_steps: int = 10
    expansion: int = 16
    lr: float = 2e-4
    lambda_l1: float = 1e-4
    grad_clip: float = 1.
    seed: int = 20261113
    validation_frames: int = 24
    test_frames: int = 16
    reference_frames: int = 16
    validate_every: int = 8
    log_every: int = 25
    action_validation_frames: int = 2

    def validate(self):
        if not self.layers or len(set(self.layers))!=len(self.layers) or min(self.layers)<0:
            raise ValueError('Layers must be unique nonnegative IDs')
        if min(self.cycles,self.frames_per_cycle,self.batch_size,self.inference_steps,self.expansion,
               self.validation_frames,self.test_frames,self.reference_frames,self.validate_every,self.log_every)<1:
            raise ValueError('Training counts must be positive')
        if self.batch_size<2 or self.lr<=0 or self.lambda_l1<0 or self.grad_clip<=0:
            raise ValueError('Invalid optimizer settings')
        if not 0<=self.action_validation_frames<=self.validation_frames:
            raise ValueError('Invalid action validation count')


def training_plan(episodes, config):
    """Episode-disjoint 80/10/10; training frames cover whole episodes without early-frame bias."""
    config.validate()
    if len(episodes)<10: raise ValueError('Need at least 10 episodes for an 80/10/10 split')
    ids=[int(e['episode_index']) for e in episodes]
    if len(set(ids))!=len(ids): raise ValueError('Duplicate episode IDs')
    ranges=sorted((int(e['dataset_from_index']),int(e['dataset_to_index'])) for e in episodes)
    if any(a<0 or b<=a for a,b in ranges) or any(b>c for (_,b),(c,_) in zip(ranges,ranges[1:])):
        raise ValueError('Invalid/overlapping episode frame ranges')
    rng=np.random.default_rng(config.seed); order=rng.permutation(len(episodes))
    nval=max(1,len(order)//10); ntest=nval
    split_indices={'train':order[:len(order)-nval-ntest],
                   'validation':order[len(order)-nval-ntest:len(order)-ntest], 'test':order[-ntest:]}
    pools={}; split_ids={}
    for split,indices in split_indices.items():
        chosen=[episodes[int(i)] for i in indices]
        split_ids[split]=sorted(int(e['episode_index']) for e in chosen)
        pools[split]=np.concatenate([np.arange(int(e['dataset_from_index']),int(e['dataset_to_index'])) for e in chosen])
    count=config.cycles*config.frames_per_cycle
    train=[]
    while len(train)<count: train.extend(rng.permutation(pools['train']).tolist())
    def fixed(split,n):
        if n>len(pools[split]): raise ValueError(f'Not enough {split} frames')
        return rng.choice(pools[split],size=n,replace=False).tolist()
    return dict(episode_ids=split_ids,available_frames={k:len(v) for k,v in pools.items()},
        train_frames=train[:count],reference_frames=fixed('train',config.reference_frames),
        validation_frames=fixed('validation',config.validation_frames),test_frames=fixed('test',config.test_frames),
        split_scope='Episode holdout, not task holdout and not held out from policy pretraining')


def new_trajectory_model(d_model, config, layer):
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(config.seed+layer)
        return PaperAdaptedTranscoder(TimeConditionedTranscoderConfig(d_model=d_model,
            expansion_factor=config.expansion,time_embedding_dim=256,time_hidden_dim=256,eps=1e-6))


def load_trajectory_bundle(path, device='cpu'):
    """Explicit architecture loader: do not load this time network as a legacy dictionary."""
    bundle=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    if bundle.get('format')!='pi05-trajectory-transcoders-v1' or not bundle.get('complete'):
        raise ValueError('Not a complete trajectory-trained bundle')
    models={}
    for name,cfg in bundle['configs'].items():
        if bundle['architectures'][name]!='diffract_adapted_time_v1': raise ValueError('Unknown architecture')
        model=PaperAdaptedTranscoder(TimeConditionedTranscoderConfig(**cfg))
        model.load_state_dict(bundle['state_dicts'][name],strict=True)
        models[name]=model.to(device).eval()
    return models,bundle


def validation_with_occupancy(model,records,normalizer):
    counts=torch.zeros(model.config.latent_dim,device=next(model.parameters()).device,dtype=torch.int64)
    def count(module,inputs,result):
        counts.add_((result[1]>0).reshape(-1,len(counts)).sum(0))
    handle=model.register_forward_hook(count)
    try: result=reconstruction_metrics(model,records,normalizer)
    finally: handle.remove()
    norms=model.decoder.weight.detach().norm(dim=0)
    result.update(dictionary_features=len(counts),observed_never_fired_fraction=float((counts==0).float().mean()),
        decoder_norm_min=float(norms.min()),decoder_norm_max=float(norms.max()),
        occupancy_scope='Fixed validation/test sample only; not a lifetime dead-feature estimate')
    return result


class TrainingDashboard:
    """Updates display IDs, preserving logs/tracebacks; persists JSONL and a labelled PNG."""
    def __init__(self, root, state_dir, config, enabled=True, refresh_seconds=10):
        self.root,self.state_dir,self.config=Path(root),Path(state_dir),config
        self.root.mkdir(parents=True,exist_ok=True); self.state_dir.mkdir(parents=True,exist_ok=True)
        self.path=self.state_dir/'metrics.jsonl'
        self.rows=[json.loads(s) for s in self.path.read_text().splitlines()] if self.path.exists() else []
        self.enabled=enabled; self.refresh_seconds=refresh_seconds; self.last=0.; self.last_image=0.
        self.banner=None; self.figure=None; self.status={}; self.started=time.monotonic();self.last_status=0.

    def emit(self, phase, *, force=False, **values):
        previous=self.status.get('phase');self.status.update(values); self.status['phase']=phase
        event=dict(event='status',phase=phase,wall_time=time.time(),**values)
        if force or previous!=phase or time.monotonic()-self.last_status>=5:
            audit_json(self.root/'live_status.json',event);self.last_status=time.monotonic()
        if force: print(phase,values,flush=True)
        self.render(force)

    def record(self, row):
        row=dict(row,wall_time=time.time())
        with self.path.open('a') as out: out.write(json.dumps(row,allow_nan=False)+'\n')
        self.rows.append(row); self.render(False)

    def render(self, force=False):
        now=time.monotonic()
        if not self.enabled or (not force and now-self.last<self.refresh_seconds): return
        self.last=now
        from IPython.display import display,HTML
        import html
        import matplotlib.pyplot as plt
        title='Trajectory transcoder training — fidelity metrics; attribution NOT tested'
        fields=' | '.join(f'<b>{html.escape(str(k))}</b>: {html.escape(str(v))}' for k,v in self.status.items())
        box=HTML(f'<h3>{title}</h3><p>{fields}</p><p>Model initialization: fresh. '
            'Training objective uses minibatch variance; plotted comparable nMSE uses a frozen training-reference variance. '
            'Validation is episode-held-out. Actions are predicted, never executed.</p>')
        if self.banner is None: self.banner=display(box,display_id=True)
        else: self.banner.update(box)
        fig,axes=plt.subplots(3,3,figsize=(18,13),constrained_layout=True)
        ax=axes.ravel(); layer=int(self.status.get('layer',self.config.layers[0]))
        train=[r for r in self.rows if r.get('event')=='train' and r['layer']==layer]
        val=[r for r in self.rows if r.get('event')=='validation' and r['layer']==layer]
        steps=[r['step'] for r in train]
        for key,label in [('reference_nmse','Train minibatch / fixed reference variance')]:
            ax[0].plot(steps,[r[key] for r in train],'.-',markersize=3,label=label)
        ax[0].plot([r['step'] for r in val],[r['nmse'] for r in val],'o-',label='Fixed validation frames')
        ax[0].set(title=f'Layer {layer}: comparable reconstruction error ↓',xlabel='Optimizer updates for this layer',ylabel='nMSE (fixed denominator)')
        if train or val: ax[0].legend(fontsize=7)
        ax[1].plot(steps,[r['l0'] for r in train],'.-',markersize=3,label='Training minibatch')
        ax[1].plot([r['step'] for r in val],[r['mean_l0'] for r in val],'o-',label='Validation')
        ax[1].set(title=f'Layer {layer}: active features/token',xlabel='Optimizer updates',ylabel='Count of ReLU activations > 0')
        if train or val: ax[1].legend(fontsize=7)
        for key,label in [('objective_nmse','MSE / minibatch variance'),('weighted_l1','λ × mean sum |features|')]:
            ax[2].plot(steps,[r[key] for r in train],'.-',markersize=3,label=label)
        ax[2].set(title=f'Layer {layer}: objective components',xlabel='Optimizer updates',ylabel='Loss contribution')
        if train: ax[2].legend(fontsize=7)
        latest={r['layer']:r for r in self.rows if r.get('event')=='validation'}
        layers=list(self.config.layers)
        ax[3].barh([str(l) for l in layers],[latest.get(l,{}).get('nmse',np.nan) for l in layers])
        ax[3].set(title='Latest validation by layer (blank = not measured)',xlabel='nMSE ↓',ylabel='Action-expert MLP layer')
        grid=np.full((len(layers),self.config.inference_steps),np.nan)
        for i,l in enumerate(layers):
            times=latest.get(l,{}).get('per_time',{})
            for j,t in enumerate(sorted(times,key=float,reverse=True)[:self.config.inference_steps]): grid[i,j]=times[t]['nmse']
        im=ax[4].imshow(np.ma.masked_invalid(grid),aspect='auto',interpolation='nearest')
        ax[4].set(title='Validation error by denoising call',xlabel='Call index (noise → denoised)',ylabel='Layer')
        ax[4].set_yticks(range(len(layers)),layers)
        ax[4].set_xticks(range(self.config.inference_steps),range(1,self.config.inference_steps+1))
        fig.colorbar(im,ax=ax[4],label='nMSE ↓')
        action={r['layer']:r for r in self.rows if r.get('event')=='validation_action'}
        ax[5].barh([str(l) for l in layers],[100*action.get(l,{}).get('relative_l2',np.nan) for l in layers])
        ax[5].set(title=f'Validation action error (n={self.config.action_validation_frames} frames/layer)',xlabel='% relative L2, first 10 normalized actions (not movement)',ylabel='Layer')
        ax[5].set_yticks(range(len(layers)),layers);ax[5].set_ylim(-.5,len(layers)-.5)
        if not action: ax[5].text(.5,.5,'Not yet measured',ha='center',transform=ax[5].transAxes)
        progress={r['layer']:r['step'] for r in self.rows if r.get('event') in ('train','validation')}
        ax[6].barh([str(l) for l in layers],[progress.get(l,0) for l in layers])
        ax[6].set(title='Completed optimizer updates',xlabel='Updates (each layer has its own count)',ylabel='Layer')
        if self.status.get('updates_per_layer'): ax[6].set_xlim(0,self.status['updates_per_layer'])
        ax[7].plot(steps,[r['records_per_second'] for r in train],'.-',markersize=3)
        ax[7].set(title=f'Layer {layer}: optimizer throughput',xlabel='Optimizer updates',ylabel='Sampled activation records / second')
        ax[8].plot(steps,[r['lr'] for r in train],'.-',markersize=3,label='Learning rate')
        ax[8].set(title=f'Layer {layer}: schedule and gradient norm',xlabel='Optimizer updates',ylabel='Learning rate')
        ax[8].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        other=ax[8].twinx();other.plot(steps,[r['grad_norm'] for r in train],'.-',markersize=3,color='orange',alpha=.6)
        other.set_ylabel('Gradient L2 norm before clipping',color='orange')
        for a in ax: a.grid(alpha=.15)
        if self.figure is None: self.figure=display(fig,display_id=True)
        else: self.figure.update(fig)
        if force or now-self.last_image>=60:
            temp=self.root/'live_training.tmp.png';fig.savefig(temp,dpi=110);temp.replace(self.root/'live_training.png')
            shutil.copy2(self.path,self.root/'metrics.jsonl') if self.path.exists() else None
            self.last_image=now
        plt.close(fig)


def publish_training_checkpoint(state_dir, root, protocol_hash, dashboard=None):
    """Immutable shard snapshot; latest pointer is committed last, retain two complete snapshots."""
    root,state_dir=Path(root),Path(state_dir)
    name='snapshot_'+uuid.uuid4().hex[:16]; folder=root/'checkpoints'/name;folder.mkdir(parents=True)
    if dashboard: dashboard.emit('saving checkpoint',force=True,checkpoint=name)
    files={}
    for source in sorted(state_dir.iterdir()):
        if source.is_file() and source.suffix in ('.pt','.json','.jsonl'):
            target=folder/source.name;shutil.copy2(source,target);files[source.name]=file_sha256(target)
            if dashboard: dashboard.emit('saving checkpoint',saved_shards=len(files))
    audit_json(folder/'receipt.json',dict(protocol_hash=protocol_hash,files=files))
    pointer=root/'latest.json';old=json.loads(pointer.read_text()) if pointer.exists() else {}
    keep=[name]+old.get('snapshots',[])[:1]
    audit_json(pointer,dict(protocol_hash=protocol_hash,snapshots=keep))
    for previous in old.get('snapshots',[])[1:]:
        if previous.startswith('snapshot_') and Path(previous).name==previous:
            shutil.rmtree(root/'checkpoints'/previous)
    if dashboard: dashboard.emit('checkpoint saved',force=True,checkpoint=name,persistent_path=str(folder))
    return name


def restore_training_checkpoint(root, state_dir, protocol_hash):
    root,state_dir=Path(root),Path(state_dir);state_dir.mkdir(parents=True,exist_ok=True)
    pointer=root/'latest.json'
    if not pointer.exists():
        if any(state_dir.iterdir()): raise ValueError('Local state exists without a committed checkpoint; use a clean local cache')
        return
    latest=json.loads(pointer.read_text())
    if latest['protocol_hash']!=protocol_hash: raise ValueError('Checkpoint protocol changed')
    folder=root/'checkpoints'/latest['snapshots'][0];receipt=json.loads((folder/'receipt.json').read_text())
    if receipt['protocol_hash']!=protocol_hash: raise ValueError('Snapshot protocol changed')
    for name,digest in receipt['files'].items():
        if Path(name).name!=name or file_sha256(folder/name)!=digest: raise ValueError('Corrupt checkpoint shard')
    for path in state_dir.iterdir():
        if path.is_file(): path.unlink()
    for name in receipt['files']: shutil.copy2(folder/name,state_dir/name)


def optimize_trajectory_layer(model, optimizer, records, normalizer, config, layer, cycle, cursor,
                              updates_per_cycle, dashboard, deadline, device):
    """Deterministic per-update sampling makes interrupted and uninterrupted training identical."""
    total=config.cycles*updates_per_cycle;last_time=time.monotonic();last_cursor=cursor
    while cursor<updates_per_cycle and time.monotonic()<deadline:
        step=cycle*updates_per_cycle+cursor
        g=torch.Generator().manual_seed(config.seed+1000003*cycle+cursor)
        indices=torch.randint(len(records['t']),(config.batch_size,),generator=g)
        x,y,t=(records[k][indices].to(device) for k in ('x','y','t'))
        lr=config.lr*.5*(1+math.cos(math.pi*step/total));optimizer.param_groups[0]['lr']=lr
        optimizer.zero_grad(set_to_none=True)
        pred,z=model(x,t)
        raw_mse=(pred.float()-y.float()).square().sum(-1).mean()
        nmse=raw_mse/(y.float().var(0,unbiased=False).sum()+1e-6)
        weighted_l1=config.lambda_l1*z.float().abs().sum(-1).mean();loss=nmse+weighted_l1
        if not torch.isfinite(loss): raise ValueError('Nonfinite loss; checkpoint preserved, native traceback follows')
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),config.grad_clip,error_if_nonfinite=True)
        optimizer.step();model.renormalize_decoder_columns();cursor+=1
        if cursor%config.log_every==0 or cursor==updates_per_cycle:
            elapsed=max(time.monotonic()-last_time,1e-6)
            dashboard.record(dict(event='train',layer=layer,cycle=cycle+1,step=cycle*updates_per_cycle+cursor,
                reference_nmse=float(raw_mse.detach())/normalizer,objective_nmse=float(nmse.detach()),
                weighted_l1=float(weighted_l1.detach()),loss=float(loss.detach()),l0=float((z>0).sum(-1).float().mean()),
                lr=lr,grad_norm=float(grad),records_per_second=(cursor-last_cursor)*config.batch_size/elapsed))
            last_time=time.monotonic();last_cursor=cursor
        dashboard.emit('training',layer=layer,cycle=f'{cycle+1}/{config.cycles}',
                       optimizer_step=cycle*updates_per_cycle+cursor,
                       gpu_allocated_GiB=round(torch.cuda.memory_allocated()/2**30,2) if str(device).startswith('cuda') else 0)
    return cursor

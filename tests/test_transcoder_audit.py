import ast
import builtins
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from pi05_mi.transcoder_audit import (AuditAccumulator,AuditConfig,FitBucket,action_error,
    checkpoint_evidence,audit_json)
from pi05_mi.transcoders import TimeConditionedTranscoder,TimeConditionedTranscoderConfig
from embed_pi05_transcoder_audit_notebook import definitions,notebook


def test_streaming_moments_equal_direct_calculation_and_survive_merge(tmp_path):
    torch.manual_seed(5)
    x=torch.randn(130,4); y=x*3+20; pred=y+.15*x; z=torch.relu(x)
    expected=float(((pred.double()-y.double())**2).sum()/((y.double()-y.double().mean(0))**2).sum())
    total=FitBucket(); first=FitBucket(); second=FitBucket()
    first.update(x[:50],y[:50],pred[:50],z[:50],.5)
    second.update(x[50:],y[50:],pred[50:],z[50:],.5)
    total.merge(first);total.merge(second)
    assert total.n==130 and total.row()['own_variance_nmse']==pytest.approx(expected)
    acc=AuditAccumulator()
    for start in range(0,130,10):
        acc.update('trajectory',0,x[start:start+10],y[start:start+10],pred[start:start+10],z[start:start+10],.5,start)
    acc.save(tmp_path/'stats.npz'); restored=AuditAccumulator.load(tmp_path/'stats.npz')
    assert restored.buckets['trajectory|0|4'].n==130
    assert restored.buckets['trajectory|0|4'].row()['own_variance_nmse']==pytest.approx(expected)
    assert np.array_equal(restored.occupancy['trajectory|0'],(z>0).sum(0).numpy())


def test_nonfinite_records_rejected_before_stats_change():
    b=FitBucket(); y=torch.ones(2,4); bad=y.clone();bad[0,0]=float('inf')
    with pytest.raises(ValueError,match='Non-finite'): b.update(y,bad,y,y,1.)
    assert b.n==0


def test_all_ten_inference_times_have_separate_bins():
    acc=AuditAccumulator(); x=torch.ones(1,2,4)
    for i in range(10):acc.update('trajectory',0,x,x,x,x,1-i/10,i)
    assert {int(k.split('|')[-1]) for k in acc.buckets}==set(range(10))
    assert all(b.calls==1 for b in acc.buckets.values())


def checkpoint_fixture(layers=3):
    models={f'paligemma_with_expert.gemma_expert.model.layers.{i}.mlp':
        TimeConditionedTranscoder(TimeConditionedTranscoderConfig(d_model=4,d_features=8)) for i in range(layers)}
    return dict(configs={name:dict(d_model=4,d_features=8) for name in models},
        state_dicts={name:m.state_dict() for name,m in models.items()},
        collection_mode='random-timestep',feed_forward=27233,metrics_history=[],
        optimizer_state_dict={'param_groups':[{'params':[0,1]}],'state':{}})


def test_checkpoint_counter_is_not_optimizer_evidence(tmp_path):
    checkpoint=checkpoint_fixture(1)
    evidence=checkpoint_evidence(checkpoint)
    assert evidence['collection_feed_forwards']==27233
    assert evidence['optimizer_update_status']=='no_optimizer_updates_recorded'
    checkpoint['optimizer_state_dict']['state']={0:{'step':torch.tensor(7.)},1:{'step':torch.tensor(9.)}}
    assert checkpoint_evidence(checkpoint)['optimizer_update_status']=='positive_steps_for_all_saved_optimizer_parameters'
    checkpoint['state_dicts'][next(iter(checkpoint['configs']))]['decoder.weight'][0,0]=float('nan')
    checkpoint['metrics_history']=[dict(loss=float('nan'),minibatches=0)]
    evidence=checkpoint_evidence(checkpoint)
    assert not evidence['all_dictionary_weights_finite'] and evidence['nonfinite_history_metrics']
    audit_json(tmp_path/'evidence.json',evidence) # bad weights must still produce inspectable JSON


def test_action_errors_do_not_truncate_shapes_silently():
    a=np.ones((4,2));b=a.copy();b[-1]+=1
    metrics=action_error(a,b,2)
    assert metrics['prefix']['relative_l2']==0 and metrics['full']['relative_l2']>0
    with pytest.raises(ValueError): action_error(a,b[:2],2)
    assert action_error(a,a.copy(),2)['exactly_equal']


def kernel(monkeypatch):
    module=ModuleType('_audit_test_kernel');monkeypatch.setitem(sys.modules,module.__name__,module)
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        assert not name.startswith('pi05_mi')
        assert name not in ('probe_pi05_transcoder_counterfactual','run_pi05_transcoder_audit','subprocess')
        return original(name,*args,**kwargs)
    for cell in definitions():
        with monkeypatch.context() as m:
            m.setattr(builtins,'__import__',guarded)
            exec(compile(''.join(cell['source']),'<notebook '+cell['id']+'>','exec'),module.__dict__)
    return module


def test_notebook_is_literal_and_matches_generator(monkeypatch):
    saved=json.loads((ROOT/'notebooks/pi05_transcoder_training_audit.ipynb').read_text())
    assert saved==notebook()
    for cell in saved['cells']:
        if cell['cell_type']=='code':
            source=''.join(cell['source']); ast.parse(source)
            if cell['id']!='audit-install':
                assert 'subprocess' not in source and 'runpy' not in source
    module=kernel(monkeypatch)
    assert len(module.AuditConfig().cases())==12
    assert module.run_transcoder_audit.__code__.co_filename.startswith('<notebook')


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__();self.up_proj=torch.nn.Linear(4,4);self.down_proj=torch.nn.Linear(4,4)
    def forward(self,x):return self.down_proj(torch.tanh(self.up_proj(x)))


class Core(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config=NS(chunk_size=3,max_action_dim=4)
        self.paligemma_with_expert=torch.nn.Module()
        self.paligemma_with_expert.gemma_expert=torch.nn.Module()
        self.paligemma_with_expert.gemma_expert.model=torch.nn.Module()
        layers=torch.nn.ModuleList([torch.nn.Module() for _ in range(3)])
        for layer in layers:layer.mlp=MLP()
        self.paligemma_with_expert.gemma_expert.model.layers=layers
    def denoise_step(self,prefix_pad_masks,past_key_values,x_t,timestep):
        value=x_t+timestep[:,None,None]
        for layer in self.paligemma_with_expert.gemma_expert.model.layers:value=layer.mlp(value)
        return value
    def sample_time(self,n,device):return torch.rand(n,device=device)


class Policy(torch.nn.Module):
    def __init__(self):super().__init__();self.model=Core()
    def reset(self):pass
    def predict_action_chunk(self,batch,noise,num_steps):
        x=noise.clone()
        for step in range(num_steps):
            x=x-self.model.denoise_step(None,None,x,torch.tensor([1-step/num_steps]))/num_steps
        return x


def test_full_inline_run_counts_pairing_partial_resume_and_manifest(monkeypatch,tmp_path):
    module=kernel(monkeypatch);torch.manual_seed(19)
    policy=Policy()
    setattr(policy,'_pi05_mi_load_report',dict(missing=[],unexpected=[],provided=5))
    builds=[]; closes=[]
    class Inner:
        _init_states=[0,1,2]
        init_state_id=0
        def reset(self,seed):return dict(pixels={'front':np.ones((4,4,3),dtype=np.uint8)},task='pick up bowl'),{}
    def harness(args,load_policy=True):
        builds.append((args.task_id,load_policy))
        return NS(policy=policy,vec_env=NS(close=lambda:closes.append(1)),inner_env=Inner(),
                  preprocess_observation=lambda obs:obs,preprocessor=lambda obs:obs,env_preprocessor=lambda obs:obs)
    module.build_harness=harness
    checkpoint=tmp_path/'source.pt';torch.save(checkpoint_fixture(),checkpoint)
    cfg=module.AuditConfig(tasks=(0,1,2),starts=(0,),inference_steps=2,replacement_layers=(0,2))
    root=tmp_path/'run'
    # Simulate a budget boundary after one complete case.
    times=iter([0,0,100]);module.time=NS(monotonic=lambda:next(times))
    kwargs=dict(source_digest='test',runtime_report={'packages':{}},device='cpu',minutes=1,cache_dir=tmp_path/'cache')
    partial=module.run_transcoder_audit(cfg,checkpoint,root,**kwargs)
    assert partial['completed_cases']==1 and partial['status']=='partial'
    assert len(closes)==2
    saved=(root/'cases/task0_init0_noise0/result.json').read_bytes()
    module.time=NS(monotonic=lambda:0)
    report=module.run_transcoder_audit(cfg,checkpoint,root,**kwargs)
    assert report['completed_cases']==3 and report['status']=='complete'
    assert (root/'cases/task0_init0_noise0/result.json').read_bytes()==saved
    assert report['first_step_control_failures']==0
    assert all(row['records']==18 for row in report['fit']) # 3 cases * 3 tokens * 2 steps
    assert report['actions']['baseline_repeat']['relative_l2']['task_balanced_mean']==0
    assert report['actions']['all_layers']['relative_l2']['task_balanced_mean']>0
    assert len(report['reconstruction_contrasts'])==6
    assert all(row['tasks']==3 and row['observations']==3 for row in report['reconstruction_contrasts'])
    assert len(policy.model.paligemma_with_expert.gemma_expert.model.layers[0].mlp._forward_hooks)==0
    builds_before=len(builds)
    module.run_transcoder_audit(cfg,checkpoint,root,**kwargs)
    assert len(builds)==builds_before # completed resume does not load policy/simulator
    kwargs['source_digest']='changed'
    with pytest.raises(ValueError,match='changed'):module.run_transcoder_audit(cfg,checkpoint,root,**kwargs)


def test_metadata_phase_does_not_load_policy(monkeypatch,tmp_path):
    module=kernel(monkeypatch)
    module.build_harness=lambda *a,**kw:pytest.fail('Metadata phase must not create a policy')
    checkpoint=tmp_path/'source.pt';torch.save(checkpoint_fixture(1),checkpoint)
    evidence=module.run_transcoder_audit(module.AuditConfig(),checkpoint,tmp_path/'run',
        source_digest='metadata',runtime_report={},phase='metadata',cache_dir=tmp_path/'cache')
    assert evidence['collection_mode']=='random-timestep'
    assert (tmp_path/'run/checkpoint_audit.json').exists()

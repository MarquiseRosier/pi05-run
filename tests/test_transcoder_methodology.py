import ast
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import pi05_mi.transcoder_methodology as method
import run_pi05_transcoder_methodology as runner
from embed_pi05_methodology_notebook import notebook, definitions


CFG=dict(d_model=4,d_features=8,time_embedding_dim=4,time_hidden_dim=4)


def small_config(**kwargs):
    return method.MethodConfig(train_tasks=(0,),validation_tasks=(1,),test_tasks=(2,3,4),
        train_starts=(0,),eval_starts=(0,),noise_draws=1,inference_steps=2,
        fit_seeds=(1,2,3),updates=2,batch_size=4,checkpoint_every=1,**kwargs)


def test_splits_and_paired_initialization_and_paper_recipe():
    c=small_config()
    with pytest.raises(ValueError,match='overlap'): replace(c,test_tasks=(0,)).cases()
    models=[method.method_model(CFG,a,1) for a in method.ARMS]
    for m in models[1:3]:
        assert all(torch.equal(v,m.state_dict()[k]) for k,v in models[0].state_dict().items())
    paper=models[3]
    assert paper.config.time_dim==256 and len(paper.time_mlp)==5
    assert torch.allclose(paper.decoder.weight.norm(dim=0),torch.ones(8),atol=1e-6)
    enc=paper.encoder.weight/paper.encoder.weight.norm(dim=1,keepdim=True)
    assert torch.allclose(enc,paper.decoder.weight.T,atol=1e-6)
    scale,shift=paper._time_scale_shift(torch.tensor([.2,.9]),dtype=torch.float32,device='cpu')
    assert torch.count_nonzero(scale)==torch.count_nonzero(shift)==0


def test_loss_holds_normalizer_fixed_except_explicit_recipe_arm():
    model=method.method_model(CFG,method.ARMS[0],1)
    x=torch.arange(24,dtype=torch.float32).reshape(6,4)/20
    y=x*2; t=torch.ones(6)
    losses=[method.method_loss(model,x,t,y,10.,a,1e-4)[0] for a in method.ARMS]
    assert torch.equal(losses[0],losses[1]) and torch.equal(losses[1],losses[2])
    assert not torch.allclose(losses[2],losses[3])


@pytest.mark.parametrize('arm',[method.ARMS[0],method.ARMS[3]])
def test_interrupted_fit_resumes_exact_updates_rng_and_optimizer(tmp_path,monkeypatch,arm):
    c=replace(small_config(),updates=4)
    g=torch.Generator().manual_seed(123)
    data=dict(x=torch.randn(12,4,generator=g),y=torch.randn(12,4,generator=g),t=torch.rand(12,generator=g))
    calls=iter([0,0,0,100])
    monkeypatch.setattr(method.time,'monotonic',lambda:next(calls,100))
    assert not method.fit_method(tmp_path/'resumed',c,CFG,arm,1,data,data,2.,50.,device='cpu')
    state=torch.load(tmp_path/'resumed'/'fits'/f'{arm}_seed1'/'resume.pt',weights_only=True)
    assert state['step']==2
    monkeypatch.setattr(method.time,'monotonic',lambda:0.)
    assert method.fit_method(tmp_path/'resumed',c,CFG,arm,1,data,data,2.,50.,device='cpu')
    assert method.fit_method(tmp_path/'full',c,CFG,arm,1,data,data,2.,50.,device='cpu')
    a=torch.load(tmp_path/'resumed'/'fits'/f'{arm}_seed1'/'model.pt',weights_only=True)
    b=torch.load(tmp_path/'full'/'fits'/f'{arm}_seed1'/'model.pt',weights_only=True)
    assert all(torch.equal(v,b['state_dict'][k]) for k,v in a['state_dict'].items())
    assert not (tmp_path/'resumed'/'fits'/f'{arm}_seed1'/'resume.pt').exists()


class FakePolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model=self.Core()
    class Core(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config=NS(chunk_size=3,max_action_dim=4)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(77)
                self.mlp=torch.nn.Sequential(torch.nn.Linear(4,4),torch.nn.Tanh())
        def sample_time(self,n,device): return torch.rand(n,device=device)*.95+.025
        def denoise_step(self,x_t,timestep,context):
            return self.mlp(x_t+context+timestep[:,None,None])
    def reset(self): pass
    def predict_action_chunk(self,batch,noise,num_steps):
        x=noise
        for i in range(num_steps):
            x=x-self.model.denoise_step(x_t=x,timestep=torch.tensor([1-i/num_steps]),context=batch['x'])/num_steps
        return x[:,:,:2]


def install_fake_runtime(monkeypatch,module):
    calls=[]
    class Env:
        _init_states=[0]
        init_state_id=0
        def reset(self,seed): return dict(pixels=dict(cam=np.full((1,3,4),(seed%13)/20,dtype=np.float32))),{}
        def close(self): pass
    def build(args,load_policy=True):
        calls.append(load_policy)
        env=Env()
        return NS(policy=FakePolicy() if load_policy else None,vec_env=env,inner_env=env,
            preprocess_observation=lambda x:dict(x=torch.from_numpy(x['pixels']['cam'])),
            preprocessor=lambda x:x,env_preprocessor=lambda x:x)
    monkeypatch.setattr(module,'build_harness',build)
    monkeypatch.setattr(module,'_infer_prompt',lambda *a:'original task')
    monkeypatch.setattr(module,'_add_batch_axis',lambda x:x)
    monkeypatch.setattr(module,'assert_weights_loaded',lambda *a,**k:dict(missing=[],provided=2))
    monkeypatch.setattr(module,'list_pi05_action_expert_mlp_targets',lambda p:[NS(layer_index=8,module=p.model.mlp)])
    return calls


def test_hooks_pair_noise_times_and_do_not_contaminate_teacher(monkeypatch):
    install_fake_runtime(monkeypatch,runner)
    p=FakePolicy(); hooks=runner.MethodHooks(p,8)
    batch={'x':torch.ones(1,3,4)}; noise=torch.zeros(1,3,4)
    try:
        native,c=hooks.predict(batch,noise,2,capture=True)
        _,b=hooks.predict(batch,noise,2,mode=method.ARMS[1],capture=True)
        _,a=hooks.predict(batch,noise,2,mode=method.ARMS[0],draw_times=torch.tensor([.3,.8]),capture=True)
        assert torch.equal(c['y'][:3],b['y'][:3])
        assert not torch.equal(c['x'][3:],b['x'][3:])
        assert a['t'].tolist()==pytest.approx([.3]*3+[.8]*3)
        modified,_=hooks.predict(batch,noise,2,replacement='zero')
        repeat,_=hooks.predict(batch,noise,2)
        assert np.array_equal(native,repeat) and not np.array_equal(native,modified)
    finally: hooks.close()


def inline_kernel(monkeypatch):
    module=ModuleType('_method_notebook');monkeypatch.setitem(sys.modules,module.__name__,module)
    for cell in definitions():
        source=''.join(cell['source'])
        for node in ast.walk(ast.parse(source)):
            if isinstance(node,ast.ImportFrom):
                assert not (node.module or '').startswith(('pi05_mi','run_pi05','probe_pi05'))
        exec(compile(source,'<method-notebook>','exec'),module.__dict__)
    return module


def test_literal_notebook_end_to_end_resume_and_manifest(tmp_path,monkeypatch):
    saved=json.loads((ROOT/'notebooks/pi05_transcoder_methodology_controls.ipynb').read_text())
    assert saved==notebook()
    for cell in saved['cells']:
        if cell['cell_type']=='code':
            source=''.join(cell['source']); ast.parse(source)
            if cell['id']!='method-install': assert 'subprocess' not in source and 'runpy' not in source
    module=inline_kernel(monkeypatch); calls=install_fake_runtime(monkeypatch,module)
    name='paligemma_with_expert.gemma_expert.model.layers.8.mlp'
    ckpt=tmp_path/'old.pt'
    torch.save(dict(configs={name:CFG},state_dicts={name:method.method_model(CFG,method.ARMS[0],5).state_dict()}),ckpt)
    policy=tmp_path/'snapshot';policy.mkdir();(policy/'model.safetensors').write_bytes(b'fake-model-for-test')
    root=tmp_path/'run';config=small_config()
    kwargs=dict(source_digest='test',runtime_report={},policy_path=policy,minutes=10,device='cpu',cache_dir=tmp_path/'cache')
    report=module.run_methodology(config,ckpt,root,**kwargs)
    assert report['status']=='complete' and report['completed_model_cases']==39
    assert len(report['comparisons'])==3 and report['controls']['repeat']['task_balanced_mean']==0
    assert report['controls']['zero']['task_balanced_mean']>0
    for arm in method.ARMS:
        for seed in config.fit_seeds:
            done=json.loads((root/'fits'/f'{arm}_seed{seed}'/'complete.json').read_text())
            assert done['updates']==2 and done['records_presented']==8 and done['unique_available_records']==6
    comparison=report['comparisons']['C_trajectory_minus_B_noise_grid']['nmse']
    assert comparison['tasks']==3 and len(comparison['per_seed_mean'])==3
    before=(root/'evaluation'/'task2_init0_noise0'/'existing.json').read_bytes()
    resumed=module.run_methodology(config,ckpt,root,**kwargs)
    assert resumed==report and (root/'evaluation'/'task2_init0_noise0'/'existing.json').read_bytes()==before
    with pytest.raises(ValueError,match='Protocol'):
        module.run_methodology(replace(config,updates=3),ckpt,root,**kwargs)
    # Corrupt raw records are detected rather than silently changing the training dataset.
    path=root/'cases'/'task0_init0_noise0'/'A_noise_sampled.npz';path.write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='Corrupt'):
        module.run_methodology(config,ckpt,root,**kwargs)


def test_partial_report_withholds_comparison(tmp_path):
    report=method.method_report(tmp_path,small_config())
    assert report['status']=='partial' and not report['comparisons']


def test_task_interval_does_not_treat_fit_seeds_as_tasks():
    rows=[]
    for task in (7,8,9):
        for seed in (1,2,3):
            for arm,score in ((method.ARMS[0],0.),(method.ARMS[1],task/10)):
                rows.append(dict(model=f'{arm}_seed{seed}',case=dict(case_id=str(task),task_id=task),nmse=score))
    interval=method.paired_method_interval(rows,method.ARMS[1],method.ARMS[0],'nmse',(1,2,3))
    assert interval['tasks']==3 and interval['observations']==9
    assert interval['task_balanced_mean']==pytest.approx(.8)

import ast
from dataclasses import replace
import json
import shutil
from pathlib import Path
import sys
import time
from types import ModuleType,SimpleNamespace as NS

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import pi05_mi.trajectory_training as core
import run_pi05_trajectory_training as runner
from embed_pi05_trajectory_training_notebook import definitions,notebook,LAUNCH


def config(**kwargs):
    return core.TrajectoryTrainConfig(layers=(0,1),cycles=2,frames_per_cycle=2,batch_size=4,
        inference_steps=2,expansion=2,reference_frames=4,validation_frames=4,test_frames=4,
        validate_every=1,log_every=1,action_validation_frames=1,**kwargs)


EPISODES=[dict(episode_index=i,dataset_from_index=i*8,dataset_to_index=(i+1)*8) for i in range(20)]


def test_episode_split_and_full_frame_schedule_are_disjoint_and_reproducible():
    cfg=replace(config(),cycles=100)
    plan=core.training_plan(EPISODES,cfg)
    assert plan==core.training_plan(EPISODES,cfg)
    assert [len(plan['episode_ids'][s]) for s in ('train','validation','test')]==[16,2,2]
    for split,key in [('train','train_frames'),('validation','validation_frames'),('test','test_frames')]:
        assert {i//8 for i in plan[key]}<=set(plan['episode_ids'][split])
    assert len(set(plan['train_frames'][:128]))==128
    assert not set(plan['train_frames'])&set(plan['validation_frames'])
    bad=EPISODES+[dict(episode_index=30,dataset_from_index=1,dataset_to_index=4)]
    with pytest.raises(ValueError,match='overlapping'): core.training_plan(bad,cfg)


class FakeSource:
    chunk_size=2
    widths={0:4,1:4}
    names={0:'model.layers.0.mlp',1:'model.layers.1.mlp'}
    def __init__(self,module=runner,interrupt=False): self.module=module;self.calls=0;self.interrupt=interrupt
    def collect(self,indices,seed,dashboard,deadline):
        self.calls+=1
        if self.interrupt and self.calls==4: raise self.module.CollectionBudgetExpired()
        g=torch.Generator().manual_seed(seed)
        n=len(indices)*4;x=torch.randn(n,4,generator=g)
        return {l:dict(x=x+l*.1,y=torch.tanh(x*1.2)+l*.2,t=torch.tensor([1.,1.,.5,.5]*len(indices))) for l in (0,1)}
    def action_metrics(self,indices,seed,replacements):
        return dict(relative_l2=.01,cases=len(indices),repeat_floor=0.,rows=[])


def inline_kernel(monkeypatch):
    module=ModuleType('_trajectory_notebook');monkeypatch.setitem(sys.modules,module.__name__,module)
    for cell in definitions():
        source=''.join(cell['source'])
        for node in ast.walk(ast.parse(source)):
            if isinstance(node,ast.ImportFrom):
                assert not (node.module or '').startswith(('pi05_mi','run_pi05','probe_pi05'))
        exec(compile(source,'<trajectory-notebook>','exec'),module.__dict__)
    return module


def test_inline_training_resume_best_export_and_frozen_protocol(tmp_path,monkeypatch):
    module=inline_kernel(monkeypatch);cfg=config()
    interrupted=tmp_path/'interrupted';full=tmp_path/'full'
    source=FakeSource(module,interrupt=True)
    kwargs=dict(minutes=10,checkpoint_minutes=30,charts=False,device='cpu')
    result=module.run_trajectory_training(cfg,source,EPISODES,interrupted,tmp_path/'cache1',{'teacher':'fixed'},**kwargs)
    assert result['status']=='paused_budget'
    pointer=json.loads((interrupted/'latest.json').read_text())
    snapshot=interrupted/'checkpoints'/pointer['snapshots'][0]
    state=torch.load(snapshot/'layer_0.pt',weights_only=True)
    assert state['step']==2
    source.interrupt=False
    result=module.run_trajectory_training(cfg,source,EPISODES,interrupted,tmp_path/'cache1',{'teacher':'fixed'},**kwargs)
    other=module.run_trajectory_training(cfg,FakeSource(module),EPISODES,full,tmp_path/'cache2',{'teacher':'fixed'},**kwargs)
    assert result['status']==other['status']=='complete'
    a=torch.load(interrupted/'trained_transcoders.pt',weights_only=True)
    b=torch.load(full/'trained_transcoders.pt',weights_only=True)
    assert a['selection']=='best validation nMSE per layer' and a['updates_per_layer']==4
    assert set(a['selected_steps'].values())<={2,4}
    for name,weights in a['state_dicts'].items():
        assert all(torch.equal(value,b['state_dicts'][name][key]) for key,value in weights.items())
    models,bundle=core.load_trajectory_bundle(interrupted/'trained_transcoders.pt')
    assert len(models)==2 and bundle['complete']
    assert all(len(m.time_mlp)==5 for m in models.values())
    with pytest.raises(ValueError,match='protocol'):
        module.run_trajectory_training(replace(cfg,cycles=3),source,EPISODES,interrupted,tmp_path/'cache1',{'teacher':'fixed'},**kwargs)


def test_snapshot_pointer_integrity_and_two_snapshot_retention(tmp_path):
    state=tmp_path/'state';state.mkdir();root=tmp_path/'run'
    (state/'metrics.jsonl').write_text('')
    torch.save({'step':1},state/'layer_0.pt')
    for _ in range(3): core.publish_training_checkpoint(state,root,'fixed')
    assert len(list((root/'checkpoints').iterdir()))==2
    restored=tmp_path/'restored';core.restore_training_checkpoint(root,restored,'fixed')
    assert torch.load(restored/'layer_0.pt',weights_only=True)['step']==1
    pointer=json.loads((root/'latest.json').read_text())
    (root/'checkpoints'/pointer['snapshots'][0]/'layer_0.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='Corrupt'): core.restore_training_checkpoint(root,restored,'fixed')


def test_validation_failure_resumes_without_repeating_optimizer_updates(tmp_path):
    class InterruptedValidation(FakeSource):
        failed=False
        def action_metrics(self,*args,**kwargs):
            if not self.failed:
                self.failed=True
                raise RuntimeError('simulated validation interruption')
            return super().action_metrics(*args,**kwargs)
    source=InterruptedValidation();cfg=config();root=tmp_path/'run';local=tmp_path/'local'
    args=dict(minutes=10,checkpoint_minutes=30,charts=False,device='cpu')
    with pytest.raises(RuntimeError,match='validation interruption'):
        runner.run_trajectory_training(cfg,source,EPISODES,root,local,{},**args)
    report=runner.run_trajectory_training(cfg,source,EPISODES,root,local,{},**args)
    assert report['status']=='complete' and report['updates_per_layer']==4
    latest=json.loads((root/'latest.json').read_text())['snapshots'][0]
    state=root/'checkpoints'/latest
    for l in cfg.layers:
        assert json.loads((state/f'validated_{l}.json').read_text())['step']==4
        assert torch.load(state/f'layer_{l}.pt',weights_only=True)['step']==4


def test_notebook_is_current_literal_python_and_has_chart_setup():
    saved=json.loads((ROOT/'notebooks/pi05_transcoder_trajectory_training.ipynb').read_text())
    assert saved==notebook()
    for cell in saved['cells']:
        if cell['cell_type']=='code':
            source=''.join(cell['source']);ast.parse(source)
            if cell['id'] not in ('trajectory-install','trajectory-extra-install'):
                assert 'subprocess' not in source and 'runpy' not in source
    assert 'matplotlib' in ''.join(next(c['source'] for c in saved['cells'] if c['id']=='trajectory-extra-install'))


def test_launch_checks_cache_and_dataset_before_hashing_or_training(tmp_path):
    # Execute the actual launch cell with setup omitted, as after a runtime restart.
    namespace=dict(TRAJECTORY_TRAINING_VERSION='test',POLICY_REPO='test/policy',Path=Path)
    with pytest.raises(RuntimeError,match='Prepare Model Cache And LIBERO Assets'):
        exec(LAUNCH,namespace)
    namespace['snapshots']={}
    with pytest.raises(RuntimeError,match='Prepare Model Cache And LIBERO Assets'):
        exec(LAUNCH,namespace)
    namespace['snapshots']={repo:str(tmp_path) for repo in
        ('test/policy','google/paligemma-3b-pt-224','lerobot/libero-assets')}
    with pytest.raises(RuntimeError,match='Open The Full Observation Dataset'):
        exec(LAUNCH,namespace)


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.model=self.Core()
    class Core(torch.nn.Module):
        def __init__(self):
            super().__init__();self.config=NS(chunk_size=2,max_action_dim=4)
            self.mlps=torch.nn.ModuleList([torch.nn.Linear(4,4),torch.nn.Linear(4,4)])
        def denoise_step(self,x_t,timestep,context):
            x=x_t+context+timestep[:,None,None]
            for mlp in self.mlps: x=mlp(x)
            return x
    def reset(self):pass
    def predict_action_chunk(self,batch,noise,num_steps):
        x=noise
        for i in range(num_steps):
            x=x-self.model.denoise_step(x_t=x,timestep=torch.tensor([1-i/num_steps]),context=batch['observation.state'])/num_steps
        return x[:,:,:2]


def test_native_multilayer_collection_counts_times_and_no_teacher_mutation(monkeypatch,tmp_path):
    policy=Policy().eval().requires_grad_(False);meta=NS(camera_keys=[])
    class Data:
        def __init__(self):self.meta=meta
        def __getitem__(self,i):return {'index':torch.tensor(i),'observation.state':torch.ones(2,4)*i,'task':'original instruction'}
    monkeypatch.setattr(runner,'build_harness',lambda args:NS(policy=policy,vec_env=NS(close=lambda:None),preprocessor=lambda x:x))
    monkeypatch.setattr(runner,'assert_weights_loaded',lambda *a,**k:{'missing':[]})
    monkeypatch.setattr(runner,'list_pi05_action_expert_mlp_targets',lambda p:[NS(layer_index=i,name=f'layer{i}',d_model=4,module=m) for i,m in enumerate(p.model.mlps)])
    source=runner.DatasetTrajectorySource(Data(),'fake',config(),device='cpu')
    dashboard=core.TrainingDashboard(tmp_path,tmp_path/'state',config(),enabled=False)
    before=source.predict(0,123)
    try:
        records=source.collect([0,1],123,dashboard,time.monotonic()+20)
        assert all(r['x'].shape==(8,4) for r in records.values())
        assert records[0]['t'].tolist()==[1.,1.,.5,.5]*2
        # The second denoising call must receive the evolved action state, not fresh noise.
        noise=torch.randn((1,2,4),generator=torch.Generator().manual_seed(123))
        first_input=noise+1.
        first_velocity=first_input
        for mlp in policy.model.mlps:
            first_velocity=torch.nn.functional.linear(first_velocity,mlp.weight,mlp.bias)
        second_input=noise-first_velocity/2+.5
        torch.testing.assert_close(records[0]['x'][:2],first_input[0])
        torch.testing.assert_close(records[0]['x'][2:4],second_input[0])
        assert not torch.allclose(records[0]['x'][2:4],noise[0]+.5)
        assert np.array_equal(before,source.predict(0,123))
        model=core.new_trajectory_model(4,config(),0)
        result=source.action_metrics([0,1],123,{0:model})
        assert result['cases']==2 and result['repeat_floor']==0 and result['relative_l2']>0
    finally:source.close()


def test_resume_preserves_reference_variance_without_recollection(tmp_path):
    cfg=config();source=FakeSource(interrupt=True)
    root,local=tmp_path/'run',tmp_path/'local'
    kwargs=dict(minutes=10,checkpoint_minutes=30,charts=False,device='cpu')
    assert runner.run_trajectory_training(cfg,source,EPISODES,root,local,{},**kwargs)['status']=='paused_budget'
    snapshot=json.loads((root/'latest.json').read_text())['snapshots'][0]
    original=(root/'checkpoints'/snapshot/'normalizers.json').read_bytes()
    shutil.rmtree(local)
    class NoReference(FakeSource):
        def collect(self,indices,seed,*args):
            assert seed!=cfg.seed+700,'Frozen reference must not be collected again on resume'
            return super().collect(indices,seed,*args)
    assert runner.run_trajectory_training(cfg,NoReference(),EPISODES,root,local,{},**kwargs)['status']=='complete'
    snapshot=json.loads((root/'latest.json').read_text())['snapshots'][0]
    assert (root/'checkpoints'/snapshot/'normalizers.json').read_bytes()==original


def test_training_update_matches_explicit_loss_and_decoder_projection(tmp_path):
    import copy
    cfg=config();model=core.new_trajectory_model(4,cfg,0);expected=copy.deepcopy(model)
    records=FakeSource().collect([0,1],17,None,None)[0]
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=0.)
    other=torch.optim.AdamW(expected.parameters(),lr=cfg.lr,weight_decay=0.)
    idx=torch.randint(len(records['t']),(cfg.batch_size,),generator=torch.Generator().manual_seed(cfg.seed))
    target=records['y'][idx];prediction,z=expected(records['x'][idx],records['t'][idx])
    # Independent scalar expression, including the population (not sample) variance.
    error=((prediction-target)**2).sum()/cfg.batch_size
    variance=((target-target.mean(0))**2).sum()/cfg.batch_size
    loss=error/(variance+1e-6)+cfg.lambda_l1*z.abs().sum()/cfg.batch_size
    loss.backward();torch.nn.utils.clip_grad_norm_(expected.parameters(),cfg.grad_clip)
    other.step();expected.renormalize_decoder_columns()
    dash=core.TrainingDashboard(tmp_path,tmp_path/'state',cfg,enabled=False)
    assert core.optimize_trajectory_layer(model,optimizer,records,1.,cfg,0,0,0,1,dash,time.monotonic()+30,'cpu')==1
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,expected.state_dict()[key])
    torch.testing.assert_close(model.decoder.weight.norm(dim=0),torch.ones(cfg.expansion*4))


def test_dashboard_renders_labelled_artifact(tmp_path,monkeypatch):
    import matplotlib
    matplotlib.use('Agg')
    import IPython.display
    monkeypatch.setattr(IPython.display,'display',lambda *a,**k:NS(update=lambda *a:None))
    cfg=replace(config(),layers=tuple(range(18)))
    dash=core.TrainingDashboard(tmp_path,tmp_path/'state',cfg,enabled=False)
    for layer in cfg.layers:
        dash.record(dict(event='validation',layer=layer,step=32,nmse=.1+layer*.01,mean_l0=500+layer,
            per_time={'1.000000':{'nmse':.05},'.500000':{'nmse':.15}}))
    dash.record(dict(event='train',layer=8,step=32,reference_nmse=.12,objective_nmse=.13,
        weighted_l1=.01,l0=508,records_per_second=12000,lr=.0002,grad_norm=.4))
    dash.enabled=True;dash.emit('training',force=True,layer=8,cycle='1/256')
    assert (tmp_path/'live_training.png').stat().st_size>10000
    assert (tmp_path/'metrics.jsonl').exists()

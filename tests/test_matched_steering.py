"""Causal controls, leakage barriers, and a complete CPU experiment harness."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from pi05_mi.matched_steering import (SteeringConfig, preference, fit_layer_handles,
    intervention_delta, select_layers, paired_report, SteeringHooks, probe_auc)
from run_pi05_matched_steering import contrast_prompts


def test_test_tasks_never_share_training_or_validation_starts():
    config = SteeringConfig()
    cases = config.cases()
    assert len({c['case_id'] for c in cases}) == len(cases)
    tasks = {s: {c['task_id'] for c in cases if c['split'] == s} for s in ('train', 'validation', 'test')}
    assert not tasks['train'] & tasks['validation'] and not tasks['train'] & tasks['test']
    assert not tasks['validation'] & tasks['test']
    fixed=json.loads((ROOT/'configs/pi05_matched_steering_test_set.json').read_text())
    assert fixed['test_cases']==[c for c in cases if c['split']=='test']
    with pytest.raises(ValueError, match='disjoint'):
        SteeringConfig(test_tasks=(0,))


@pytest.mark.parametrize('path', sorted((ROOT / 'tests/fixtures/libero_spatial').glob('*.bddl')), ids=lambda p:p.stem)
def test_opposite_prompts_reference_actual_bowl_regions(path):
    prompts, spec, sources = contrast_prompts(NS(_env=NS(env=NS(bddl_file_name=str(path)))))
    assert prompts['1'] != prompts['-1'] and sources
    assert spec['target_instance'] == 'akita_black_bowl_1'


def test_movement_score_has_physical_sign_and_translation_invariance():
    assert preference([1,0,0], [1,0,0], [-1,0,0]) == 1
    assert preference([-1,0,0], [1,0,0], [-1,0,0]) == -1
    a = preference([.2,.1,.4], [1,0,0], [-1,0,0])
    assert preference([2.2,2.1,2.4], [3,2,2], [1,2,2]) == pytest.approx(a)
    with pytest.raises(ValueError, match='1 cm'):
        preference([0,0,0], [0,0,0], [0,0,0])


def test_measurement_step_never_calls_lerobot_autoreset():
    from run_pi05_matched_steering import step_without_autoreset
    def forbidden(*args):
        raise AssertionError('LeRobot would reset the simulator here')
    observation={'robot0_eef_pos':np.array([.1,.2,.3])}
    inner=NS(step=forbidden, reset=forbidden,
             _env=NS(step=lambda action:(observation,1.,False,{}),check_success=lambda:True))
    raw,done,success,_=step_without_autoreset(inner,np.zeros(7))
    assert raw is observation and success and not done


def arm():
    return dict(direction=[1.,0.], center=[0.,0.], random_direction=[0.,1.],
                indices=[0], contrast=[1.], random_indices=[1], random_contrast=[1.])


@pytest.mark.parametrize('family', ['dense','sparse'])
def test_zero_is_exact_and_steering_budget_is_matched(family):
    y = torch.tensor([[[2.,3.], [4.,5.]]])
    z = torch.ones(1,2,2)
    decoder = torch.eye(2)
    zero, norms = intervention_delta(y,z,decoder,arm(),family=family,mode='zero',sign=1,budget=.1)
    assert torch.equal(y+zero,y) and norms['max_relative_rms'] == 0
    delta, norms = intervention_delta(y,z,decoder,arm(),family=family,mode='steer',sign=1,budget=.1)
    assert norms['max_relative_rms'] <= .100001
    assert torch.all(delta[...,0] > 0) and torch.all(delta[...,1] == 0)
    negative, _ = intervention_delta(y,z,decoder,arm(),family=family,mode='steer',sign=-1,budget=.1)
    assert torch.all(negative[...,0] < 0)


def test_sparse_suppression_cannot_make_latent_negative():
    y = torch.tensor([[[2.,3.]]]); z = torch.tensor([[[.001, 1.]]])
    delta, norms = intervention_delta(y,z,torch.eye(2),arm(),family='sparse',mode='steer',sign=-1,budget=.5)
    assert abs(float(delta[0,0,0])) <= .001001
    assert norms['mean_relative_rms'] < .01
    # The native residual is retained, rather than substituting decoder(z).
    torch.testing.assert_close(y+delta,torch.tensor([[[1.999,3.]]]))


@pytest.mark.parametrize('budget',[.0001,.01,.1])
def test_budget_is_enforced_after_bfloat16_quantization(budget):
    y=torch.tensor([[[2.,3.],[4.,5.]]],dtype=torch.bfloat16)
    for family in ('dense','sparse'):
        delta,norms=intervention_delta(y,torch.ones(1,2,2),torch.eye(2),arm(),
            family=family,mode='steer',sign=1,budget=budget)
        measured=(y+delta).float()-y.float()
        relative=measured.square().mean(-1).sqrt()/y.float().square().mean(-1).sqrt()
        assert float(relative.max())<=budget+1e-7
        assert norms['max_relative_rms']<=budget+1e-7


def test_sparse_and_dense_ablation_remove_only_the_selected_component():
    y = torch.tensor([[[.1,1.]]]); z = torch.tensor([[[.1,1.]]])
    for family in ('dense','sparse'):
        delta,_ = intervention_delta(y,z,torch.eye(2),arm(),family=family,mode='ablate',sign=1,budget=1.)
        torch.testing.assert_close(y+delta, torch.tensor([[[0.,1.]]]))


def test_hook_uses_native_mlp_output_and_decoder_delta_not_reconstruction():
    from pi05_mi.transcoders import TimeConditionedTranscoder, TimeConditionedTranscoderConfig
    tc = TimeConditionedTranscoder(TimeConditionedTranscoderConfig(d_model=2, d_features=2))
    with torch.no_grad():
        tc.encoder.weight.copy_(torch.eye(2));tc.encoder.bias.zero_()
        tc.decoder.weight.copy_(torch.eye(2));tc.decoder.bias.fill_(1000.)
    native = torch.nn.Linear(2,2)
    with torch.no_grad():native.weight.copy_(2*torch.eye(2));native.bias.fill_(3.)
    context = NS(timestep_for=lambda x: torch.ones(x.shape[0]))
    hooks = SteeringHooks([NS(name='layer',layer_index=0,module=native)], {'layer':tc}, context)
    x=torch.ones(1,2,2)
    hooks.reset(capture=True)
    with torch.inference_mode(): baseline=native(x)
    assert hooks.arrays()['dense_0'].tolist()==[5.,5.]
    assert hooks.arrays()['sparse_0'].tolist()==[1.,1.]
    hooks.reset(spec=dict(handle={'layer':0,'sparse':arm()},family='sparse',mode='steer',sign=1,budget=.1))
    with torch.inference_mode(): changed=native(x)
    assert float(changed.max()) < 6 and torch.equal(changed[...,1],baseline[...,1])
    hooks.close()
    torch.testing.assert_close(native(x),baseline)


def test_handles_discover_signal_and_random_sparse_units_are_disjoint():
    rng=np.random.default_rng(7);labels=np.tile([-1,1],20)
    dense=rng.normal(size=(40,8));dense[:,2]+=labels*3
    sparse=np.maximum(0,2+rng.normal(size=(40,20)));sparse[:,4]+=labels*.8
    handle=fit_layer_handles(dense,sparse,labels,layer=0,k=3,seed=2)
    assert abs(handle['dense']['direction'][2])>.5
    held_out=rng.normal(size=(100,8)); held_labels=np.tile([-1,1],50); held_out[:,2]+=held_labels*3
    assert probe_auc(handle,'dense',held_out,held_labels)>.95
    assert 4 in handle['sparse']['indices']
    assert not set(handle['sparse']['indices']) & set(handle['sparse']['random_indices'])


def test_layer_selection_is_shared_and_cannot_use_test_results():
    rows=[dict(split='validation',family=f,layer=l,score=s) for f,l,s in
          [('dense',0,10),('sparse',0,0),('dense',8,8),('sparse',8,8)]]
    chosen,_=select_layers(rows,[{'layer':0},{'layer':8}])
    assert chosen['dense']['layer']==chosen['sparse']['layer']==8
    rows[0]['split']='test'
    with pytest.raises(ValueError,match='validation'):
        select_layers(rows,[{'layer':0},{'layer':8}])


def test_report_requires_both_arms_directions_and_task_clusters():
    rows=[dict(split='test',case_id=f't{t}',task_id=t,mode='steer',dose=1.,desired_sign=s,family=f,score=v)
          for t in range(3) for s in (-1,1) for f,v in [('dense',2),('sparse',5)]]
    report=paired_report(rows,draws=30)
    assert report['sparse_minus_dense']==3 and report['task_bootstrap_95ci']==[3,3]
    with pytest.raises(ValueError,match='paired'):
        paired_report(rows[:-1])


def test_complete_experiment_freezes_before_test_and_resumes(tmp_path,monkeypatch):
    import run_pi05_matched_steering as runner
    import pi05_monitor_runtime
    checkpoint=tmp_path/'checkpoint.pt';checkpoint.write_bytes(b'fake fixture')
    root=tmp_path/'run'
    calls=[]
    class FakeRuntime:
        def __init__(self,*args):pass
        def set_case(self,case):
            if case['split']=='test': assert (root/'frozen_selection.json').exists()
        def trial(self,case,recipient_sign,*,spec=None,capture=False):
            calls.append((case['case_id'],recipient_sign,spec is not None))
            endpoint=.3*recipient_sign
            if spec and spec['mode']!='zero': endpoint+=spec['sign']*(.1 if spec['family']=='dense' else .2)*spec['budget']/.1
            result=dict(**case,initial_state_sha256=case['case_id'],initial_images_sha256=case['case_id'],
                        final_preference=endpoint,initial_preference=0.,eef_displacement_m=[endpoint,0,0],
                        executed_actions=[[endpoint]*7],trajectory_xyz=[[0,0,0],[endpoint,0,0]],
                        harm=False,steps=1,intervention_norms=[{'mean_relative_rms':spec['budget'] if spec else 0.}])
            arrays={'dense_0':np.array([recipient_sign,case['init_state_id']*.01,1.,2.]),
                    'sparse_0':2+recipient_sign*np.linspace(-.9,.9,20)+case['init_state_id']*.01}
            return result,arrays,{'image':np.zeros((4,4,3),np.uint8)}
        def close(self):pass
    monkeypatch.setattr(runner,'SteeringRuntime',FakeRuntime)
    monkeypatch.setattr(pi05_monitor_runtime,'validate_runtime',lambda **kwargs:{'status':'fixture'})
    config=SteeringConfig(train_tasks=(0,),validation_tasks=(1,),test_tasks=(2,),
                          train_starts=3,validation_starts=2,test_starts=2,layers=(0,),sparse_k=2)
    report=runner.run_steering_experiment(config,checkpoint,root,device='cpu')
    assert report['sparse_minus_dense']==pytest.approx(5.)
    assert report['test_starts']==2
    assert {row['split'] for row in report['trials']}=={'test'}
    discovery=json.loads((root/'discovery.json').read_text())
    assert all(case.startswith('task0_') for case in discovery['included_training_cases'])
    before=len(calls)
    runner.run_steering_experiment(config,checkpoint,root,device='cpu',phase='test')
    assert len(calls)==before
    with pytest.raises(ValueError,match='protocol changed'):
        runner.run_steering_experiment(config,checkpoint,root,device='cpu',source_digest='changed')

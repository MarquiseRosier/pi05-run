"""Preliminary-result gates: denominators, paired sampling, time units and literal notebooks."""
import ast
import builtins
import hashlib
import io
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'scripts')]
from pi05_mi.occlusion_pilot import PilotConfig, round_robin_groups, labels_at_steps, atomic_json, manifest
from fit_pi05_occlusion_monitor import delayed_operating_metrics, endpoint_view, paired_monitor_difference, prediction_metrics
from fit_pi05_occlusion_monitor import threshold_from_good_episodes, operating_metrics
from run_pi05_occlusion_pilot import verify_monitor_start, Slot, write_images
from pi05_monitor_evidence import inventory_run, report_for_run, export_evidence
from run_pi05_abstract_pilot import freeze_abstract_protocol, run_abstract_pilot
from embed_pi05_abstract_notebooks import notebook, NAMES


def test_round_robin_preserves_every_pair_and_rotates_task_and_split():
    plan = PilotConfig(tasks=(0,1,2),train_starts=4,calibration_starts=4,test_starts=4).plan()
    ordered = round_robin_groups(plan)
    assert len(ordered)==156 and {j['episode_id'] for j in plan}=={j['episode_id'] for j in ordered}
    groups = []
    for job in ordered:
        if not groups or job['group_id'] != groups[-1]['group_id']:
            groups.append(job)
    assert len(groups)==36 and len({g['group_id'] for g in groups})==36
    assert [(g['task_id'],g['split']) for g in groups[:3]]==[(0,'train'),(1,'calibration'),(2,'test')]
    partial = ordered[2:15] + ordered[25:]
    assert sorted(j['episode_id'] for j in round_robin_groups(partial))==sorted(j['episode_id'] for j in partial)
    assert round_robin_groups([])==[]


def episode(name, harm, *, task=0, group=None, success=False):
    return dict(episode_id=name, group_id=group or name, task_id=task, first_harm_step=harm,
                success=success, steps=200, sample_steps=np.array([0,8,104]), control_freq=20)


def test_delay_seconds_strictly_precedes_harm_and_misses_remain_in_denominator():
    eps = [episode('a',100), episode('b',101), episode('miss',101), episode('late',100)]
    scores = {'a':np.array([1,0,0]),'b':np.array([1,0,0]),'miss':np.zeros(3),'late':np.array([0,0,1])}
    result = delayed_operating_metrics(eps,scores,.5,5)
    assert result['delay_control_steps']==100 and result['delay_seconds']==5
    assert result['early_warnings']==1 and result['harmful_episodes']==4
    assert result['harm_preceded_fraction']==.25
    assert delayed_operating_metrics(eps,scores,.5,0)['early_warnings']==2
    with pytest.raises(ValueError): delayed_operating_metrics(eps,scores,.5,-1)


def test_exact_future_window_alignment_and_censoring():
    ep = dict(first_harm_step=48,steps=50,success=False)
    y,keep=labels_at_steps([0,8,16,40,48],ep,target='future_harm',future_steps=40)
    assert y.tolist()==[0,1,1,1,0] and keep.tolist()==[True,True,True,True,False]
    y,keep=labels_at_steps([0,8,16,40],dict(first_harm_step=None,steps=48,success=True),target='future_harm',future_steps=40)
    assert keep.tolist()==[True,True,False,False] and not y.any()


def test_bootstrap_keeps_task_strata_and_reports_absent_classes():
    eps = [episode(f'{t}-{i}',80 if i else None, task=t, success=not i) for t in range(3) for i in range(2)]
    early={e['episode_id']:np.array([1.,0,0]) for e in eps}
    miss={e['episode_id']:np.zeros(3) for e in eps}
    result=paired_monitor_difference(eps,early,.5,miss,.5,draws=500)
    coverage=result['harm_preceded_fraction']
    assert coverage['difference']==1 and coverage['task_ids']==[0,1,2] and coverage['groups']==6
    assert coverage['valid_draws']+coverage['invalid_draws']==500
    assert coverage['invalid_draws']>0
    assert coverage['paired_group_bootstrap_95ci']==[1.,1.]
    # A single group per task cannot manufacture an interval from duplicated frames.
    small=paired_monitor_difference(eps[::2],early,.5,miss,.5,draws=20)
    assert small['false_stop_fraction']['paired_group_bootstrap_95ci'] is None


def test_endpoint_sensitivity_is_an_immutable_fixed_monitor_view():
    ep=episode('a',3)
    ep['physics']=[dict(step=1,peak_contact_n=6,max_wrong_bowl_lift_m=0),
                   dict(step=2,peak_contact_n=12,max_wrong_bowl_lift_m=0),
                   dict(step=3,peak_contact_n=0,max_wrong_bowl_lift_m=.06)]
    for threshold, first in [(5,1),(10,2),(20,3)]:
        view=endpoint_view([ep],threshold)[0]
        assert view['first_harm_step']==first and view['first_wrong_lift_step']==3
    assert ep['first_harm_step']==3 and 'first_wrong_lift_step' not in ep


def test_one_class_prediction_returns_counts_and_no_metric():
    result=prediction_metrics([0,0,0],[.1,.2,.3],[.5,.5,1])
    assert result['status']=='not_estimable' and result['negative_windows']==3 and result['auprc'] is None
    assert prediction_metrics([],[],[])['windows']==0


def test_float32_ties_do_not_undo_conservative_calibration_under_numpy2():
    score=np.array([.7,.7,.7],dtype=np.float32)
    threshold=threshold_from_good_episodes(np.array([score.max()],dtype=np.float32),.1)
    ep=episode('good',None,success=True)
    metrics=operating_metrics([ep],{'good':score},threshold)
    assert threshold>float(score.max()) and metrics['false_stops']==0


def test_monitor_reset_guard_checks_unoccluded_pixels_and_physics(tmp_path):
    job=dict(group_id='task4_init0',episode_id='task4_init0_clean')
    state=np.zeros(10); pixels={'image':np.zeros((3,4,3),np.uint8)}
    expected=verify_monitor_start(tmp_path,job,state,pixels)
    assert verify_monitor_start(tmp_path,dict(job,episode_id='task4_init0_box'),state,pixels)==expected
    changed=state.copy();changed[1]=np.nextafter(0.,1.)
    with pytest.raises(RuntimeError,match='before inference'):
        verify_monitor_start(tmp_path,job,changed,pixels)
    pixels['image'][0,0,0]=1
    with pytest.raises(RuntimeError,match='before inference'):
        verify_monitor_start(tmp_path,job,state,pixels)
    assert json.loads((tmp_path/'episodes'/job['episode_id']/'reset_diagnostics.json').read_text())['changed_cameras']==['image']


def test_terminal_contact_is_measured_without_lerobot_autoreset():
    calls=[]
    inner=NS(_env=NS(step=lambda a: calls.append('physical_step') or (None,0,False,{}),check_success=lambda:True))
    def forbidden(a): raise AssertionError('LeRobot inner.step resets and must not run')
    inner.step=forbidden
    slot=object.__new__(Slot);slot.inner=inner;slot.steps=0;slot.config=NS(horizon=280)
    slot.physics=NS(finish_control_step=lambda step: calls.append('contact') or {'step':step,'peak_contact_n':12})
    slot.events=[];slot.event_log=io.StringIO();slot.success=False
    slot.advance(np.zeros(7))
    assert calls==['physical_step','contact'] and slot.success and slot.done and slot.steps==1
    assert slot.events[0]['peak_contact_n']==12


def test_policy_input_png_is_lossless(tmp_path):
    from PIL import Image
    rng=np.random.default_rng(3); image=rng.integers(0,256,(10,11,3),dtype=np.uint8)
    masks={'bowl__before':np.ones((10,11),bool),'bowl__after':np.zeros((10,11),bool)}
    write_images(tmp_path,{'image':image},{'image':(masks,np.zeros((10,11,4),np.uint8))})
    np.testing.assert_array_equal(np.asarray(Image.open(tmp_path/'image.png')), image)


def test_all_new_notebooks_are_generated_current_and_parse():
    for which,name in NAMES.items():
        actual=json.loads((ROOT/'notebooks'/name).read_text())
        assert actual==notebook(which)
        for cell in actual['cells']:
            if cell['cell_type']=='code': ast.parse(''.join(cell['source']))
    a=json.dumps(notebook('a'))
    assert '--no-build-isolation-package' in a and 'cmake==3.31.6' in a
    assert 'torch==2.10.0' in a and 'Restart session' in a
    assert 'run_abstract_pilot' in a and 'fit_function=fit_occlusion_monitor' in a
    assert 'run_pi05_occlusion_pilot import' not in a


@pytest.fixture
def kernel(monkeypatch):
    import torch, sklearn, matplotlib, concurrent.futures
    module=ModuleType('_abstract_notebook_test');monkeypatch.setitem(sys.modules,module.__name__,module)
    module.LOCAL_REPO=ROOT
    original=builtins.__import__
    forbidden={'pi05_mi.occlusion_pilot','pi05_mi.monitor_features','run_pi05_occlusion_pilot',
               'fit_pi05_occlusion_monitor','run_pi05_monitor_budget','pi05_monitor_evidence','run_pi05_abstract_pilot','subprocess'}
    def guard(name,*args,**kwargs):
        if name in forbidden: raise AssertionError('Must execute notebook definitions: '+name)
        return original(name,*args,**kwargs)
    for cell in notebook('a')['cells']:
        if cell.get('id','').startswith(('pilot-','matched-feature-','evidence-definitions','sprint-definitions')):
            with monkeypatch.context() as scope:
                scope.setattr(builtins,'__import__',guard)
                exec(compile(''.join(cell['source']),'<cell '+cell['id']+'>','exec'),module.__dict__)
    return module


def test_literal_notebook_collection_plan_uses_layer8_and_156_episodes(kernel,tmp_path):
    kernel.collect_occlusion_pilot(['--plan-only','--output-dir',str(tmp_path),'--tasks','0,1,2',
        '--train-starts','4','--calibration-starts','4','--test-starts','4','--layers','8','--image-stride','8'])
    data=json.loads((tmp_path/'manifest.json').read_text())
    assert len(data['episodes'])==156 and data['config']['layers']==[8] and data['image_stride']==8
    assert 'no_terminal_autoreset' in data['reset_protocol']


def test_protocol_is_frozen_and_budget_does_not_reset(tmp_path):
    args=dict(checkpoint='trained.pt',batch_size=4,started_unix=1000,source_digest='abc')
    p=freeze_abstract_protocol(tmp_path,**args)
    assert p['deadline_unix']==8200 and p['planned_episodes']==156 and p['layers']==[8]
    assert freeze_abstract_protocol(tmp_path,**args)==p
    with pytest.raises(ValueError,match='Frozen sprint changed'):
        freeze_abstract_protocol(tmp_path,**dict(args,batch_size=1))
    assert json.loads((tmp_path/'sprint_protocol.json').read_text())['batch_size']==4


def test_expired_setup_exports_instead_of_loading_policy(monkeypatch,tmp_path):
    import run_pi05_abstract_pilot as runner
    p=freeze_abstract_protocol(tmp_path,checkpoint='trained.pt',batch_size=1,started_unix=1000,source_digest='a')
    exported=[]
    monkeypatch.setattr(runner,'export_evidence',lambda root: exported.append(root) or root)
    def forbidden(args): raise AssertionError('Must not start after setup deadline')
    assert runner.run_abstract_pilot(tmp_path,p,collect_function=forbidden,fit_function=forbidden)==tmp_path
    assert exported==[tmp_path]


def test_sprint_preserves_original_error_and_exports_partial_evidence(monkeypatch,tmp_path):
    import run_pi05_abstract_pilot as runner
    import time, threading
    p=freeze_abstract_protocol(tmp_path,checkpoint='trained.pt',batch_size=1,started_unix=time.time(),source_digest='a')
    error=ValueError('checkpoint load'); tid=threading.get_ident();exported=[]
    def fail(args):
        assert threading.get_ident()==tid
        raise error
    monkeypatch.setattr(runner,'export_evidence',lambda root: exported.append(root))
    with pytest.raises(ValueError) as caught:
        run_abstract_pilot(tmp_path,p,collect_function=fail,fit_function=lambda a:None)
    assert caught.value is error and exported==[tmp_path]
    assert 'checkpoint load' in (tmp_path/'collection.log').read_text()


def test_empty_packet_marks_comparison_not_tested_and_has_no_invented_metrics(tmp_path):
    root=tmp_path/'run';root.mkdir()
    atomic_json(root/'manifest.json',manifest(PilotConfig(tasks=(0,1,2),train_starts=1,calibration_starts=1,test_starts=1)))
    out=export_evidence(root)
    e=json.loads((out/'abstract_evidence.json').read_text())
    assert e['status']=='not_tested' and e['estimate'] is None and e['test_harmful_episodes']==0
    assert e['inventory']['planned_episodes']==39 and len(e['inventory']['missing_episodes'])==39
    assert (out/'evidence_packet.zip').exists()
    assert 'neither a representation advantage nor collision prevention' in (out/'abstract_candidate.txt').read_text()


def test_core_case_list_matches_collector_plan_exactly():
    registered=json.loads((ROOT/'configs/pi05_abstract_monitor_core.json').read_text())
    assert registered['episodes']==PilotConfig(tasks=(0,1,2),train_starts=4,calibration_starts=4,test_starts=4).plan()


def test_complete_synthetic_fit_packet_and_stale_report_rejection(tmp_path):
    from fit_pi05_occlusion_monitor import evaluate
    root=tmp_path/'run'; root.mkdir()
    config=PilotConfig(tasks=(0,),train_starts=1,calibration_starts=1,test_starts=1)
    atomic_json(root/'manifest.json',manifest(config))
    schema={f:dict(saved_width=3) for f in ('expert_input_l08','expert_output_l08','expert_transcoder_l08')}
    atomic_json(root/'matched_feature_schema.json',schema)
    rng=np.random.default_rng(42)
    for job in config.plan():
        folder=root/'episodes'/job['episode_id'];folder.mkdir(parents=True)
        state=np.zeros(5);np.save(folder/'initial_state.npy',state);atomic_json(folder/'target.json',{})
        harm=job['condition'] in ('box','static_box','screen')
        x=rng.normal(size=(6,3)).astype('float32')
        np.savez_compressed(folder/'features.npz',state=x,**{f:x.copy() for f in schema})
        ep=dict(job,success=not harm,steps=56,first_harm_step=24 if harm else None,control_freq=20,
                initial_state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
                observations=[dict(step=t,visibility={},inference_batch_seconds=.2,inference_batch_size=1) for t in range(0,48,8)],
                physics=[dict(step=t,peak_contact_n=11 if harm and t==24 else 0,max_wrong_bowl_lift_m=0,executed_action=[0]*7) for t in range(1,57)])
        atomic_json(folder/'episode.json',ep)
    evaluate(NS(run_dir=root,target='future_harm',future_steps=32,false_stop_rate=.1,C=.1,allow_partial=False))
    report,path=report_for_run(root)
    assert report and path.exists()
    assert report['models']['expert_output_l08']['fixed_10N_monitor_endpoint_sensitivity']['20']['operating']['harmful_episodes']==0
    out=export_evidence(root)
    evidence=json.loads((out/'abstract_evidence.json').read_text())
    assert evidence['status']=='inconclusive' and evidence['test_harmful_episodes']==3
    assert evidence['test_good_episodes']==2 and evidence['paired_95ci'] is None
    assert evidence['estimate']==0 and 'image_encoder,all_test,not_available' in (out/'monitor_comparison.csv').read_text()
    # One modified feature invalidates the report rather than reusing old scores.
    first=root/'episodes'/config.plan()[0]['episode_id']/'features.npz'
    with np.load(first) as f: arrays={k:f[k].copy() for k in f.files}
    arrays['state'][0,0]+=1
    np.savez_compressed(first,**arrays)
    assert report_for_run(root)==(None,None)

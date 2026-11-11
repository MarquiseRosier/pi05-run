"""No previous rollout may influence another condition's initial simulator state."""
import hashlib
import json
from pathlib import Path
import random
import sys
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'scripts')]
from run_pi05_matched_steering import SteeringRuntime, verify_trial_start, saved_trial


def test_fresh_simulator_is_seeded_before_construction_and_policy_is_reused(monkeypatch):
    builds=[]; closed=[]
    fixture=next((ROOT/'tests/fixtures/libero_spatial').glob('*.bddl'))
    def build(args,load_policy):
        assert load_policy is False
        construction_noise=(np.random.random(),random.random())
        def reset(seed):
            return {'construction_noise':construction_noise,'state_id':inner.init_state_id},{}
        inner=NS(_init_states=list(range(10)),reset=reset,
                 _env=NS(env=NS(bddl_file_name=str(fixture))))
        harness=NS(inner_env=inner,vec_env=NS(close=lambda:closed.append(harness)))
        builds.append(harness)
        return harness
    helper=ModuleType('probe_pi05_transcoder_counterfactual');helper.build_harness=build
    monkeypatch.setitem(sys.modules,helper.__name__,helper)
    runtime=object.__new__(SteeringRuntime)
    runtime.args=NS(task_id=0);runtime.harness=None;runtime.policy=object()
    policy=runtime.policy
    case=dict(task_id=4,init_state_id=0,seed=117)
    first=runtime.prepare_trial(case)
    builds[0].inner_env.init_state_id=9  # Prior rollout or wrapper reset changed it.
    np.random.random(1000);random.random()
    second=runtime.prepare_trial(case)
    assert first==second and len(builds)==2 and closed==[builds[0]]
    assert runtime.harness is builds[1] and runtime.policy is policy


def test_initial_arrays_and_pixels_are_saved_and_compared_without_rounding(tmp_path):
    state=np.array([.1,2.,3.]);images={'image':np.zeros((3,4,3),np.uint8)}
    hashes=verify_trial_start(tmp_path,state,images)
    assert (tmp_path/'initial_state.npy').exists() and (tmp_path/'initial_images.npz').exists()
    assert verify_trial_start(tmp_path,state.copy(),images)==hashes
    changed=state.copy();changed[1]=np.nextafter(changed[1],np.inf)
    with pytest.raises(RuntimeError,match='before inference'):
        verify_trial_start(tmp_path,changed,images)
    info=json.loads((tmp_path/'reset_diagnostics.json').read_text())
    assert info['different_entries']==1 and info['max_absolute_difference']>0
    assert info['time_difference']==0
    # A saved diagnostic must not be mistaken for a completed trial on retry.
    assert verify_trial_start(tmp_path,state,images)==hashes


def test_image_only_drift_is_detected_even_before_first_trial_completed(tmp_path):
    state=np.zeros(5);image=np.zeros((3,4,3),np.uint8)
    verify_trial_start(tmp_path,state,{'image':image})
    image[0,0,0]=1
    with pytest.raises(RuntimeError,match='Initial images changed'):
        verify_trial_start(tmp_path,state,{'image':image})
    assert json.loads((tmp_path/'reset_diagnostics.json').read_text())['image_differences']=={'image':1}


def test_saved_trial_rejects_drift_before_policy_work_or_completion_marker(tmp_path):
    case=dict(case_id='task4_init0',split='train')
    folder=tmp_path/'cases'/case['case_id'];folder.mkdir(parents=True)
    images={'image':np.zeros((3,4,3),np.uint8)}
    verify_trial_start(folder,np.zeros(4),images)
    inference=[]
    class Runtime:
        def trial(self,*args,start_validator,**kwargs):
            start_validator(np.ones(4),images)
            inference.append(True)
    with pytest.raises(RuntimeError,match='Reset drift'):
        saved_trial(Runtime(),tmp_path,case,1,'native_1')
    assert inference==[] and not (folder/'native_1.json').exists()


def test_old_completed_hash_still_must_match(tmp_path):
    state=np.zeros(4);images={'image':np.zeros((3,4,3),np.uint8)}
    (tmp_path/'native_-1.json').write_text(json.dumps(dict(initial_state_sha256='old-state',
        initial_images_sha256=hashlib.sha256(images['image'].tobytes()).hexdigest())))
    with pytest.raises(RuntimeError,match='Reset drift'):
        verify_trial_start(tmp_path,state,images)
    assert not (tmp_path/'initial_state.npy').exists()

import ast
import builtins
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from embed_pi05_steering_notebook import definition_cells, notebook


def test_notebook_matches_generator_and_has_new_branch_and_direct_launch():
    saved=json.loads((ROOT/'notebooks/pi05_matched_steering.ipynb').read_text())
    assert saved==notebook()
    for cell in saved['cells']:
        if cell['cell_type']=='code': ast.parse(''.join(cell['source']))
    controls=''.join(next(c['source'] for c in saved['cells'] if c.get('id')=='steering-controls'))
    assert 'feature/marquise-matched-steering' in controls
    launch=''.join(next(c['source'] for c in saved['cells'] if c.get('id')=='steering-launch'))
    assert 'report = run_steering_experiment(' in launch
    assert 'subprocess' not in launch and 'runpy' not in launch


def test_inline_definitions_run_without_importing_repo_experiment_or_workers(monkeypatch):
    import torch
    module=ModuleType('_steering_kernel');monkeypatch.setitem(sys.modules,module.__name__,module)
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        assert name not in {'pi05_mi.matched_steering','pi05_mi.occlusion_pilot',
                            'run_pi05_matched_steering','run_pi05_occlusion_pilot','subprocess'}
        return original(name,*args,**kwargs)
    for cell in definition_cells():
        with monkeypatch.context() as context:
            context.setattr(builtins,'__import__',guarded)
            exec(compile(''.join(cell['source']),'<notebook '+cell['id']+'>','exec'),module.__dict__)
    config=module.SteeringConfig()
    assert len(config.cases())==71
    fixture=next((ROOT/'tests/fixtures/libero_spatial').glob('*.bddl'))
    prompts,_,_=module.contrast_prompts(NS(_env=NS(env=NS(bddl_file_name=str(fixture)))))
    assert prompts['1']!=prompts['-1']
    assert module.run_steering_experiment.__code__.co_filename.startswith('<notebook ')
    delta,_=module.intervention_delta(torch.ones(1,2,2),None,None,
        {'direction':[1.,0.],'random_direction':[0.,1.],'center':[0.,0.]},
        family='dense',mode='zero',sign=1,budget=0.)
    assert np.count_nonzero(delta.numpy())==0

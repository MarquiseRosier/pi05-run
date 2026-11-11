"""Notebook definitions execute independently of the repository's pilot modules."""
import builtins
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from embed_pi05_monitor_notebook import embedded_cells, SOURCES


def notebook_cells():
    return json.loads((ROOT / 'notebooks/pi05_occlusion_monitor_pilot.ipynb').read_text())['cells']


def test_embedded_cells_match_maintained_implementations():
    ids = {entry[0] for entry in SOURCES}
    assert [c for c in notebook_cells() if c.get('id') in ids] == embedded_cells()


@pytest.fixture
def kernel(monkeypatch):
    module = ModuleType('_test_monitor_notebook')
    monkeypatch.setitem(sys.modules, module.__name__, module)
    module.LOCAL_REPO = ROOT
    original_import = builtins.__import__
    forbidden = {'pi05_mi.occlusion_pilot', 'run_pi05_occlusion_pilot',
                 'run_pi05_monitor_budget', 'fit_pi05_occlusion_monitor', 'subprocess'}
    def guarded_import(name, *args, **kwargs):
        if name in forbidden:
            raise AssertionError(f'Notebook must not delegate to {name}')
        return original_import(name, *args, **kwargs)
    for cell in embedded_cells():
        with monkeypatch.context() as context:
            # Load common third-party dependencies before guarding pilot imports.
            import numpy  # noqa: F401
            import concurrent.futures  # noqa: F401
            context.setattr(builtins, '__import__', guarded_import)
            exec(compile(''.join(cell['source']), f"<notebook cell {cell['id']}>", 'exec'), module.__dict__)
    return module


def test_inline_plan_and_changed_batch_preserve_previous_manifest(kernel, tmp_path):
    args = ['--plan-only', '--preset', 'smoke', '--output-dir', str(tmp_path)]
    kernel.collect_occlusion_pilot(args + ['--batch-size', '4'])
    path = tmp_path / 'manifest.json'
    original = path.read_bytes()
    assert json.loads(original)['notebook_pilot_version'] == kernel.NOTEBOOK_PILOT_VERSION
    # Identical resume works; changed batching reports the exact setting.
    kernel.collect_occlusion_pilot(args + ['--batch-size', '4'])
    with pytest.raises(ValueError, match=r'config.batch_size: 4 -> 1.*new RUN_NAME'):
        kernel.collect_occlusion_pilot(args + ['--batch-size', '1'])
    assert path.read_bytes() == original
    assert kernel.collect_occlusion_pilot.__code__.co_filename.startswith('<notebook cell')


def test_runner_calls_notebook_collector_and_fitter(kernel, tmp_path):
    calls = []
    kernel.collect_occlusion_pilot = lambda args: calls.append(('collect', args))
    kernel.fit_occlusion_monitor = lambda args: calls.append(('fit', args))
    kernel.analysis_readiness = lambda root: (True, None)
    kernel.run_monitor(['--output-dir', str(tmp_path), '--minutes', '3', '--batch-size', '1'])
    assert [name for name, _ in calls] == ['collect', 'fit']
    assert calls[0][1][calls[0][1].index('--batch-size') + 1] == '1'
    assert kernel.run_monitor.__code__.co_filename.startswith('<notebook cell')

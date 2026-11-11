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


def test_inline_target_resolver_uses_active_domain(kernel):
    from types import SimpleNamespace
    paths = sorted((ROOT / 'tests/fixtures/libero_spatial').glob('*.bddl'))
    assert len(paths) == 10
    for path in paths:
        inner = SimpleNamespace(_env=SimpleNamespace(env=SimpleNamespace(bddl_file_name=str(path))))
        assert kernel.target_bowl_spec(inner)['target_instance'] == 'akita_black_bowl_1'
    assert kernel.target_bowl_spec.__code__.co_filename.startswith('<notebook cell')


def test_inline_segmentation_decoder_uses_wide_arithmetic(kernel):
    import numpy as np
    from types import SimpleNamespace as NS
    scene = NS(ngeom=256, geoms=[NS(segid=-1)] * 256)
    scene.geoms[0] = NS(segid=255, objtype=5, objid=23)
    decoded = kernel.decode_segmentation_rgb(np.array([[[0, 1, 0]]], np.uint8), scene)
    assert decoded.tolist() == [[[5, 23]]]
    assert kernel.install_segmentation_reader.__code__.co_filename.startswith('<notebook cell')


def test_inline_infeasible_miss_does_not_abort_or_claim_matched_control(kernel):
    import numpy as np
    frame = np.zeros((32, 32, 3), np.uint8)
    output, info = kernel.ImageOccluder('miss_box').apply('image', frame, np.ones((32, 32), bool))
    np.testing.assert_array_equal(output, frame)
    assert info['placement_status'] == 'no_same_size_miss'
    assert not kernel.episode_control_valid({'observations': [{'visibility': {'image': info}}]})


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

"""Scientific validity checks; GPU integration remains a separate acceptance gate."""
import json
import ast
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'src'),
                str(Path(__file__).resolve().parents[1] / 'scripts')]
from pi05_mi.occlusion_pilot import (PilotConfig, ImageOccluder, assert_split_isolation,
                                    labels_at_steps, stable_indices)
from fit_pi05_occlusion_monitor import threshold_from_good_episodes, operating_metrics


def test_notebook_token_loader_uses_environment_without_shared_file(monkeypatch):
    notebook = json.loads((Path(__file__).resolve().parents[1] /
                          'notebooks/pi05_occlusion_monitor_pilot.ipynb').read_text())
    code = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
    for source in code:
        ast.parse(source)
    assert 'SHARED_HF_TOKEN_FILE' not in '\n'.join(code)
    assert 'DRIVE_HF_TOKEN_FILE' not in '\n'.join(code)
    cache = ast.parse(next(s for s in code if s.startswith('# @title Prepare')))
    loader = next(n for n in cache.body if isinstance(n, ast.FunctionDef) and n.name == 'get_hf_token')
    import os
    from types import ModuleType, SimpleNamespace
    scope = {'os': os}
    exec(compile(ast.Module(body=[loader], type_ignores=[]), '<token loader>', 'exec'), scope)
    monkeypatch.setenv('HF_TOKEN', 'test-environment-value')
    assert scope['get_hf_token'](required=True) == 'test-environment-value'
    monkeypatch.delenv('HF_TOKEN')
    google = ModuleType('google'); colab = ModuleType('google.colab')
    colab.userdata = SimpleNamespace(get=lambda name: 'test-colab-value')
    google.colab = colab
    monkeypatch.setitem(sys.modules, 'google', google)
    monkeypatch.setitem(sys.modules, 'google.colab', colab)
    assert scope['get_hf_token'](required=True) == 'test-colab-value'
    assert os.environ['HF_TOKEN'] == 'test-colab-value'
    monkeypatch.delenv('HF_TOKEN')
    colab.userdata = SimpleNamespace(get=lambda name: '')
    assert scope['get_hf_token']() == ''
    with pytest.raises(RuntimeError, match='HF_TOKEN environment variable'):
        scope['get_hf_token'](required=True)


def test_plans_do_not_leak_pairs_or_screen():
    config = PilotConfig(tasks=tuple(range(10)), train_starts=14, calibration_starts=5, test_starts=5)
    plan = config.plan()
    assert len(plan) == 1010
    assert len({x['group_id'] for x in plan}) == 240
    assert_split_isolation(plan)
    assert sum(x['split'] == 'train' for x in plan) == 560
    assert sum(x['split'] == 'calibration' for x in plan) == 200
    assert sum(x['split'] == 'test' for x in plan) == 250
    broken = [dict(plan[0]), dict(plan[0], split='test')]
    with pytest.raises(ValueError):
        assert_split_isolation(broken)


def test_boxes_and_screen_only_change_intended_pixels():
    image = np.full((64, 64, 3), 20, np.uint8)
    mask = np.zeros((64, 64), bool); mask[20:30, 20:30] = True
    moving, miss, screen = ImageOccluder('box'), ImageOccluder('miss_box'), ImageOccluder('screen')
    hidden, a = moving.apply('image', image, mask)
    missed, b = miss.apply('image', image, mask)
    wrist, c = screen.apply('image2', image, mask)
    assert a['target_pixels_after'] == 0
    assert b['target_pixels_after'] == 100
    assert a['overlay_pixels'] == b['overlay_pixels']
    assert np.array_equal(wrist, image)
    assert not np.array_equal(missed, image)
    assert np.array_equal(image, np.full_like(image, 20))  # Original frame untouched.


def test_static_box_stays_put_while_moving_box_tracks():
    frame = np.zeros((80, 80, 3), np.uint8)
    old = np.zeros((80, 80), bool); old[10:20, 10:20] = True
    new = np.zeros_like(old); new[50:60, 50:60] = True
    static, moving = ImageOccluder('static_box'), ImageOccluder('box')
    static.apply('image', frame, old); moving.apply('image', frame, old)
    assert static.apply('image', frame, new)[1]['target_pixels_after'] == 100
    assert moving.apply('image', frame, new)[1]['target_pixels_after'] == 0


def test_absent_target_is_unknown_not_claimed_visible():
    frame = np.zeros((40, 40, 3), np.uint8)
    _, info = ImageOccluder('box').apply('image', frame, np.zeros((40, 40), bool))
    assert info['fraction_remaining'] is None
    assert info['rectangle'] is None


def test_no_post_harm_or_censored_negative_labels():
    ep = dict(success=False, steps=80, first_harm_step=30)
    y, keep = labels_at_steps([0, 16, 32, 72], ep, target='future_harm', future_steps=16)
    assert y.tolist() == [0, 1, 0, 0]
    assert keep.tolist() == [True, True, False, False]
    ep['first_harm_step'] = None
    y, keep = labels_at_steps([0, 64, 72], ep, target='future_harm', future_steps=16)
    assert keep.tolist() == [True, True, False]


def test_calibration_controls_episode_false_stops_with_ties():
    scores = [.1] * 5 + [.8] * 4 + [.9]
    t = threshold_from_good_episodes(scores, .1)
    assert sum(x >= t for x in scores) <= 1
    with pytest.raises(ValueError):
        threshold_from_good_episodes([], .1)


def test_delay_can_remove_early_warning():
    ep = dict(episode_id='a', success=False, steps=40, first_harm_step=18,
              sample_steps=np.array([0, 8, 16, 24]))
    scores = {'a': np.array([.1, .1, .9, .9])}
    assert operating_metrics([ep], scores, .5)['early_warnings'] == 1
    assert operating_metrics([ep], scores, .5, delay=2)['early_warnings'] == 0


def test_feature_selection_does_not_depend_on_data():
    assert np.array_equal(stable_indices(1000, 32, 'layer0'), stable_indices(1000, 32, 'layer0'))
    assert len(set(stable_indices(1000, 32, 'layer0'))) == 32


def test_fitter_runs_end_to_end_and_resumes_without_refitting(tmp_path):
    from types import SimpleNamespace
    import joblib
    from pi05_mi.occlusion_pilot import manifest, write_manifest
    from fit_pi05_occlusion_monitor import evaluate
    config = PilotConfig(tasks=(0,), train_starts=1, calibration_starts=1, test_starts=1)
    write_manifest(tmp_path, manifest(config))
    rng = np.random.default_rng(6)
    train = []
    for job in config.plan():
        folder = tmp_path / 'episodes' / job['episode_id']; folder.mkdir(parents=True)
        harm = job['condition'] in ('box', 'static_box', 'screen')
        # Synthetic only: exercises serialization and split isolation, not scientific effectiveness.
        array = rng.normal(size=(6, 3)).astype('float32')
        if job['split'] == 'test':
            array += 1000  # Deliberate test-only shift must not enter the fitted scaler.
        if job['split'] == 'train':
            train.append(array[:3] if harm else array[:4])
        np.savez_compressed(folder / 'features.npz', state=array,
                            expert_input_l00=array * 2, expert_output_l00=array * 3,
                            expert_transcoder_l00=array * 4)
        ep = {**job, 'success': not harm, 'steps': 56, 'first_harm_step': 24 if harm else None,
              'initial_state_sha256': job['group_id'], 'observations': [{'step': t} for t in range(0, 48, 8)]}
        (folder / 'episode.json').write_text(json.dumps(ep))
    args = SimpleNamespace(run_dir=tmp_path, target='future_harm', future_steps=32,
                           false_stop_rate=.1, C=.1, allow_partial=False)
    evaluate(args)
    pointer = json.loads((tmp_path / 'latest_future_harm.json').read_text())
    model_file = Path(pointer['report']).parent / 'state.joblib'
    saved = joblib.load(model_file)
    report = json.loads(Path(pointer['data']).read_text())
    comparison = report['matched_comparisons']['expert_transcoder_l00_minus_expert_output_l00']
    assert comparison['seen_occluders']['harm_preceded_fraction']['difference'] == 0
    assert comparison['seen_occluders']['harm_preceded_fraction']['paired_group_bootstrap_95ci'] is None
    assert np.max(np.abs(saved['pipeline']['standardscaler'].mean_)) < 10
    mtime = model_file.stat().st_mtime_ns
    evaluate(args)
    assert model_file.stat().st_mtime_ns == mtime
    state = json.loads((tmp_path / 'run_state.json').read_text())
    assert state['phase'] == 'report' and state['status'] == 'complete'


def test_partial_loader_excludes_incomplete_paired_groups(tmp_path):
    from pi05_mi.occlusion_pilot import manifest, write_manifest
    from fit_pi05_occlusion_monitor import load_episodes
    config = PilotConfig(tasks=(0,), train_starts=1, calibration_starts=1, test_starts=1)
    write_manifest(tmp_path, manifest(config))
    job = config.plan()[0]
    folder = tmp_path / 'episodes' / job['episode_id']; folder.mkdir(parents=True)
    np.savez(folder / 'features.npz', state=np.zeros((1, 3)))
    (folder / 'episode.json').write_text(json.dumps({**job, 'observations': [{'step': 0}],
                                                   'initial_state_sha256': 'same'}))
    assert load_episodes(tmp_path, allow_partial=True) == []


def test_paired_monitor_difference_preserves_direction_and_pairing():
    from fit_pi05_occlusion_monitor import paired_monitor_difference
    episodes = [dict(episode_id=str(i), group_id=str(i), success=False, steps=40,
                     first_harm_step=18, sample_steps=np.array([0, 8, 16])) for i in range(4)]
    early = {str(i): np.array([.1, .8, .9]) for i in range(4)}
    missed = {str(i): np.zeros(3) for i in range(4)}
    result = paired_monitor_difference(episodes, early, .5, missed, .5, draws=30)
    assert result['harm_preceded_fraction']['difference'] == 1
    assert result['harm_preceded_fraction']['paired_group_bootstrap_95ci'] == [1, 1]
    assert result['false_stop_fraction']['difference'] is None


def test_analysis_requires_completed_pairs_in_each_split(tmp_path):
    from run_pi05_monitor_budget import analysis_readiness
    from pi05_mi.occlusion_pilot import manifest, write_manifest
    assert not analysis_readiness(tmp_path)[0]
    config = PilotConfig(tasks=(0,), train_starts=1, calibration_starts=1, test_starts=1)
    write_manifest(tmp_path, manifest(config))
    assert not analysis_readiness(tmp_path)[0]
    jobs = config.plan()
    for job in jobs[:-1]:
        folder = tmp_path / 'episodes' / job['episode_id']; folder.mkdir(parents=True)
        (folder / 'episode.json').write_text('{}')
        np.savez(folder / 'features.npz', state=np.zeros((1, 3)))
    ready, reason = analysis_readiness(tmp_path)
    assert not ready and 'test' in reason
    folder = tmp_path / 'episodes' / jobs[-1]['episode_id']; folder.mkdir(parents=True)
    (folder / 'episode.json').write_text('{}')
    assert not analysis_readiness(tmp_path)[0]  # Completed marker alone is insufficient.
    np.savez(folder / 'features.npz', state=np.zeros((1, 3)))
    assert analysis_readiness(tmp_path) == (True, None)


def test_failed_collection_skips_fit_preserves_error_and_rejects_stale_report(tmp_path, monkeypatch):
    import run_pi05_monitor_budget as budget
    calls = []
    (tmp_path / 'old.md').write_text('STALE SUCCESS MUST NOT APPEAR')
    (tmp_path / 'latest_future_harm.json').write_text(json.dumps({'report': str(tmp_path / 'old.md')}))
    def failed_collection(command, log, seconds):
        calls.append(command)
        budget.RunJournal(tmp_path).update('collection', 'failed', error='ImportError: wrong LeRobot')
        return dict(status='failed', exit_code=1)
    monkeypatch.setattr(budget, 'run_phase', failed_collection)
    monkeypatch.setattr(sys, 'argv', ['budget', '--output-dir', str(tmp_path), '--minutes', '3'])
    with pytest.raises(SystemExit) as error:
        budget.main()
    assert error.value.code == 1
    assert len(calls) == 1
    result = json.loads((tmp_path / 'budget_summary.json').read_text())
    assert result['analysis']['status'] == 'skipped'
    text = (tmp_path / 'budget_summary.md').read_text()
    assert 'ImportError: wrong LeRobot' in text
    assert 'STALE SUCCESS' not in text
    assert json.loads((tmp_path / 'run_state.json').read_text())['status'] == 'partial'


def test_runtime_guard_explains_pin_before_importing_new_api(monkeypatch):
    import importlib.metadata
    from pi05_monitor_runtime import validate_runtime
    monkeypatch.setattr(importlib.metadata, 'version', lambda _: '0.7.0')
    with pytest.raises(RuntimeError, match='requires lerobot==0.4.4'):
        validate_runtime()


def test_notebook_runtime_pin_and_preservation():
    from pi05_monitor_runtime import REQUIREMENT
    notebook = json.loads((Path(__file__).resolve().parents[1] /
                          'notebooks/pi05_occlusion_monitor_pilot.ipynb').read_text())
    code = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
    for source in code:
        ast.parse(source)
    install = next(s for s in code if s.startswith('# @title Install Pinned'))
    assert REQUIREMENT in install
    assert '"--clear"' not in install
    assert "version('lerobot') == '0.4.4'" in install
    clone = next(s for s in code if s.startswith('# @title Clone'))
    assert 'from pi05_monitor_runtime import validate_runtime' in clone


@pytest.mark.parametrize('mode', ['missing', 'empty', 'driver_error', 'fallback', 'l4', 't4'])
def test_notebook_gpu_check_handles_unavailable_tools(monkeypatch, mode):
    import shutil
    from types import SimpleNamespace
    notebook = json.loads((Path(__file__).resolve().parents[1] /
                          'notebooks/pi05_occlusion_monitor_pilot.ipynb').read_text())
    source = next(''.join(c['source']) for c in notebook['cells']
                  if ''.join(c['source']).startswith('# @title Mount Drive'))
    function = next(n for n in ast.parse(source).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'validate_colab_runtime')
    monkeypatch.setattr(shutil, 'which', lambda _: '/mock/nvidia-smi' if mode in ('empty', 'driver_error') else None)
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        outputs = {'fallback': 'NVIDIA A100, 580, 81920\n', 'l4': 'NVIDIA L4, 580, 23034\n',
                   't4': 'NVIDIA T4, 580, 15360\n'}
        return SimpleNamespace(returncode=1 if mode == 'driver_error' else 0, stderr='',
                               stdout=outputs.get(mode, ''))
    scope = dict(Path=lambda _: SimpleNamespace(is_file=lambda: mode in ('fallback', 'l4', 't4')),
                 os=SimpleNamespace(access=lambda *_: True, X_OK=1),
                 subprocess=SimpleNamespace(run=run), REQUIRED_GPU='Auto',
                 MIN_GPU_MEMORY_GB=22, MIN_SYSTEM_RAM_GB=24, parse_gib_from_meminfo=lambda: 168)
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<gpu check>', 'exec'), scope)
    if mode in ('fallback', 'l4'):
        profile = scope['validate_colab_runtime']()
        assert calls[0][0] == '/usr/bin/nvidia-smi'
        assert profile['batch_size'] == (1 if mode == 'l4' else 4)
    elif mode == 't4':
        with pytest.raises(RuntimeError, match='Auto mode supports'):
            scope['validate_colab_runtime']()
    else:
        with pytest.raises(RuntimeError, match='Change runtime type'):
            scope['validate_colab_runtime']()
        if mode == 'missing':
            assert not calls

"""The monitor must expose output and original errors in its caller's thread."""
import json
import os
from pathlib import Path
import signal
import sys
import threading

import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'scripts'),
                str(Path(__file__).resolve().parents[1] / 'src')]
from run_pi05_monitor_budget import run_phase


def test_phase_is_live_same_process_and_thread_and_rotates_history(tmp_path, capsys):
    log = tmp_path / 'collection.log'
    log.write_text('OLD FAILURE\n')
    caller = (os.getpid(), threading.get_ident())
    def work():
        assert (os.getpid(), threading.get_ident()) == caller
        print('live stdout')
        print('live stderr', file=sys.stderr)
    assert run_phase(work, log, 10)['status'] == 'complete'
    live = capsys.readouterr()
    assert 'live stdout' in live.out and 'live stderr' in live.err
    assert 'live stdout' in log.read_text() and 'live stderr' in log.read_text()
    assert 'OLD FAILURE' not in log.read_text()
    assert next(tmp_path.glob('collection.*.log')).read_text() == 'OLD FAILURE\n'


def test_original_exception_and_streams_survive_phase_failure(tmp_path):
    before = (sys.stdout, sys.stderr)
    error = ValueError('missing embedding')
    def failing_model():
        raise error
    with pytest.raises(ValueError) as raised:
        run_phase(failing_model, tmp_path / 'collection.log', 10)
    assert raised.value is error
    assert (sys.stdout, sys.stderr) == before
    assert 'failing_model' in (tmp_path / 'collection.log').read_text()
    state = json.loads((tmp_path / 'run_state.json').read_text())
    assert state['status'] == 'failed' and state['error'] == 'ValueError: missing embedding'


def test_collection_entrypoint_keeps_notebook_signals_and_resets_stop(tmp_path):
    import run_pi05_occlusion_pilot as collector
    before = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    collector.STOP_REQUESTED = True
    collector.main(['--plan-only', '--preset', 'smoke', '--output-dir', str(tmp_path)])
    assert not collector.STOP_REQUESTED
    assert {sig: signal.getsignal(sig) for sig in before} == before


def test_expired_phase_never_starts_work(tmp_path):
    def forbidden():
        raise AssertionError('must not run')
    assert run_phase(forbidden, tmp_path / 'analysis.log', 0)['status'] == 'budget_exhausted'


def test_retained_library_stream_still_works_after_phase(tmp_path, capsys):
    streams = []
    run_phase(lambda: streams.append(sys.stderr), tmp_path / 'collection.log', 10)
    streams[0].write('later library warning\n')
    assert 'later library warning' in capsys.readouterr().err


@pytest.mark.parametrize('cleanup_fails', [False, True])
def test_model_failure_closes_simulator_and_preserves_error(monkeypatch, cleanup_fails):
    from types import SimpleNamespace
    from unittest.mock import Mock
    import probe_pi05_transcoder_counterfactual as probe
    failure = ValueError('checkpoint mismatch')
    close = Mock(side_effect=RuntimeError('EGL cleanup') if cleanup_fails else None)
    env = SimpleNamespace(envs=[object()], close=close)
    fake_modules = {
        'lerobot.configs.policies': SimpleNamespace(
            PreTrainedConfig=SimpleNamespace(from_pretrained=lambda *a, **kw: SimpleNamespace())),
        'lerobot.envs.configs': SimpleNamespace(LiberoEnv=lambda **kw: None),
        'lerobot.envs.factory': SimpleNamespace(make_env=lambda *a, **kw: {'libero_spatial': {0: env}},
                                              make_env_pre_post_processors=lambda **kw: (None, None)),
        'lerobot.policies.factory': SimpleNamespace(make_policy=Mock(side_effect=failure),
                                                  make_pre_post_processors=lambda **kw: (None, None)),
        'lerobot.scripts.lerobot_eval': SimpleNamespace(preprocess_observation=lambda x: x),
    }
    for name, module in fake_modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(probe, 'ensure_libero_config', lambda: None)
    monkeypatch.setattr(probe, 'install_load_recorder', lambda: None)
    args = SimpleNamespace(suite='libero_spatial', task_id=0, prompt='',
                           policy_path='checkpoint', device='cpu', policy_dtype='float32')
    with pytest.raises(ValueError) as raised:
        probe.build_harness(args)
    assert raised.value is failure
    close.assert_called_once()

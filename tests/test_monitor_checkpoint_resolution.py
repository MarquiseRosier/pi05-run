from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from resolve_pi05_monitor_checkpoint import SHARED_CHECKPOINT, resolve_checkpoint


def make_file(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'test fixture, not real weights')
    return path


def test_shared_default_and_explicit_override(tmp_path):
    drive, repo = tmp_path / 'drive', tmp_path / 'repo'
    shared = make_file(drive / SHARED_CHECKPOINT)
    other = make_file(drive / 'other.pt')
    assert resolve_checkpoint('', drive, repo) == shared
    assert resolve_checkpoint('other.pt', drive, repo) == other
    with pytest.raises(FileNotFoundError, match='Explicit'):
        resolve_checkpoint('typo.pt', drive, repo)  # Never silently substitutes a requested file.


def test_unique_fallback_and_ambiguity(tmp_path):
    drive, repo = tmp_path / 'drive', tmp_path / 'repo'
    with pytest.raises(FileNotFoundError, match='Weights live in Drive'):
        resolve_checkpoint('', drive, repo)
    first = make_file(drive / 'outputs/transcoders/moved' / SHARED_CHECKPOINT.name)
    assert resolve_checkpoint('', drive, repo) == first.resolve()
    make_file(repo / 'checkpoints/another' / SHARED_CHECKPOINT.name)
    with pytest.raises(ValueError, match='Multiple copies'):
        resolve_checkpoint('', drive, repo)

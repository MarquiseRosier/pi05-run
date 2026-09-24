#!/usr/bin/env python
"""Smoke test: a policy whose weights did not load must abort, not run.

LeRobot's from_pretrained prints a warning and returns a random model when the
state dict fails to load, and strict=False lets a partial checkpoint through.
These checks lock the guard that turns both into a hard error.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pi05_mi.pi05_weights import LOAD_REPORT_ATTR, assert_weights_loaded, install_load_recorder, load_report  # noqa: E402


class _Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = torch.nn.Linear(2, 2)
        self.expert = torch.nn.Linear(2, 2)


def _raises(fn, needle: str) -> None:
    try:
        fn()
    except RuntimeError as exc:
        assert needle in str(exc), f"expected {needle!r} in error, got: {exc}"
        return
    raise AssertionError(f"expected a RuntimeError mentioning {needle!r}")


def test_never_loaded_is_refused() -> None:
    install_load_recorder(_Policy)
    policy = _Policy()
    assert load_report(policy) is None
    _raises(lambda: assert_weights_loaded(policy, source="ckpt"), "never loaded")


def test_partial_load_is_refused_and_names_the_missing_keys() -> None:
    install_load_recorder(_Policy)
    policy = _Policy()
    full = _Policy().state_dict()
    partial = {k: v for k, v in full.items() if not k.startswith("vision.")}
    policy.load_state_dict(partial, strict=False)
    report = load_report(policy)
    assert sorted(report["missing"]) == ["vision.bias", "vision.weight"]
    _raises(lambda: assert_weights_loaded(policy), "vision.weight")


def test_complete_load_passes_and_tolerates_unexpected_keys() -> None:
    install_load_recorder(_Policy)
    policy = _Policy()
    state = _Policy().state_dict()
    state["stale.extra"] = torch.zeros(1)
    policy.load_state_dict(state, strict=False)
    report = assert_weights_loaded(policy)
    assert report["missing"] == [] and report["unexpected"] == ["stale.extra"]
    assert report["provided"] == len(state)


def test_recorder_is_idempotent_and_preserves_strict_behaviour() -> None:
    install_load_recorder(_Policy)
    first = _Policy.load_state_dict
    install_load_recorder(_Policy)
    assert _Policy.load_state_dict is first, "installing twice must not stack wrappers"
    policy = _Policy()
    try:
        policy.load_state_dict({"expert.weight": torch.zeros(2, 2)}, strict=True)
    except RuntimeError:
        pass  # torch's own strict error still propagates
    else:
        raise AssertionError("strict=True must still raise on a partial state dict")


def test_recorder_targets_lerobot_pi05_by_default() -> None:
    try:
        from lerobot.policies.pi05 import modeling_pi05
    except Exception as exc:  # pragma: no cover - environment without lerobot
        print(f"  (skipped: lerobot not importable: {type(exc).__name__})")
        return
    install_load_recorder()
    assert getattr(modeling_pi05.PI05Policy.load_state_dict, "_pi05_mi_recorder", False)


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok {test.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    main()

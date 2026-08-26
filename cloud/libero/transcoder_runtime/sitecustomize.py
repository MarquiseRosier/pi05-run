"""Opt-in Pi0.5 transcoder runtime bootstrap for LeRobot runs."""

from __future__ import annotations

import os
import sys


def _enabled() -> bool:
    mode = os.environ.get("PI05_TRANSCODER_MODE", "original").strip().lower()
    return mode in {"probe", "replace"}


if _enabled():
    try:
        import pi05_transcoder_runtime  # noqa: F401
    except Exception as exc:  # pragma: no cover - startup guard
        print(f"[pi05_transcoder] failed to start: {exc}", file=sys.stderr)


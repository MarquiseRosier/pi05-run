"""Opt-in activation-capture bootstrap for LeRobot runs."""

from __future__ import annotations

import os
import sys


def _enabled() -> bool:
    value = os.environ.get("LEROBOT_CAPTURE_ACTIVATIONS") or os.environ.get("CAPTURE_ACTIVATIONS")
    return str(value).lower() in {"1", "true", "yes", "on"}


if _enabled():
    try:
        import lerobot_activation_capture  # noqa: F401
    except Exception as exc:  # pragma: no cover - startup guard
        print(f"[activation_capture] failed to start: {exc}", file=sys.stderr)


def _transcoder_enabled() -> bool:
    mode = os.environ.get("PI05_TRANSCODER_MODE", "original").strip().lower()
    return mode in {"probe", "replace"}


if _transcoder_enabled():
    try:
        import pi05_transcoder_runtime  # noqa: F401
    except Exception as exc:  # pragma: no cover - startup guard
        print(f"[pi05_transcoder] failed to start: {exc}", file=sys.stderr)

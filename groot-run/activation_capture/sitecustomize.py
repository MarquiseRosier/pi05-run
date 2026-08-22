"""Opt-in activation-capture bootstrap for GR00T server runs."""

from __future__ import annotations

import os
import sys


def _enabled() -> bool:
    value = os.environ.get("GROOT_CAPTURE_ACTIVATIONS") or os.environ.get("CAPTURE_ACTIVATIONS")
    return str(value).lower() in {"1", "true", "yes", "on"}


if _enabled():
    try:
        import groot_activation_capture  # noqa: F401
    except Exception as exc:  # pragma: no cover - startup guard
        print(f"[activation_capture] failed to start: {exc}", file=sys.stderr)

"""Refuse to run on a Pi0.5 whose pretrained weights did not actually load.

LeRobot's ``PI05Policy.from_pretrained`` wraps the whole state-dict load in a
``try``/``except`` that prints ``Warning: Could not load state dict`` and
returns the freshly initialised model anyway; and with ``strict=False`` a
partially matching checkpoint loads silently with missing keys. Either way the
policy runs on random weights and every downstream number is meaningless, while
nothing in the process exits non-zero.

The fix is to record what ``load_state_dict`` reported and check it after the
policy is built. A policy with no record was never loaded; a policy with
missing keys is not the checkpoint.
"""

from __future__ import annotations

import copy
from typing import Any

LOAD_REPORT_ATTR = "_pi05_mi_load_report"


def restore_tied_parameter_aliases(model, state_dict):
    """Restore omitted safetensors keys only for parameters the model actually shares.

    PaliGemma ties its token embedding to lm_head. Safetensors may save only
    one name; nn.Module.load_state_dict(strict=True) still requires both.
    Never invent a tensor or infer a tie from matching names or shapes.
    """
    groups = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        groups.setdefault(id(parameter), []).append(name)
    restored = {}
    result = state_dict
    for names in groups.values():
        present = [name for name in names if name in state_dict]
        if not present:
            continue
        for name in names:
            if name not in state_dict:
                if result is state_dict:
                    result = copy.copy(state_dict)
                result[name] = state_dict[present[0]]
                restored[name] = present[0]
    return result, restored


def install_load_recorder(cls: type | None = None) -> None:
    """Make ``cls.load_state_dict`` leave a record of missing and unexpected keys.

    Defaults to LeRobot's ``PI05Policy``. Idempotent.
    """
    if cls is None:
        from lerobot.policies.pi05 import modeling_pi05

        cls = modeling_pi05.PI05Policy
    original = cls.load_state_dict
    if getattr(original, "_pi05_mi_recorder", False):
        return

    def load_state_dict_recording(self, state_dict, strict: bool = True, assign: bool = False):
        state_dict, restored = restore_tied_parameter_aliases(self, state_dict)
        # Clear any previous successful report before a later load can fail.
        setattr(self, LOAD_REPORT_ATTR, None)
        try:
            result = original(self, state_dict, strict=strict, assign=assign)
        except Exception as exc:
            setattr(self, LOAD_REPORT_ATTR, {'error': f'{type(exc).__name__}: {exc}'})
            raise
        setattr(
            self,
            LOAD_REPORT_ATTR,
            {
                "missing": list(getattr(result, "missing_keys", [])),
                "unexpected": list(getattr(result, "unexpected_keys", [])),
                "provided": len(state_dict),
                "strict": bool(strict),
                "restored_tied_aliases": restored,
            },
        )
        return result

    load_state_dict_recording._pi05_mi_recorder = True  # type: ignore[attr-defined]
    cls.load_state_dict = load_state_dict_recording


def load_report(policy: Any) -> dict[str, Any] | None:
    return getattr(policy, LOAD_REPORT_ATTR, None)


def assert_weights_loaded(policy: Any, *, source: str = "", max_listed: int = 5) -> dict[str, Any]:
    """Raise unless the policy's pretrained weights loaded completely.

    Unexpected keys (present in the checkpoint, absent from the model) are
    tolerated and returned in the report; missing keys are not, because a
    missing parameter is one running on its random initialisation.
    """
    report = load_report(policy)
    where = f" from {source}" if source else ""
    if report is None:
        raise RuntimeError(
            f"Pi0.5 weights were never loaded{where}. LeRobot swallowed the failure and returned a "
            "randomly initialised model; look for 'Could not load state dict' in the log above. "
            "Refusing to run: every measurement would be about random weights."
        )
    if report.get('error'):
        raise RuntimeError(
            f"Pi0.5 checkpoint loading failed{where}:\n{report['error']}\n"
            "Refusing to run with incomplete weights."
        )
    missing = report["missing"]
    if missing:
        shown = "\n  ".join(missing[:max_listed])
        more = f"\n  ... and {len(missing) - max_listed} more" if len(missing) > max_listed else ""
        raise RuntimeError(
            f"{len(missing)} Pi0.5 parameters did not load{where} and are running on random "
            f"initialisation:\n  {shown}{more}\n"
            "A vision-tower key layout mismatch is the usual cause. Refusing to run."
        )
    return report

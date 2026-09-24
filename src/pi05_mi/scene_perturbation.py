"""Targeted visual perturbation of a loaded MuJoCo scene.

Used to build counterfactual observation pairs: render the *same* simulator
state twice, changing only one object's appearance, so that any downstream
activation difference is attributable to that object's pixels rather than to a
diverged trajectory.

Everything here mutates an already-loaded ``mjModel`` in place and is reverted
by a saved snapshot. No XML editing, no environment rebuild.

Color precedence in MuJoCo (verified against mujoco 3.8.1)
----------------------------------------------------------
A geom is drawn with ``geom_rgba`` when that value differs from MuJoCo's parsed
default of ``[0.5, 0.5, 0.5, 1.0]``; otherwise the assigned material takes over
(``mat_rgba`` modulating any texture). So:

* setting ``geom_rgba`` to an explicit color overrides a textured material,
* and a geom left at the default gray is driven by its material.

That is why :func:`set_geom_color` refuses to silently write the default gray:
doing so would hand control back to the material and look like a no-op.

Both paths take effect on the next render with no texture re-upload. Editing
``tex_data`` is the exception -- texture bytes live in the GPU context and need
``mjr_uploadTexture`` -- which is why this module perturbs color, not texture.
"""

from __future__ import annotations

import colorsys
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

MUJOCO_DEFAULT_GEOM_RGBA = (0.5, 0.5, 0.5, 1.0)

# Names that belong to the robot, gripper, arena or mount rather than to a
# manipulable task object. Matched case-insensitively as substrings.
NON_OBJECT_NAME_HINTS = (
    "robot",
    "gripper",
    "finger",
    "mount",
    "table",
    "wall",
    "floor",
    "ground",
    "world",
    "light",
    "camera",
    "base",
    "pedestal",
    "controller_box",
)


@dataclass(frozen=True)
class SceneGeom:
    """One renderable geom, resolved to the body that owns it."""

    geom_id: int
    geom_name: str | None
    body_id: int
    body_name: str | None
    material_id: int
    rgba: tuple[float, float, float, float]
    uses_material: bool

    @property
    def is_default_gray(self) -> bool:
        return np.allclose(self.rgba, MUJOCO_DEFAULT_GEOM_RGBA)


@dataclass(frozen=True)
class SceneObject:
    """A body and the geoms drawn for it."""

    body_id: int
    body_name: str
    geoms: tuple[SceneGeom, ...]

    @property
    def geom_ids(self) -> tuple[int, ...]:
        return tuple(geom.geom_id for geom in self.geoms)

    @property
    def looks_like_task_object(self) -> bool:
        lowered = self.body_name.lower()
        return not any(hint in lowered for hint in NON_OBJECT_NAME_HINTS)


@dataclass
class ColorPerturbation:
    """A reversible recolor of a set of geoms."""

    geom_ids: tuple[int, ...]
    original_rgba: np.ndarray
    original_matid: np.ndarray
    applied_rgba: np.ndarray
    label: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def revert(self, model: Any) -> None:
        """Restore the exact pre-perturbation appearance."""
        raw = resolve_mj_model(model)
        for offset, geom_id in enumerate(self.geom_ids):
            raw.geom_rgba[geom_id] = self.original_rgba[offset]
            raw.geom_matid[geom_id] = self.original_matid[offset]


_WRAPPER_ATTRS = ("_model", "model", "sim", "env", "unwrapped")


def resolve_mj_model(candidate: Any) -> Any:
    """Return the genuine ``mujoco.MjModel`` behind a robosuite/LIBERO wrapper.

    robosuite's ``binding_utils.MjModel`` is *not* an ``mujoco.MjModel``: its
    metaclass installs a ``property`` for every public attribute of the real
    class, forwarding to the instance it holds in ``_model``. So the wrapper
    answers ``hasattr(w, "geom_rgba")`` with True while still being rejected by
    the C API -- ``mujoco.mj_id2name(wrapper, ...)`` raises ``TypeError``.

    Duck-typing therefore stops one level too early. Accept only a real
    ``mujoco.MjModel`` and keep walking otherwise.
    """
    import mujoco

    seen: set[int] = set()
    queue = [candidate]
    inspected: list[str] = []
    while queue:
        node = queue.pop(0)
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, mujoco.MjModel):
            return node
        inspected.append(type(node).__name__)
        for attr in _WRAPPER_ATTRS:
            try:
                queue.append(getattr(node, attr, None))
            except Exception:
                # A wrapper property can raise before the sim is built; that
                # branch simply has nothing to offer.
                continue
    raise TypeError(
        f"Could not resolve a mujoco.MjModel from {type(candidate).__name__}. "
        f"Walked {_WRAPPER_ATTRS} through: {inspected}. Pass the raw model explicitly."
    )


def _name_of(model: Any, obj_enum: Any, index: int) -> str | None:
    import mujoco

    name = mujoco.mj_id2name(model, obj_enum, index)
    return name or None


def list_scene_geoms(model: Any) -> list[SceneGeom]:
    """Describe every geom in the model, with its owning body."""
    import mujoco

    raw = resolve_mj_model(model)
    geoms = []
    for geom_id in range(raw.ngeom):
        body_id = int(raw.geom_bodyid[geom_id])
        material_id = int(raw.geom_matid[geom_id])
        rgba = tuple(float(value) for value in raw.geom_rgba[geom_id])
        geoms.append(
            SceneGeom(
                geom_id=geom_id,
                geom_name=_name_of(raw, mujoco.mjtObj.mjOBJ_GEOM, geom_id),
                body_id=body_id,
                body_name=_name_of(raw, mujoco.mjtObj.mjOBJ_BODY, body_id),
                material_id=material_id,
                rgba=rgba,
                # A geom left at the default gray is drawn by its material.
                uses_material=material_id >= 0 and np.allclose(rgba, MUJOCO_DEFAULT_GEOM_RGBA),
            )
        )
    return geoms


def list_scene_objects(model: Any, *, task_objects_only: bool = False) -> list[SceneObject]:
    """Group geoms by body so a caller can pick an object by name."""
    grouped: dict[int, list[SceneGeom]] = {}
    for geom in list_scene_geoms(model):
        grouped.setdefault(geom.body_id, []).append(geom)

    objects = []
    for body_id, geoms in sorted(grouped.items()):
        body_name = geoms[0].body_name or f"body_{body_id}"
        objects.append(SceneObject(body_id=body_id, body_name=body_name, geoms=tuple(geoms)))
    if task_objects_only:
        objects = [obj for obj in objects if obj.looks_like_task_object]
    return objects


def find_objects(model: Any, pattern: str, *, task_objects_only: bool = False) -> list[SceneObject]:
    """Find objects whose body name matches a case-insensitive regex."""
    regex = re.compile(pattern, re.IGNORECASE)
    return [
        obj
        for obj in list_scene_objects(model, task_objects_only=task_objects_only)
        if regex.search(obj.body_name)
    ]


def resolve_geom_ids(model: Any, target: str | SceneObject | Iterable[int]) -> tuple[int, ...]:
    """Accept a name pattern, a resolved object, or explicit geom ids."""
    if isinstance(target, SceneObject):
        return target.geom_ids
    if isinstance(target, str):
        matches = find_objects(model, target)
        if not matches:
            available = [obj.body_name for obj in list_scene_objects(model, task_objects_only=True)]
            raise LookupError(f"No body matched {target!r}. Task-like bodies present: {available}")
        return tuple(geom_id for obj in matches for geom_id in obj.geom_ids)
    ids = tuple(int(value) for value in target)
    if not ids:
        raise ValueError("No geom ids given to perturb.")
    return ids


def _as_rgba(color: Sequence[float], *, fallback_alpha: float) -> np.ndarray:
    values = [float(component) for component in color]
    if len(values) == 3:
        values.append(fallback_alpha)
    if len(values) != 4:
        raise ValueError(f"Expected an RGB or RGBA color, got {len(values)} components.")
    return np.asarray(values, dtype=np.float64)


def _guard_against_default_gray(rgba: np.ndarray) -> np.ndarray:
    """Nudge a color that would hand rendering back to the material.

    ``geom_rgba`` only wins over an assigned material when it differs from
    MuJoCo's default gray. Writing exactly that value would silently render as
    no change, so shift it by one part in a thousand -- visually identical,
    but unambiguous to the renderer.
    """
    if np.allclose(rgba, MUJOCO_DEFAULT_GEOM_RGBA):
        rgba = rgba.copy()
        rgba[0] += 1e-3
    return rgba


def set_geom_color(
    model: Any,
    target: str | SceneObject | Iterable[int],
    color: Sequence[float],
    *,
    label: str = "",
    detach_material: bool = False,
) -> ColorPerturbation:
    """Recolor every geom of ``target`` and return a reversible handle.

    Args:
        detach_material: also clear ``geom_matid``. Not needed for a flat
            recolor (an explicit ``geom_rgba`` already wins), but it strips the
            texture pattern as well, which makes the change more uniform.
    """
    raw = resolve_mj_model(model)
    geom_ids = resolve_geom_ids(raw, target)

    original_rgba = np.array([raw.geom_rgba[gid].copy() for gid in geom_ids], dtype=np.float64)
    original_matid = np.array([int(raw.geom_matid[gid]) for gid in geom_ids], dtype=np.int64)

    applied = np.stack(
        [_guard_against_default_gray(_as_rgba(color, fallback_alpha=original_rgba[i][3])) for i in range(len(geom_ids))]
    )
    for offset, geom_id in enumerate(geom_ids):
        raw.geom_rgba[geom_id] = applied[offset]
        if detach_material:
            raw.geom_matid[geom_id] = -1

    return ColorPerturbation(
        geom_ids=tuple(geom_ids),
        original_rgba=original_rgba,
        original_matid=original_matid,
        applied_rgba=applied,
        label=label or f"set_color{tuple(round(float(c), 3) for c in applied[0])}",
    )


def blend_geom_color(
    model: Any,
    target: str | SceneObject | Iterable[int],
    color: Sequence[float],
    amount: float,
    *,
    label: str = "",
) -> ColorPerturbation:
    """Interpolate each geom's color toward ``color`` by ``amount`` in [0, 1].

    This is the dose knob for a dose-response check. A straight RGB blend is
    monotone in ``amount`` by construction, so the pixel delta grows smoothly
    with the dose -- which :func:`shift_geom_hue` does *not* guarantee for a
    near-gray object, where raising saturation off the color-circle center
    dominates the rotation.

    ``amount=0`` reproduces the original color and is the natural null dose.
    """
    if not 0.0 <= amount <= 1.0:
        raise ValueError(f"amount must be in [0, 1], got {amount}")

    raw = resolve_mj_model(model)
    geom_ids = resolve_geom_ids(raw, target)
    target_rgb = _as_rgba(color, fallback_alpha=1.0)[:3]

    original_rgba = np.array([raw.geom_rgba[gid].copy() for gid in geom_ids], dtype=np.float64)
    original_matid = np.array([int(raw.geom_matid[gid]) for gid in geom_ids], dtype=np.int64)

    applied = []
    for row in original_rgba:
        blended = row.copy()
        blended[:3] = (1.0 - amount) * row[:3] + amount * target_rgb
        applied.append(_guard_against_default_gray(blended))
    applied_arr = np.stack(applied)

    for offset, geom_id in enumerate(geom_ids):
        raw.geom_rgba[geom_id] = applied_arr[offset]

    return ColorPerturbation(
        geom_ids=tuple(geom_ids),
        original_rgba=original_rgba,
        original_matid=original_matid,
        applied_rgba=applied_arr,
        label=label or f"blend{amount:.3f}->{tuple(round(float(c), 2) for c in target_rgb)}",
        extras={"amount": amount, "target_rgb": [float(c) for c in target_rgb]},
    )


def shift_geom_hue(
    model: Any,
    target: str | SceneObject | Iterable[int],
    hue_shift: float,
    *,
    saturation_floor: float = 0.35,
    label: str = "",
) -> ColorPerturbation:
    """Rotate each geom's hue, preserving its brightness.

    Useful when the perturbed region must keep its luminance and footprint, so
    the change is purely chromatic.

    Near-gray geoms have no meaningful hue, so their saturation is raised to
    ``saturation_floor`` first; otherwise the rotation would be a no-op. Note
    that for such geoms the pixel delta is then dominated by that saturation
    jump rather than by ``hue_shift``, so hue is *not* a monotone dose knob on
    near-gray objects -- use :func:`blend_geom_color` for dose-response.
    """
    raw = resolve_mj_model(model)
    geom_ids = resolve_geom_ids(raw, target)

    original_rgba = np.array([raw.geom_rgba[gid].copy() for gid in geom_ids], dtype=np.float64)
    original_matid = np.array([int(raw.geom_matid[gid]) for gid in geom_ids], dtype=np.int64)

    applied = []
    for row in original_rgba:
        red, green, blue, alpha = (float(value) for value in row)
        hue, lightness, saturation = colorsys.rgb_to_hls(red, green, blue)
        saturation = max(saturation, saturation_floor)
        hue = (hue + hue_shift) % 1.0
        red, green, blue = colorsys.hls_to_rgb(hue, lightness, saturation)
        applied.append(_guard_against_default_gray(np.array([red, green, blue, alpha], dtype=np.float64)))
    applied_arr = np.stack(applied)

    for offset, geom_id in enumerate(geom_ids):
        raw.geom_rgba[geom_id] = applied_arr[offset]

    return ColorPerturbation(
        geom_ids=tuple(geom_ids),
        original_rgba=original_rgba,
        original_matid=original_matid,
        applied_rgba=applied_arr,
        label=label or f"hue_shift{hue_shift:+.3f}",
        extras={"hue_shift": hue_shift},
    )


def image_delta_stats(baseline: np.ndarray, perturbed: np.ndarray) -> dict[str, float]:
    """Quantify how much of the frame the perturbation actually moved.

    An activation delta is only interpretable next to this: it separates "the
    features respond strongly to a tiny pixel change" from "the image barely
    changed" and from "half the frame changed".
    """
    base_raw = np.asarray(baseline)
    pert_raw = np.asarray(perturbed)
    if base_raw.shape != pert_raw.shape:
        raise ValueError(f"Image shapes differ: {base_raw.shape} vs {pert_raw.shape}")

    # Decide the scale from both frames and from the dtype. Checking only the
    # baseline's max would misread an all-black baseline as already normalized.
    is_byte = base_raw.dtype == np.uint8 or pert_raw.dtype == np.uint8
    base = base_raw.astype(np.float64)
    pert = pert_raw.astype(np.float64)
    scale = 255.0 if is_byte or max(base.max(initial=0.0), pert.max(initial=0.0)) > 1.5 else 1.0
    diff = np.abs(pert - base) / scale
    per_pixel = diff.max(axis=-1) if diff.ndim == 3 else diff
    changed = per_pixel > (1.0 / 255.0)
    return {
        "mean_abs_delta": float(diff.mean()),
        "max_abs_delta": float(diff.max()),
        "changed_pixel_fraction": float(changed.mean()),
        "changed_pixel_count": int(changed.sum()),
        "l2_delta": float(np.linalg.norm(diff)),
        "relative_l2": float(np.linalg.norm(diff) / (np.linalg.norm(base / scale) or 1.0)),
    }

#!/usr/bin/env python
"""Smoke test for targeted MuJoCo scene perturbation.

Runs against a real ``mjModel`` built to mirror the parts of a LIBERO scene
that matter here: named object bodies, textured-mesh-style materials, robot and
arena bodies that must not be mistaken for task objects, and multi-geom bodies.

The renders are real, so these checks verify that a perturbation actually
reaches the pixels -- which is the claim the whole counterfactual probe rests
on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pi05_mi.scene_perturbation import (  # noqa: E402
    MUJOCO_DEFAULT_GEOM_RGBA,
    ColorPerturbation,
    SceneObject,
    blend_geom_color,
    find_objects,
    image_delta_stats,
    list_scene_objects,
    resolve_geom_ids,
    resolve_mj_model,
    set_geom_color,
    shift_geom_hue,
)

SCENE_XML = """
<mujoco>
  <asset>
    <texture name="wood" type="2d" builtin="checker" rgb1="0.7 0.5 0.3" rgb2="0.5 0.35 0.2"
             width="64" height="64"/>
    <material name="wood_mat" texture="wood"/>
    <material name="ramekin_mat" rgba="0.85 0.85 0.8 1"/>
  </asset>
  <worldbody>
    <light pos="0 0 3"/>
    <camera name="agentview" pos="0 -1.4 0.9" xyaxes="1 0 0 0 0.6 1"/>
    <body name="table">
      <geom name="table_top" type="box" size="1 1 .02" pos="0 0 -.02" material="wood_mat"/>
    </body>
    <body name="robot0_link" pos="-0.8 0 0">
      <geom name="robot0_geom" type="capsule" size=".05 .2" rgba="0.3 0.3 0.35 1"/>
    </body>
    <body name="akita_black_bowl_1" pos="0 0 0.06">
      <geom name="bowl_outer" type="cylinder" size=".09 .05" rgba="0.05 0.05 0.05 1"/>
      <geom name="bowl_rim" type="cylinder" size=".095 .01" pos="0 0 .05" rgba="0.08 0.08 0.08 1"/>
    </body>
    <body name="plate_1" pos="0.35 0 0.01">
      <geom name="plate_geom" type="cylinder" size=".11 .01" material="wood_mat"/>
    </body>
    <body name="red_block_1" pos="0 0.3 0.04">
      <geom name="red_block_geom" type="box" size=".04 .04 .04" rgba="0.9 0.15 0.1 1"/>
    </body>
    <body name="wooden_ramekin_1" pos="-0.35 0 0.04">
      <geom name="ramekin_geom" type="cylinder" size=".05 .04" material="ramekin_mat"/>
    </body>
  </worldbody>
</mujoco>
"""


def _model():
    import mujoco

    return mujoco.MjModel.from_xml_string(SCENE_XML)


class _Renderer:
    """Render the same state repeatedly without stepping physics."""

    def __init__(self, model):
        import mujoco

        self.mujoco = mujoco
        self.model = model
        self.data = mujoco.MjData(model)
        self.renderer = mujoco.Renderer(model, height=128, width=128)

    def shot(self) -> np.ndarray:
        self.mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera="agentview")
        return self.renderer.render().astype(np.float64)

    def close(self) -> None:
        self.renderer.close()


def test_objects_are_grouped_by_body_with_all_their_geoms() -> None:
    objects = {obj.body_name: obj for obj in list_scene_objects(_model())}
    assert "akita_black_bowl_1" in objects
    bowl = objects["akita_black_bowl_1"]
    assert len(bowl.geoms) == 2, "a multi-geom object must expose every geom, not just the first"
    assert {geom.geom_name for geom in bowl.geoms} == {"bowl_outer", "bowl_rim"}


def test_robot_and_arena_bodies_are_not_offered_as_task_objects() -> None:
    names = {obj.body_name for obj in list_scene_objects(_model(), task_objects_only=True)}
    assert "akita_black_bowl_1" in names
    assert "plate_1" in names
    assert "robot0_link" not in names
    assert "table" not in names
    assert "world" not in names


def test_find_objects_matches_partial_names_case_insensitively() -> None:
    model = _model()
    assert [obj.body_name for obj in find_objects(model, "black_bowl")] == ["akita_black_bowl_1"]
    assert [obj.body_name for obj in find_objects(model, "BLACK_BOWL")] == ["akita_black_bowl_1"]
    assert find_objects(model, "no_such_object") == []


def test_resolve_geom_ids_reports_available_objects_when_lookup_fails() -> None:
    model = _model()
    try:
        resolve_geom_ids(model, "definitely_not_here")
    except LookupError as exc:
        assert "akita_black_bowl_1" in str(exc), "the error must tell the caller what it could have picked"
    else:
        raise AssertionError("expected a LookupError for an unmatched object name")


def test_recoloring_an_untextured_object_changes_the_rendered_pixels() -> None:
    model = _model()
    renderer = _Renderer(model)
    try:
        base = renderer.shot()
        perturbation = set_geom_color(model, "black_bowl", (0.95, 0.1, 0.1))
        after = renderer.shot()
        stats = image_delta_stats(base, after)
        assert stats["changed_pixel_fraction"] > 0.001, f"recolor did not reach the pixels: {stats}"

        perturbation.revert(model)
        restored = renderer.shot()
        assert np.array_equal(restored, base), "revert must restore the original frame exactly"
    finally:
        renderer.close()


def test_recoloring_a_textured_object_also_changes_pixels() -> None:
    """geom_rgba outranks an assigned material, so textured LIBERO meshes recolor too."""
    model = _model()
    renderer = _Renderer(model)
    try:
        base = renderer.shot()
        set_geom_color(model, "plate_1", (0.1, 0.2, 0.95))
        stats = image_delta_stats(base, renderer.shot())
        assert stats["changed_pixel_fraction"] > 0.001, f"textured object did not recolor: {stats}"
    finally:
        renderer.close()


def test_default_gray_is_nudged_so_it_cannot_silently_no_op() -> None:
    """Writing MuJoCo's default gray would hand rendering back to the material."""
    model = _model()
    perturbation = set_geom_color(model, "plate_1", MUJOCO_DEFAULT_GEOM_RGBA)
    raw = resolve_mj_model(model)
    applied = raw.geom_rgba[perturbation.geom_ids[0]]
    assert not np.allclose(applied, MUJOCO_DEFAULT_GEOM_RGBA)
    assert np.allclose(applied, MUJOCO_DEFAULT_GEOM_RGBA, atol=2e-3), "the nudge must stay visually identical"


def test_detach_material_strips_the_texture_and_is_reverted() -> None:
    model = _model()
    raw = resolve_mj_model(model)
    plate_geoms = resolve_geom_ids(model, "plate_1")
    original_matid = int(raw.geom_matid[plate_geoms[0]])
    assert original_matid >= 0, "the plate should start with a material assigned"

    perturbation = set_geom_color(model, "plate_1", (0.9, 0.1, 0.1), detach_material=True)
    assert raw.geom_matid[plate_geoms[0]] == -1
    perturbation.revert(model)
    assert int(raw.geom_matid[plate_geoms[0]]) == original_matid


def test_blend_gives_a_monotone_dose_response() -> None:
    """The dose knob the counterfactual sweep relies on."""
    model = _model()
    renderer = _Renderer(model)
    try:
        base = renderer.shot()
        deltas = []
        for amount in (0.0, 0.25, 0.5, 1.0):
            perturbation = blend_geom_color(model, "black_bowl", (1.0, 0.2, 0.1), amount)
            deltas.append(image_delta_stats(base, renderer.shot())["l2_delta"])
            perturbation.revert(model)
        assert deltas[0] == 0.0, "amount=0 must be an exact null dose"
        assert deltas[1] < deltas[2] < deltas[3], f"blend dose-response is not monotone: {deltas}"
    finally:
        renderer.close()


def test_blend_rejects_an_out_of_range_dose() -> None:
    model = _model()
    for bad in (-0.1, 1.5):
        try:
            blend_geom_color(model, "black_bowl", (1.0, 0.0, 0.0), bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for amount={bad}")


def test_hue_shift_is_monotone_on_a_saturated_object() -> None:
    """Hue rotation is a valid dose knob only where the object already has hue."""
    model = _model()
    renderer = _Renderer(model)
    try:
        base = renderer.shot()
        deltas = []
        for shift in (0.05, 0.25, 0.5):
            perturbation = shift_geom_hue(model, "red_block", shift)
            deltas.append(image_delta_stats(base, renderer.shot())["l2_delta"])
            perturbation.revert(model)
        assert deltas[0] < deltas[1] < deltas[2], f"hue dose-response is not monotone: {deltas}"
    finally:
        renderer.close()


def test_hue_shift_moves_a_near_black_object() -> None:
    """A near-black bowl has no usable hue; the saturation floor must rescue it."""
    model = _model()
    renderer = _Renderer(model)
    try:
        base = renderer.shot()
        shift_geom_hue(model, "black_bowl", 0.4)
        stats = image_delta_stats(base, renderer.shot())
        assert stats["changed_pixel_fraction"] > 0.001, f"near-black object did not shift: {stats}"
    finally:
        renderer.close()


def test_perturbing_one_object_leaves_the_others_untouched() -> None:
    model = _model()
    raw = resolve_mj_model(model)
    ramekin_geoms = resolve_geom_ids(model, "ramekin")
    before = raw.geom_rgba[ramekin_geoms[0]].copy()
    set_geom_color(model, "black_bowl", (1.0, 0.0, 0.0))
    assert np.array_equal(raw.geom_rgba[ramekin_geoms[0]], before)


def test_identical_renders_have_exactly_zero_delta() -> None:
    """The null control the counterfactual probe depends on."""
    model = _model()
    renderer = _Renderer(model)
    try:
        first, second = renderer.shot(), renderer.shot()
        stats = image_delta_stats(first, second)
        assert stats["max_abs_delta"] == 0.0
        assert stats["changed_pixel_fraction"] == 0.0
    finally:
        renderer.close()


def test_image_delta_stats_handles_uint8_and_float_inputs() -> None:
    base_u8 = np.zeros((8, 8, 3), dtype=np.uint8)
    pert_u8 = base_u8.copy()
    pert_u8[0, 0] = 255
    stats = image_delta_stats(base_u8, pert_u8)
    assert stats["max_abs_delta"] == 1.0, "uint8 input must be scaled to 0-1"
    assert stats["changed_pixel_count"] == 1
    assert abs(stats["changed_pixel_fraction"] - 1 / 64) < 1e-12


def _robosuite_style_wrapper(raw):
    """Replicate robosuite's binding_utils.MjModel wrapping behaviour.

    This is the shape that matters: a metaclass installs a property for every
    public attribute of mujoco.MjModel forwarding to the wrapped instance, so
    the wrapper passes every hasattr check while still being rejected by the C
    API. A naive fake that only holds ``_model`` does not reproduce that, which
    is precisely how the real bug slipped through.
    """
    import mujoco

    class _Meta(type):
        def __new__(cls, name, bases, dct):
            for attr in dir(mujoco.MjModel):
                if not attr.startswith("_") and attr not in dct:
                    dct[attr] = property(
                        lambda self, attr=attr: getattr(self._model, attr),
                        lambda self, value, attr=attr: setattr(self._model, attr, value),
                    )
            return super().__new__(cls, name, bases, dct)

    class _WrappedModel(metaclass=_Meta):
        def __init__(self, model):
            self._model = model

    return _WrappedModel(raw)


def _real_robosuite_wrapper(raw):
    """Build the genuine robosuite wrapper when robosuite is importable."""
    try:
        from robosuite.utils.binding_utils import MjModel as RobosuiteMjModel
    except Exception:
        return None
    return RobosuiteMjModel(raw)


def test_robosuite_wrapper_passes_hasattr_but_is_not_a_real_model() -> None:
    """Pin the property that makes duck-typing the wrong test here."""
    import mujoco

    raw = _model()
    for wrapper in filter(None, [_robosuite_style_wrapper(raw), _real_robosuite_wrapper(raw)]):
        assert hasattr(wrapper, "geom_rgba")
        assert hasattr(wrapper, "geom_matid")
        assert hasattr(wrapper, "geom_bodyid")
        assert not isinstance(wrapper, mujoco.MjModel)
        try:
            mujoco.mj_id2name(wrapper, mujoco.mjtObj.mjOBJ_GEOM, 0)
        except TypeError:
            pass
        else:
            raise AssertionError("the C API should reject the wrapper")


def test_resolve_mj_model_unwraps_the_robosuite_wrapper() -> None:
    raw = _model()
    for wrapper in filter(None, [_robosuite_style_wrapper(raw), _real_robosuite_wrapper(raw)]):
        assert resolve_mj_model(wrapper) is raw, "must unwrap to the genuine model, not stop at the wrapper"


def test_resolve_mj_model_unwraps_the_full_libero_chain() -> None:
    """LIBERO -> robosuite env -> MjSim -> wrapper -> raw model."""
    raw = _model()
    wrapper = _robosuite_style_wrapper(raw)

    class _Sim:
        def __init__(self): self.model = wrapper

    class _RobosuiteEnv:
        def __init__(self): self.sim = _Sim()

    class _OffScreenRenderEnv:
        def __init__(self): self.env = _RobosuiteEnv()

    resolved = resolve_mj_model(_OffScreenRenderEnv())
    assert resolved is raw
    # The whole point of resolving: the C API must now accept it.
    import mujoco
    assert mujoco.mj_id2name(resolved, mujoco.mjtObj.mjOBJ_GEOM, 0) is not None


def test_perturbation_works_through_a_robosuite_wrapper() -> None:
    """End to end through the wrapper, since that is how it is called for real."""
    raw = _model()
    wrapper = _robosuite_style_wrapper(raw)
    objects = list_scene_objects(wrapper, task_objects_only=True)
    assert "akita_black_bowl_1" in {o.body_name for o in objects}

    perturbation = set_geom_color(wrapper, "black_bowl", (0.95, 0.1, 0.1))
    gid = perturbation.geom_ids[0]
    # Writes must land on the shared underlying arrays, visible through both views.
    assert np.allclose(raw.geom_rgba[gid], wrapper.geom_rgba[gid])
    assert raw.geom_rgba[gid][0] > 0.9
    perturbation.revert(wrapper)
    assert raw.geom_rgba[gid][0] < 0.1


def test_resolve_mj_model_reports_what_it_walked() -> None:
    try:
        resolve_mj_model(object())
    except TypeError as exc:
        assert "mujoco.MjModel" in str(exc)
        assert "object" in str(exc), "the error should name the types it inspected"
    else:
        raise AssertionError("expected a TypeError for an unresolvable object")


def test_resolve_mj_model_survives_a_property_that_raises() -> None:
    """Half-built sims expose attributes that throw; that branch is just empty."""
    raw = _model()

    class _Exploding:
        @property
        def sim(self):
            raise RuntimeError("sim not created yet")

        @property
        def model(self):
            return raw

    assert resolve_mj_model(_Exploding()) is raw


def test_perturbation_accepts_a_resolved_object_and_explicit_geom_ids() -> None:
    model = _model()
    bowl = find_objects(model, "black_bowl")[0]
    assert isinstance(bowl, SceneObject)
    by_object = set_geom_color(model, bowl, (0.2, 0.9, 0.2))
    assert by_object.geom_ids == bowl.geom_ids
    by_object.revert(model)

    by_ids = set_geom_color(model, list(bowl.geom_ids), (0.2, 0.2, 0.9))
    assert isinstance(by_ids, ColorPerturbation)
    assert by_ids.geom_ids == bowl.geom_ids


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok {test.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    main()

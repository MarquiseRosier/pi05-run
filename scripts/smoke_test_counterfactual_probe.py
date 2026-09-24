#!/usr/bin/env python
"""Smoke test for the counterfactual probe's glue code.

Every failure this probe has hit in Colab so far has been glue -- an inherited
env var, an interactive import, a wrapper that duck-types like the thing it
wraps. None of it was the measurement logic. So these checks exercise the
boundaries against the *real* libraries wherever they are importable locally
(lerobot, mujoco), rather than against hand-written fakes that agree with my
assumptions.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import probe_pi05_transcoder_counterfactual as P  # noqa: E402


def _libero_style_observation(height: int = 128, width: int = 128) -> dict:
    """Mirror what LiberoEnv._format_raw_obs returns for one env.

    Shapes and dtypes match lerobot/envs/libero.py: channel-last uint8 images
    and a nested robot_state dict.
    """
    rng = np.random.default_rng(0)
    return {
        "pixels": {
            "image": rng.integers(0, 256, (height, width, 3), dtype=np.uint8),
            "image2": rng.integers(0, 256, (height, width, 3), dtype=np.uint8),
        },
        "robot_state": {
            "eef": {
                "pos": rng.standard_normal(3).astype(np.float32),
                "quat": rng.standard_normal(4).astype(np.float32),
                "mat": rng.standard_normal((3, 3)).astype(np.float32),
            },
            "gripper": {
                "qpos": rng.standard_normal(2).astype(np.float32),
                "qvel": rng.standard_normal(2).astype(np.float32),
            },
            "joints": {
                "pos": rng.standard_normal(7).astype(np.float32),
                "vel": rng.standard_normal(7).astype(np.float32),
            },
        },
    }


def test_add_batch_axis_matches_what_the_vector_env_would_emit() -> None:
    batched = P._add_batch_axis(_libero_style_observation())
    assert batched["pixels"]["image"].shape == (1, 128, 128, 3)
    assert batched["pixels"]["image"].dtype == np.uint8, "preprocess_observation asserts uint8"
    assert batched["robot_state"]["eef"]["pos"].shape == (1, 3)
    assert batched["robot_state"]["eef"]["mat"].shape == (1, 3, 3)
    assert batched["robot_state"]["joints"]["vel"].shape == (1, 7)


def test_batched_observation_survives_real_lerobot_preprocess_observation() -> None:
    """The next thing that would break in Colab, checked against the real function.

    ``preprocess_observation`` asserts channel-last uint8 and rearranges to
    channel-first float. If the batch axis were wrong those asserts fire.
    """
    try:
        from lerobot.envs.utils import preprocess_observation
    except Exception as exc:  # pragma: no cover - lerobot missing locally
        print(f"  (skipped: lerobot not importable: {type(exc).__name__})")
        return

    batched = P._add_batch_axis(_libero_style_observation())
    out = preprocess_observation(batched)

    image_keys = [k for k in out if k.startswith("observation.images.")]
    assert sorted(image_keys) == ["observation.images.image", "observation.images.image2"]
    for key in image_keys:
        tensor = out[key]
        assert tuple(tensor.shape) == (1, 3, 128, 128), f"{key} should be channel-first batched"
        assert tensor.dtype.is_floating_point
        assert 0.0 <= float(tensor.min()) and float(tensor.max()) <= 1.0, "must be scaled to [0,1]"

    state = out["observation.robot_state"]
    assert tuple(state["eef"]["pos"].shape) == (1, 3)
    assert tuple(state["eef"]["mat"].shape) == (1, 3, 3)


def test_batch_axis_is_load_bearing_for_robot_state() -> None:
    """Why _add_batch_axis exists, stated precisely.

    ``preprocess_observation`` unsqueezes a 3-D image itself, so images survive
    an unbatched observation. ``_convert_nested_dict`` does no such thing for
    ``robot_state``. Skipping the batch axis therefore yields images at
    (1, C, H, W) beside state at (3,) -- silently inconsistent rather than a
    clean failure, which is the worst kind.
    """
    try:
        from lerobot.envs.utils import preprocess_observation
    except Exception:
        return

    raw = _libero_style_observation()
    unbatched = preprocess_observation(raw)
    assert tuple(unbatched["observation.images.image"].shape) == (1, 3, 128, 128)
    assert tuple(unbatched["observation.robot_state"]["eef"]["pos"].shape) == (3,), (
        "robot_state is not auto-batched, so the mismatch would pass silently"
    )

    batched = preprocess_observation(P._add_batch_axis(raw))
    assert tuple(batched["observation.images.image"].shape) == (1, 3, 128, 128)
    assert tuple(batched["observation.robot_state"]["eef"]["pos"].shape) == (1, 3)
    leading = {
        tuple(batched["observation.images.image"].shape)[0],
        tuple(batched["observation.robot_state"]["eef"]["pos"].shape)[0],
        tuple(batched["observation.robot_state"]["joints"]["vel"].shape)[0],
    }
    assert leading == {1}, f"every tensor must share one batch dim, got {leading}"


def test_environment_hardening_rewrites_only_the_inline_backend() -> None:
    saved = {k: os.environ.get(k) for k in ("MPLBACKEND", "MUJOCO_GL", "PYOPENGL_PLATFORM")}
    try:
        os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
        os.environ.pop("MUJOCO_GL", None)
        os.environ.pop("PYOPENGL_PLATFORM", None)
        P._harden_environment()
        assert os.environ["MPLBACKEND"] == "Agg"
        # EGL only exists on Linux; defaulting it on macOS makes `import mujoco` raise.
        if sys.platform.startswith("linux"):
            assert os.environ["MUJOCO_GL"] == "egl"
            assert os.environ["PYOPENGL_PLATFORM"] == "egl"
        else:
            assert "MUJOCO_GL" not in os.environ, "must not force EGL off Linux"

        os.environ["MPLBACKEND"] = "pdf"
        os.environ["MUJOCO_GL"] = "osmesa"
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
        P._harden_environment()
        assert os.environ["MPLBACKEND"] == "pdf", "an explicit backend must be left alone"
        assert os.environ["MUJOCO_GL"] == "osmesa"
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_libero_config_is_written_with_paths_that_exist() -> None:
    import tempfile

    import yaml

    saved = os.environ.get("LIBERO_CONFIG_PATH")
    tmp = Path(tempfile.mkdtemp())
    pkg = tmp / "site" / "libero"
    (pkg / "libero" / "bddl_files").mkdir(parents=True)
    (pkg / "libero" / "init_files").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    sys.path.insert(0, str(tmp / "site"))
    try:
        os.environ["LIBERO_CONFIG_PATH"] = str(tmp / "cfg")
        written = P.ensure_libero_config()
        data = yaml.safe_load(Path(written).read_text())
        assert set(data) == {"benchmark_root", "bddl_files", "init_states", "datasets", "assets"}
        assert Path(data["bddl_files"]).is_dir()
        assert Path(data["init_states"]).is_dir()
        # Idempotent: an existing config is never clobbered.
        Path(written).write_text("benchmark_root: /custom\n")
        assert Path(P.ensure_libero_config()).read_text() == "benchmark_root: /custom\n"
    finally:
        sys.path.remove(str(tmp / "site"))
        for name in [m for m in sys.modules if m == "libero" or m.startswith("libero.")]:
            del sys.modules[name]
        if saved is None:
            os.environ.pop("LIBERO_CONFIG_PATH", None)
        else:
            os.environ["LIBERO_CONFIG_PATH"] = saved


def test_shape_report_rejects_a_drifting_batch() -> None:
    """The guard that keeps a paired comparison honest."""
    from pi05_mi.langfuse_tracing import _tensor_shape_map

    import torch

    a = {"observation.images.image": torch.zeros(1, 3, 224, 224), "task": ["x"]}
    b = {"observation.images.image": torch.zeros(1, 3, 128, 128), "task": ["x"]}
    assert _tensor_shape_map(a) != _tensor_shape_map(b), (
        "differing image sizes must produce differing shape maps, "
        "otherwise the probe's drift guard cannot fire"
    )
    assert _tensor_shape_map(a) == _tensor_shape_map(dict(a))


SPATIAL_TASK0_BDDL = """
(define (problem LIBERO_Tabletop_Manipulation)
  (:language Pick the akita black bowl between the plate and the ramekin and place it on the plate)
  (:objects
    akita_black_bowl_1 akita_black_bowl_2 - akita_black_bowl
    cookies_1 - cookies
    plate_1 - plate
  )
  (:obj_of_interest
    akita_black_bowl_1
    plate_1
  )
)
"""

# Exactly the body names the real discovery pass reported for libero_spatial task 0.
REAL_BODY_NAMES = [
    "akita_black_bowl_1_main",
    "akita_black_bowl_2_main",
    "cookies_1_main",
    "glazed_rim_porcelain_ramekin_1_main",
    "plate_1_main",
    "wooden_cabinet_1_cabinet_top",
    "flat_stove_1_burner",
]


def _scene(body_names):
    import mujoco

    xml = "<mujoco><worldbody>" + "".join(
        f'<body name="{b}"><geom name="{b}_g" type="box" size=".05 .05 .05" rgba="0.05 0.05 0.05 1"/></body>'
        for b in body_names
    ) + "</worldbody></mujoco>"
    return mujoco.MjModel.from_xml_string(xml)


def _harness_with_bddl(text: str | None):
    import tempfile

    class _Inner:
        _task_bddl_file = None

    if text is not None:
        path = Path(tempfile.mkdtemp()) / "task.bddl"
        path.write_text(text)
        _Inner._task_bddl_file = str(path)

    class _Harness:
        inner_env = _Inner()

    return _Harness()


def test_obj_of_interest_is_read_from_the_task_bddl() -> None:
    names = P.read_objects_of_interest(_harness_with_bddl(SPATIAL_TASK0_BDDL))
    assert names == ["akita_black_bowl_1", "plate_1"]
    assert P.read_objects_of_interest(_harness_with_bddl(None)) == []


def test_auto_selection_picks_the_task_object_and_its_identical_sibling() -> None:
    from pi05_mi.scene_perturbation import find_objects

    model = _scene(REAL_BODY_NAMES)
    target, placebo, _notes = P.auto_select_targets(_harness_with_bddl(SPATIAL_TASK0_BDDL), model)

    assert [o.body_name for o in find_objects(model, target)] == ["akita_black_bowl_1_main"]
    assert [o.body_name for o in find_objects(model, placebo)] == ["akita_black_bowl_2_main"]


def test_auto_selected_patterns_are_exact_so_they_cannot_match_a_sibling() -> None:
    """A loose pattern would perturb both bowls and destroy the control."""
    from pi05_mi.scene_perturbation import find_objects

    model = _scene(REAL_BODY_NAMES)
    target, placebo, _ = P.auto_select_targets(_harness_with_bddl(SPATIAL_TASK0_BDDL), model)
    assert len(find_objects(model, target)) == 1
    assert len(find_objects(model, placebo)) == 1
    assert not set(o.body_name for o in find_objects(model, target)) & set(
        o.body_name for o in find_objects(model, placebo)
    )


def test_auto_selection_falls_back_when_there_is_no_sibling_instance() -> None:
    from pi05_mi.scene_perturbation import find_objects

    model = _scene(["akita_black_bowl_1_main", "plate_1_main", "cookies_1_main"])
    target, placebo, notes = P.auto_select_targets(_harness_with_bddl(SPATIAL_TASK0_BDDL), model)
    assert [o.body_name for o in find_objects(model, target)] == ["akita_black_bowl_1_main"]
    assert placebo is not None, "a placebo should still be chosen from a different object"
    assert [o.body_name for o in find_objects(model, placebo)] != ["akita_black_bowl_1_main"]
    assert any("no sibling instance" in note for note in notes)


def test_auto_selection_falls_back_when_the_bddl_is_unreadable() -> None:
    model = _scene(REAL_BODY_NAMES)
    target, _placebo, notes = P.auto_select_targets(_harness_with_bddl(None), model)
    assert target is not None, "must still choose something rather than crash"
    assert any("falling back" in note for note in notes)


def test_auto_selection_reports_nothing_to_pick_on_an_empty_scene() -> None:
    model = _scene(["robot0_link", "table"])  # filtered out as non-task bodies
    target, placebo, notes = P.auto_select_targets(_harness_with_bddl(None), model)
    assert target is None and placebo is None
    assert notes


class _CachingRobosuiteEnv:
    """Reproduce robosuite's observable caching.

    ``_get_observations()`` returns cached observable values and only refreshes
    them when ``force_update=True`` or a step occurs. A re-render that omits
    the flag hands back a stale frame, which is exactly how a perturbation came
    to look like a no-op while every identity check still passed.
    """

    def __init__(self):
        self.colour = 10
        self._cache = self._render()
        self.forced_updates = 0

    def _render(self):
        return np.full((8, 8, 3), self.colour, dtype=np.uint8)

    def _get_observations(self, force_update=False):
        if force_update:
            self.forced_updates += 1
            self._cache = self._render()
        return {"agentview_image": self._cache}


class _FakeLiberoEnv:
    def __init__(self):
        self.robosuite = _CachingRobosuiteEnv()

        class _Outer:
            env = self.robosuite

        self._env = _Outer()

    def _format_raw_obs(self, raw):
        return {"pixels": {"image": raw["agentview_image"]}}


class _FakeHarness:
    def __init__(self):
        self.inner_env = _FakeLiberoEnv()


def test_rerender_forces_an_update_instead_of_serving_a_cached_frame() -> None:
    harness = _FakeHarness()
    sim = harness.inner_env.robosuite

    before = P.first_camera_image(P.rerender_observation(harness))
    assert sim.forced_updates == 1, "rerender must request a forced update"

    # Change the scene the way a perturbation would.
    sim.colour = 200
    after = P.first_camera_image(P.rerender_observation(harness))
    assert sim.forced_updates == 2

    stats = P.image_delta_stats(before, after)
    assert stats["changed_pixel_fraction"] == 1.0, (
        "a scene change must reach the re-rendered frame; if this is 0 the "
        "re-render is serving robosuite's cache"
    )


def test_cached_rerender_would_hide_a_perturbation() -> None:
    """Pin the failure mode, so the guard above is demonstrably load-bearing."""
    sim = _CachingRobosuiteEnv()
    cached_before = sim._get_observations()["agentview_image"].copy()
    sim.colour = 200
    cached_after = sim._get_observations()["agentview_image"]
    assert P.image_delta_stats(cached_before, cached_after)["changed_pixel_fraction"] == 0.0
    forced = sim._get_observations(force_update=True)["agentview_image"]
    assert P.image_delta_stats(cached_before, forced)["changed_pixel_fraction"] == 1.0


SIBLING_BDDL = """
(define (problem LIBERO_Tabletop_Manipulation)
  (:language Pick the akita black bowl next to the ramekin and place it on the plate)
  (:obj_of_interest
    akita_black_bowl_1
    plate_1
  )
  (:init
    (On akita_black_bowl_1 main_table_next_to_ramekin_region)
    (On akita_black_bowl_2 main_table_next_to_box_region)
  )
)
"""

TASK0_WITH_INIT = """
(define (problem LIBERO_Tabletop_Manipulation)
  (:language Pick the akita black bowl between the plate and the ramekin and place it on the plate)
  (:obj_of_interest
    akita_black_bowl_1
    plate_1
  )
  (:init
    (On akita_black_bowl_1 main_table_between_plate_ramekin_region)
    (On akita_black_bowl_2 main_table_next_to_ramekin_region)
  )
)
"""


def _suite_dir():
    import tempfile

    folder = Path(tempfile.mkdtemp()) / "libero_spatial"
    folder.mkdir(parents=True)
    (folder / "task0.bddl").write_text(TASK0_WITH_INIT)
    (folder / "sibling.bddl").write_text(SIBLING_BDDL)
    return folder


def _harness_at(path: Path):
    class _Inner:
        _task_bddl_file = str(path)

    class _Harness:
        inner_env = _Inner()

    return _Harness()


def test_init_region_is_read_for_a_suffixed_body_name() -> None:
    """Scene bodies carry a _main suffix that BDDL object names do not."""
    assert P._init_region_of(TASK0_WITH_INIT, "akita_black_bowl_2_main") == "main_table_next_to_ramekin_region"
    assert P._init_region_of(TASK0_WITH_INIT, "akita_black_bowl_1_main") == "main_table_between_plate_ramekin_region"
    assert P._init_region_of(TASK0_WITH_INIT, "no_such_object_main") is None


def test_alt_prompt_is_the_sibling_task_naming_the_placebo_location() -> None:
    """The control that separates task relevance from screen position."""
    folder = _suite_dir()
    found = P.find_prompt_referring_to(_harness_at(folder / "task0.bddl"), "akita_black_bowl_2_main")
    assert found is not None
    prompt, source = found
    assert "next to the ramekin" in prompt, prompt
    assert source == "sibling"


def test_no_alt_prompt_when_no_sibling_places_its_target_there() -> None:
    import tempfile

    folder = Path(tempfile.mkdtemp()) / "libero_spatial"
    folder.mkdir(parents=True)
    (folder / "task0.bddl").write_text(TASK0_WITH_INIT)
    assert P.find_prompt_referring_to(_harness_at(folder / "task0.bddl"), "akita_black_bowl_2_main") is None


def test_alt_prompt_lookup_survives_a_missing_bddl() -> None:
    assert P.find_prompt_referring_to(_harness_with_bddl(None), "akita_black_bowl_2_main") is None


def test_save_image_accepts_the_formats_the_probe_produces() -> None:
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(1)
    cases = {
        "uint8_hwc.png": rng.integers(0, 256, (32, 32, 3), dtype=np.uint8),
        "batched.png": rng.integers(0, 256, (1, 32, 32, 3), dtype=np.uint8),
        "float01.png": rng.random((32, 32, 3)),
        "float255_diff.png": rng.random((32, 32, 3)) * 255.0,
    }
    for name, array in cases.items():
        P.save_image(tmp / name, array)
        assert (tmp / name).exists() and (tmp / name).stat().st_size > 0, f"{name} not written"


def _measurement(state: int, prompt: str, kind: str, latent: float, action: float) -> dict:
    return {
        "state_index": state, "prompt": prompt, "kind": kind, "target": kind, "dose": 1.0,
        "pixel": {"changed_pixel_fraction": 0.01},
        "latent": {"l2_delta_mean": latent},
        "action_relative_l2": action,
    }


def test_verdict_runs_with_prompt_variants_and_grounding_data() -> None:
    """The closing summary must not reach for main()'s locals.

    This shipped broken: the prompt-grounding block referenced `args` and
    `baseline_by_prompt` from inside a module-level function, so every run
    crashed at the very end -- after all the measurement work was done.
    """
    measurements = []
    for state in (0, 1):
        for prompt in ("task", "alt"):
            measurements.append(_measurement(state, prompt, "null", 0.0, 0.0))
            measurements.append(_measurement(state, prompt, "target", 8.0, 0.3))
            measurements.append(_measurement(state, prompt, "placebo", 1.5, 0.03))

    baseline = {
        (state, prompt): np.full(4, 1.0 if prompt == "task" else 1.5)
        for state in (0, 1)
        for prompt in ("task", "alt")
    }
    P._print_verdict(measurements, baseline_by_prompt=baseline, states=2)


def test_verdict_runs_without_any_prompt_variants() -> None:
    measurements = [
        _measurement(0, "task", "null", 0.0, 0.0),
        _measurement(0, "task", "target", 8.0, 0.3),
    ]
    P._print_verdict(measurements, baseline_by_prompt={}, states=1)


def test_verdict_runs_with_no_grounding_data_at_all() -> None:
    """Defaults must work: the signature is also called from older call sites."""
    measurements = [
        _measurement(0, "task", "null", 0.0, 0.0),
        _measurement(0, "task", "target", 8.0, 0.3),
        _measurement(0, "alt", "target", 9.0, 0.4),
        _measurement(0, "alt", "placebo", 1.0, 0.02),
    ]
    P._print_verdict(measurements)


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok {test.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    main()

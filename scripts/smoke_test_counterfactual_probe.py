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

import json
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
        (state, 0, prompt): np.full(4, 1.0 if prompt == "task" else 1.5)
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


def test_shared_noise_is_reproducible_from_its_seed() -> None:
    """The same seed must give the same tensor, and different cells different ones.

    Before this, the draw came from the global RNG: no run could be reproduced,
    and what looked like run-to-run spread was an unrecorded noise-draw effect.
    """
    import types

    import torch

    policy = types.SimpleNamespace(model=types.SimpleNamespace(config=types.SimpleNamespace(chunk_size=5, max_action_dim=4)))
    device = torch.device("cpu")
    a = P.sample_shared_noise(policy, 1, device, seed=P.noise_seed_for(1000, 0, 0))
    b = P.sample_shared_noise(policy, 1, device, seed=P.noise_seed_for(1000, 0, 0))
    c = P.sample_shared_noise(policy, 1, device, seed=P.noise_seed_for(1000, 0, 1))
    d = P.sample_shared_noise(policy, 1, device, seed=P.noise_seed_for(1000, 1, 0))
    assert a.shape == (1, 5, 4) and a.dtype == torch.float32
    assert torch.equal(a, b), "same seed must reproduce the draw exactly"
    assert not torch.equal(a, c), "a second draw at the same state must differ"
    assert not torch.equal(a, d), "the same draw index at another state must differ"
    seeds = {P.noise_seed_for(1000, s, n) for s in range(50) for n in range(100)}
    assert len(seeds) == 50 * 100, "cell seeds must not collide across states and draws"


def test_footprint_is_measured_over_every_camera() -> None:
    """A recolour visible only in the wrist camera must still count.

    The agentview footprint of a static object never changes between states,
    while the wrist camera can carry most of the perturbation as the arm
    approaches. A footprint read off one camera would miss it.
    """
    h = w = 20
    base = {"agentview": np.zeros((h, w, 3), np.uint8), "wrist": np.zeros((h, w, 3), np.uint8)}
    pert = {k: v.copy() for k, v in base.items()}
    pert["wrist"][:10, :, :] = 200  # half the wrist frame changes, agentview untouched
    stats = P.image_delta_stats_all(base, pert)
    assert stats["cameras"] == 2
    assert stats["per_camera"]["agentview"]["changed_pixel_fraction"] == 0.0
    assert abs(stats["per_camera"]["wrist"]["changed_pixel_fraction"] - 0.5) < 1e-12
    assert abs(stats["changed_pixel_fraction"] - 0.25) < 1e-12, "half of one of two equal frames"
    assert stats["changed_pixel_count"] == 10 * w
    # L2 combines in quadrature: the aggregate equals the single changed camera's norm here.
    assert abs(stats["l2_delta"] - stats["per_camera"]["wrist"]["l2_delta"]) < 1e-9
    # Batched frames (1, H, W, C) are handled the same way.
    batched = {k: v[None] for k, v in base.items()}
    obs = {"pixels": batched}
    frames = P.camera_images(obs)
    assert set(frames) == {"agentview", "wrist"} and frames["wrist"].shape == (h, w, 3)
    identical = P.image_delta_stats_all(frames, frames)
    assert identical["changed_pixel_fraction"] == 0.0 and identical["l2_delta"] == 0.0


def test_recompute_script_rereads_verdicts_from_a_saved_summary() -> None:
    import subprocess
    import tempfile

    run = Path(tempfile.mkdtemp())
    summary = {
        "measurements": _linear_run(alt_sel=2.0, alt_referent_moved=False),
        "prompt_grounding_rel_l2": [0.73, 0.68, 0.79, 0.70],
    }
    (run / "counterfactual_summary.json").write_text(json.dumps(summary))
    script = Path(__file__).resolve().parent / "recompute_counterfactual_decisions.py"
    out = subprocess.run([sys.executable, str(script), str(run)], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    decision = json.loads((run / "decision_metrics.json").read_text())
    assert decision["h1"]["verdict"] == "supported"
    assert decision["h2"]["verdict"].startswith("untestable") and "referent" in decision["h2"]["verdict"]
    assert decision["recomputed_from"].endswith("counterfactual_summary.json")
    assert (run / "h1_cells.csv").exists()


def test_state_advance_uses_successive_actions_of_the_plan() -> None:
    chunk = np.arange(12, dtype=np.float32).reshape(6, 2)
    steps = P.actions_to_step(chunk, 3)
    assert [s.shape for s in steps] == [(1, 2)] * 3
    assert np.array_equal(np.concatenate(steps), chunk[:3]), "must walk the chunk, not repeat one action"
    assert np.array_equal(P.actions_to_step(chunk, 8)[-1][0], chunk[-1]), "past the chunk, hold the last action"


def _m(state, noise, prompt, kind, dose, *, D, S_l2, S_px, A):
    # Two layers of very different scale, both carrying the same ratio, and a
    # relative L2 of D/10: every scale-free summary must reproduce the D ratio.
    return {
        "state_index": state, "noise_index": noise, "prompt": prompt, "kind": kind, "target": kind,
        "dose": dose,
        "pixel": {"changed_pixel_fraction": S_px, "l2_delta": S_l2},
        "latent": {"l2_delta_mean": D, "relative_l2_mean": D / 10.0,
                   "per_layer_l2": {"layers.2.mlp": D * 0.01, "layers.17.mlp": D * 1.99}},
        "action_relative_l2": A,
    }


def _linear_run(*, alt_sel=None, null=0.0, placebo_D=None, swap_D=None, alt_referent_moved=True):
    """Target responds 4x the placebo in D; its footprint is 2x; response is linear in dose.

    With ``alt_sel`` the run also carries a prompt-swap positive control of
    latent response ``swap_D`` (default 16, so the target is 1/4 of it at the
    mean dose) under the task prompt. By default the alternate prompt moves the
    behavioural referent: under it the action is 8x more sensitive to the
    placebo's colour than the target's. ``alt_referent_moved=False`` keeps the
    action anchored on the target, as a swap that only moved words would.
    """
    ms = []
    for state in (0, 1):
        for noise in (0, 1):
            if alt_sel is not None:
                ms.append(_m(state, noise, "task", "prompt_swap", 0.0,
                             D=16.0 if swap_D is None else swap_D, S_l2=0.0, S_px=0.0, A=0.68))
            for prompt in (("task", "alt") if alt_sel is not None else ("task",)):
                ms.append(_m(state, noise, prompt, "null", 0.0, D=null, S_l2=0.0, S_px=0.0, A=0.0))
                for dose in (0.5, 1.0):
                    d_t = 8.0 * dose
                    d_p = (placebo_D if placebo_D is not None else 2.0) * dose
                    a_t, a_p = 0.4 * dose, 0.05 * dose
                    if prompt == "alt":
                        d_p = d_t / alt_sel
                        if alt_referent_moved:
                            a_t, a_p = a_p, a_t
                    ms.append(_m(state, noise, prompt, "target", dose, D=d_t, S_l2=20.0 * dose, S_px=0.02, A=a_t))
                    ms.append(_m(state, noise, prompt, "placebo", dose, D=d_p, S_l2=10.0 * dose, S_px=0.01, A=a_p))
    return ms


def _baselines(g):
    out = {}
    for state in (0, 1):
        for noise in (0, 1):
            base = np.ones(8)
            out[(state, noise, "task")] = base
            out[(state, noise, "alt")] = base * (1 + g / np.sqrt(1.0))  # rel_l2 = g exactly
    return out


def test_decision_metrics_compute_the_paper_quantities_exactly() -> None:
    """Sel, Sel_adj under both normalisers, the CI, and the dose elasticity.

    Fixture: D_t/D_p = 4, footprint ratios 2 (both S), linear dose response.
    So Sel = 4, Sel_adj = 2 for either S, elasticity = 1, and H1 is supported.
    """
    metrics = P.compute_decision_metrics(_linear_run(alt_sel=0.5), baseline_by_prompt=_baselines(0.68))
    h1 = metrics["h1"]
    assert len(h1["cells"]) == 2 * 2 * 2, "state x noise x dose cells under the task prompt"
    pooled = h1["pooled"]
    assert abs(pooled["sel_raw"] - 4.0) < 1e-9
    assert abs(pooled["S_l2_ratio"] - 2.0) < 1e-9 and abs(pooled["S_px_ratio"] - 2.0) < 1e-9
    assert abs(pooled["sel_adj_l2"] - 2.0) < 1e-9 and abs(pooled["sel_adj_px"] - 2.0) < 1e-9
    assert abs(pooled["action_sel"] - 8.0) < 1e-9
    assert h1["spread"]["sel_adj_l2"]["min"] > 1.0 and h1["spread"]["sel_adj_l2"]["sd"] == 0.0
    ci = h1["bootstrap_ci_95"]["sel_adj_l2"]
    assert ci["n_cells"] == 8 and abs(ci["low"] - 2.0) < 1e-9 and abs(ci["high"] - 2.0) < 1e-9
    assert h1["null_floor"]["max"] == 0.0 and h1["null_floor_near_zero"]
    for kind in ("target", "placebo"):
        dr = h1["dose_response"][kind]
        assert dr["monotone_in_dose"] is True
        assert abs(dr["elasticity_mean"] - 1.0) < 1e-9, dr
    assert h1["verdict"] == "supported"

    # Cluster-level spread: 4 (state, draw) clusters, identical by construction.
    cs = h1["cluster_spread"]
    assert cs["n"] == 4 and abs(cs["sel_adj_l2"]["mean"] - 2.0) < 1e-9 and cs["sel_adj_l2"]["sd"] == 0.0
    # Scale-free summaries agree with the D ratio, so the direction is robust.
    rb = h1["robustness"]
    assert abs(rb["sel_rel"] - 4.0) < 1e-9 and abs(rb["sel_layer_geomean"] - 4.0) < 1e-9, rb
    assert rb["layers"] == 2 and rb["layers_above_one"] == 2 and rb["agree_in_direction"] is True
    assert h1["thresholds"]["null_floor_fraction_of_placebo"] == 0.05
    assert metrics["h2"]["thresholds"]["grounding_rel_l2"] == 0.01

    # Positive control: mean target D over doses is 6 (=8*0.75), so 6/16 of the swap response.
    pc = h1["positive_control"]
    assert pc["n"] == 4 and abs(pc["D"] - 16.0) < 1e-9
    assert abs(pc["target_over_positive"] - 6.0 / 16.0) < 1e-9, pc
    assert abs(pc["placebo_over_positive"] - 1.5 / 16.0) < 1e-9, pc

    h2 = metrics["h2"]
    assert abs(h2["grounding"]["mean"] - 0.68) < 1e-9 and h2["grounding"]["n"] == 4
    assert abs(h2["per_prompt"]["task"]["sel_raw"] - 4.0) < 1e-9
    assert abs(h2["per_prompt"]["alt"]["sel_raw"] - 0.5) < 1e-9
    rc = h2["referent_check"]
    assert abs(rc["behavioural_anchor_task"] - 8.0) < 1e-9 and abs(rc["behavioural_anchor_alt"] - 0.125) < 1e-9
    assert rc["referent_moved"] is True
    assert h2["verdict"].startswith("supported")


def test_decision_metrics_verdict_branches() -> None:
    supported = P.compute_decision_metrics(_linear_run(alt_sel=0.5), baseline_by_prompt=_baselines(0.68))
    assert supported["h1"]["verdict"] == "supported"

    # Placebo responds as much as the target: adjusted selectivity 0.5 -> falsified,
    # and the scale-free summaries sit at 1, so the direction does not agree.
    falsified = P.compute_decision_metrics(_linear_run(placebo_D=8.0))
    assert falsified["h1"]["verdict"] == "falsified", falsified["h1"]["pooled"]
    assert falsified["h1"]["robustness"]["agree_in_direction"] is False

    # A null floor that is not near zero invalidates the paired comparison.
    invalid = P.compute_decision_metrics(_linear_run(null=1.0))
    assert invalid["h1"]["verdict"].startswith("invalid")

    # One cell below 1 while the pool stays above: weak, not supported.
    ms = _linear_run()
    for m in ms:
        if m["kind"] == "placebo" and m["state_index"] == 1 and m["noise_index"] == 1 and m["dose"] == 1.0:
            m["latent"]["l2_delta_mean"] = 8.0
    weak = P.compute_decision_metrics(ms)
    assert weak["h1"]["verdict"].startswith("weak"), weak["h1"]["spread"]["sel_adj_l2"]

    # H2 branches: stronger for the same object, weakens without inverting, and inert prompt.
    assert P.compute_decision_metrics(_linear_run(alt_sel=6.0), baseline_by_prompt=_baselines(0.68))["h2"]["verdict"].startswith("falsified")
    assert P.compute_decision_metrics(_linear_run(alt_sel=2.0), baseline_by_prompt=_baselines(0.68))["h2"]["verdict"].startswith("partial")
    assert P.compute_decision_metrics(_linear_run(alt_sel=0.5), baseline_by_prompt=_baselines(0.001))["h2"]["verdict"].startswith("untestable")
    assert P.compute_decision_metrics(_linear_run())["h2"]["verdict"].startswith("untestable")

    # The manipulation check: a swap that changed the action (g large) but left it
    # anchored on the target object moved words, not the referent. The old code
    # read this exact pattern as "partial"; it is untestable.
    unmoved = P.compute_decision_metrics(
        _linear_run(alt_sel=2.0, alt_referent_moved=False), baseline_by_prompt=_baselines(0.73)
    )
    assert unmoved["h2"]["referent_check"]["referent_moved"] is False
    assert abs(unmoved["h2"]["referent_check"]["behavioural_anchor_alt"] - 8.0) < 1e-9
    assert unmoved["h2"]["verdict"].startswith("untestable"), unmoved["h2"]["verdict"]
    assert "referent" in unmoved["h2"]["verdict"]

    # Grounding values may be supplied directly (recomputation from a saved summary).
    direct = P.compute_decision_metrics(_linear_run(alt_sel=0.5), grounding_values=[0.7, 0.75])
    assert direct["h2"]["grounding"]["n"] == 2 and direct["h2"]["verdict"].startswith("supported")

    # Without an alternate prompt there is no positive control, and that is reported as such.
    assert falsified["h1"]["positive_control"]["n"] == 0
    assert falsified["h1"]["positive_control"]["target_over_positive"] is None

    # Printing must not crash on any branch.
    for metrics in (supported, falsified, invalid, weak):
        P._print_decision_metrics(metrics)


def test_nomination_rejects_a_one_off_however_selective() -> None:
    """A feature seen in 2 of 80 cells must not be nominated for tracing.

    Selectivity is target/placebo, and the placebo floor is tiny, so a single
    large delta can top the ranking. The first real run nominated exactly such
    a feature -- 2 cells, 201x -- and tracing it would have chased a fluke.
    """
    import subprocess
    import tempfile

    layer = "paligemma_with_expert.gemma_expert.model.layers.2.mlp"
    n_features, consistent, fluke = 64, 11, 22
    fluke_cells = {((0, 0.5, "task"), 0), ((0, 0.5, "task"), 1)}

    def accumulator(cell=None):
        acc = P.LatentAccumulator()
        for step in range(10):
            vector = np.full(n_features, 0.02)
            if cell is not None:
                vector[consistent] = 2.0
                if (cell, step) in fluke_cells:
                    vector[fluke] = 40.0
            acc.max[(layer, step)] = vector
            acc.mean[(layer, step)] = vector
        return acc

    baseline = accumulator()
    rows = []
    for state in (0, 1):
        for dose in (0.5, 1.0):
            for prompt in ("task", "alt"):
                for condition in ("target", "placebo"):
                    other = accumulator((state, dose, prompt)) if condition == "target" else baseline
                    for row in P.latent_delta_rows(baseline, other, condition=condition, top_features=25):
                        row.update({"state_index": state, "dose": dose, "target": condition, "prompt": prompt})
                        rows.append(row)

    run = Path(tempfile.mkdtemp())
    P.write_csv(run / "latent_deltas.csv", rows)
    (run / "counterfactual_summary.json").write_text(
        json.dumps({"config": {"num_inference_steps": 10}})
    )
    script = Path(__file__).resolve().parent / "report_pi05_counterfactual_features.py"
    # The fixture lives at layer 2; disable the depth rule so only consistency is under test.
    out = subprocess.run(
        [sys.executable, str(script), str(run), "--min-trace-layer", "0"], capture_output=True, text=True
    ).stdout

    import csv as _csv

    ranked = {int(r["feature"]): r for r in _csv.DictReader((run / "candidate_features.csv").open())}
    assert float(ranked[fluke]["selectivity"]) > float(ranked[consistent]["selectivity"]), (
        "the fluke should still win on raw selectivity; that is why the filter is needed"
    )
    # Nomination is decided on the task prompt's cells: 2 states x 2 doses x 10 steps.
    assert ranked[fluke]["layer_cells"] == "40", "cell counts must be reported against the total"
    nomination = out[out.index("Nominated"):] if "Nominated" in out else ""
    assert f"F{consistent}" in nomination, nomination
    assert f"F{fluke}" not in nomination, "a 2/80 feature must not be nominated"


def test_vision_tower_remap_only_fires_when_the_model_wants_it() -> None:
    """The compat patch must not break a load that would otherwise succeed.

    It flattens `.vision_tower.vision_model.*` to `.vision_tower.*`. Newer
    LeRobot builds nest SigLIP under `vision_model` exactly as the checkpoint
    does, so flattening there makes load_state_dict raise -- and LeRobot
    swallows that, leaving the policy on random vision weights. Every feature
    discovered or circuit traced under those weights is meaningless.
    """
    import types

    import torch
    from lerobot.policies.pi05 import modeling_pi05

    from train_pi05_transcoders import patch_pi05_checkpoint_key_compat

    nested = (
        "model.paligemma_with_expert.paligemma.model.vision_tower"
        ".vision_model.embeddings.patch_embedding.weight"
    )
    flat = nested.replace(".vision_model.", ".")

    def remap_with(model_keys):
        modeling_pi05.PI05Policy._fix_pytorch_state_dict_keys = lambda self, sd, mc: sd
        patch_pi05_checkpoint_key_compat()
        fake = types.SimpleNamespace(state_dict=lambda: {key: None for key in model_keys})
        out = modeling_pi05.PI05Policy._fix_pytorch_state_dict_keys(
            fake, {nested: torch.zeros(1)}, None
        )
        return next(iter(out))

    assert remap_with([nested]) == nested, (
        "a model that nests vision_model must keep the checkpoint's keys untouched"
    )
    assert remap_with([flat]) == flat, "an older flat-layout model must still get the remap"


def _two_scale_run(tmp_dir):
    """Two layers whose response scales differ 40x, each with one standout.

    This is the shape of the real data: L2 deltas are ~0.1 and L17 deltas ~4.
    A fair ranking must be able to nominate either standout.
    """
    import numpy as _np

    specs = {2: (0.10, 6), 17: (4.00, 9)}
    n_features = 64

    def build(boost, scale_mult):
        acc = P.LatentAccumulator()
        for layer, (scale, star) in specs.items():
            name = f"paligemma_with_expert.gemma_expert.model.layers.{layer}.mlp"
            for step in range(10):
                rng = _np.random.default_rng(layer * 100 + step)
                vector = _np.abs(rng.normal(scale * scale_mult, scale * 0.15, n_features))
                if boost:
                    vector[star] = scale * 5
                acc.max[(name, step)] = vector
                acc.mean[(name, step)] = vector
        return acc

    baseline, target, placebo = build(False, 0.0), build(True, 1.0), build(False, 0.25)
    rows = []
    for state in (0, 1):
        for dose in (0.5, 1.0):
            for condition, other in (("target", target), ("placebo", placebo)):
                for row in P.latent_delta_rows(baseline, other, condition=condition, top_features=25):
                    row.update({"state_index": state, "dose": dose, "target": condition, "prompt": "task"})
                    rows.append(row)
    P.write_csv(tmp_dir / "latent_deltas.csv", rows)
    (tmp_dir / "counterfactual_summary.json").write_text(
        json.dumps({"config": {"num_inference_steps": 10}})
    )
    return specs


def test_ranking_is_fair_across_layers_of_different_scale() -> None:
    """Raw selectivity is floor-biased; the layer-standardised score is not.

    Selectivity divides by the layer's own placebo floor, and those floors span
    an order of magnitude across depth, so an identical relative response scores
    very differently depending on where it sits. Standardising within layer
    removes that, letting deep features compete.
    """
    import csv as _csv
    import subprocess
    import tempfile

    run = Path(tempfile.mkdtemp())
    _two_scale_run(run)
    script = Path(__file__).resolve().parent / "report_pi05_counterfactual_features.py"
    out = subprocess.run(
        [sys.executable, str(script), str(run), "--min-consistency", "0.1"],
        capture_output=True, text=True,
    ).stdout

    ranked = {
        (int(r["layer_index"]), int(r["feature"])): r
        for r in _csv.DictReader((run / "candidate_features.csv").open())
    }
    shallow, deep = ranked[(2, 6)], ranked[(17, 9)]
    assert abs(float(shallow["target_z"]) - float(deep["target_z"])) < 0.5, (
        "two equally-standout features must score alike regardless of layer scale"
    )
    assert float(deep["target_mean_abs_delta"]) > 20 * float(shallow["target_mean_abs_delta"]), (
        "the fixture must actually have very different absolute scales"
    )
    nomination = out[out.index("Nominated"):] if "Nominated" in out else ""
    assert "L17" in nomination, f"a deep feature must be nominable\n{nomination}"


def test_exact_placebo_from_the_store_overrides_the_top_k_bound() -> None:
    """A feature the placebo's top-K missed can still be placebo-responsive.

    Fixture: feature 9 responds 2.0 to the target in every cell and 1.8 to the
    placebo -- but the placebo's top-K is filled by 25 larger responders, so on
    the CSV alone feature 9 reads 'placebo below top-K' and is nominated. The
    store knows its exact placebo response, and the nominator must reject it,
    while nominating feature 11 (target 2.0, placebo 0.0).
    """
    import csv as _csv
    import subprocess
    import tempfile

    from pi05_mi.counterfactual_store import DeltaStore

    layer = "paligemma_with_expert.gemma_expert.model.layers.6.mlp"
    n_features, steps = 64, 10
    sneaky, clean, decoys = 9, 11, list(range(30, 55))  # 25 decoys fill the placebo top-K

    def code(kind):
        acc = P.LatentAccumulator()
        for step in range(steps):
            vector = np.zeros(n_features, dtype=np.float32)
            if kind != "baseline":
                vector[sneaky] = 2.0 if kind == "target" else 1.8
                vector[clean] = 2.0 if kind == "target" else 0.0
                if kind == "placebo":
                    vector[decoys] = 5.0
            acc.max[(layer, step)] = vector
            acc.mean[(layer, step)] = vector
        return acc

    baseline, target, placebo = code("baseline"), code("target"), code("placebo")
    run = Path(tempfile.mkdtemp())
    store = DeltaStore.create(run / "latents", layer_names=[layer], num_steps=steps, num_features=n_features)
    rows = []
    for state in (0, 1):
        for dose in (0.5, 1.0):
            for condition, other in (("target", target), ("placebo", placebo)):
                for row in P.latent_delta_rows(baseline, other, condition=condition, top_features=25):
                    row.update({"state_index": state, "noise_index": 0, "dose": dose, "target": condition, "prompt": "task"})
                    rows.append(row)
        store.write_block(
            state=state, noise=0, prompt="task", baseline_max=baseline.max,
            deltas={
                ("target", 0.5): P.full_delta(baseline, target), ("target", 1.0): P.full_delta(baseline, target),
                ("placebo", 0.5): P.full_delta(baseline, placebo), ("placebo", 1.0): P.full_delta(baseline, placebo),
            },
        )
    P.write_csv(run / "latent_deltas.csv", rows)
    (run / "counterfactual_summary.json").write_text(json.dumps({"config": {"num_inference_steps": steps}}))
    script = Path(__file__).resolve().parent / "report_pi05_counterfactual_features.py"
    out = subprocess.run(
        [sys.executable, str(script), str(run), "--min-trace-layer", "4"], capture_output=True, text=True
    )
    assert out.returncode == 0, out.stdout + out.stderr

    ranked = {int(r["feature"]): r for r in _csv.DictReader((run / "candidate_features.csv").open())}
    assert ranked[sneaky]["placebo_measured"] == "False", "on the CSV alone the placebo looks unmeasured"
    assert abs(float(ranked[sneaky]["placebo_exact_mean_abs_delta"]) - 1.8) < 1e-6, "the store knows better"
    assert float(ranked[clean]["placebo_exact_mean_abs_delta"]) == 0.0
    nominated = json.loads((run / "nominated_targets.json").read_text())
    keys = [row["feature_key"] for row in nominated["nominated"]]
    assert f"F{clean}" in " ".join(keys), keys
    assert f"F{sneaky}" not in " ".join(keys), f"a placebo-responsive feature must not be nominated: {keys}"
    assert nominated["criteria"]["rejected"].get("placebo-responsive", 0) >= 1
    assert "exact placebo" in nominated["criteria"]["placebo_rule"]

    # Shallow layers are excluded by the pre-registered depth rule.
    out = subprocess.run(
        [sys.executable, str(script), str(run), "--min-trace-layer", "7"], capture_output=True, text=True
    )
    nominated = json.loads((run / "nominated_targets.json").read_text())
    assert nominated["nominated"] == [] and nominated["criteria"]["rejected"].get("too shallow", 0) >= 1


def test_elevated_placebo_disqualifies_a_candidate() -> None:
    """A feature that also responds to the control object is not selective."""
    import numpy as _np

    from report_pi05_counterfactual_features import candidate_features

    layer = "paligemma_with_expert.gemma_expert.model.layers.3.mlp"
    rows = []
    for state in (0, 1):
        for condition, boosted in (("target", True), ("placebo", True)):
            for step in range(4):
                rows.append({
                    "condition": condition, "layer": layer, "denoise_step": step,
                    "state_index": state, "dose": 1.0, "features": 32,
                    "top_feature_ids": [5, 6, 7],
                    "top_feature_deltas": [9.0 if boosted else 0.1, 0.2, 0.1],
                })
    scored = {r["feature"]: r for r in candidate_features(rows, min_cells=2)}
    assert scored[5]["placebo_measured"] is True
    assert scored[5]["placebo_z"] is not None


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok {test.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    main()

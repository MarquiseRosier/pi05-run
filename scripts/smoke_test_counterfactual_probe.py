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
        assert os.environ["MUJOCO_GL"] == "egl"
        assert os.environ["PYOPENGL_PLATFORM"] == "egl"

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


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok {test.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    main()

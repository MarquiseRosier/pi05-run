#!/usr/bin/env python
"""Measure which Pi0.5 transcoder features respond to one object's appearance.

Design: paired same-state counterfactual, not a rollout comparison.

At a frozen simulator state the scene is rendered twice -- once as-is, once
with a single object recolored -- and both observations are pushed through the
policy with the *same* flow-matching noise. Physics is never stepped between
the two renders, so the observations differ only in that object's pixels and
any activation difference is attributable to it.

Running this closed-loop instead would be invalid: a recolor nudges the action,
the trajectory diverges, and within a few steps the activation delta reflects
different robot states rather than the color.

Three forward passes per state:

* ``baseline``  -- unperturbed render
* ``null``      -- the *same* unperturbed render again, same noise. Its delta is
                   the nondeterminism floor. A perturbation delta is only
                   meaningful as the excess over this.
* ``perturbed`` -- the recolored render

Use ``--list-objects`` first to see what the scene actually contains.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

def _harden_environment() -> None:
    """Neutralise host env vars that break a headless LIBERO import.

    Colab exports ``MPLBACKEND=module://matplotlib_inline.backend_inline``,
    which is only meaningful inside the kernel process. LIBERO imports
    matplotlib while building its env wrapper, so an inherited inline backend
    aborts the import in any subprocess.
    """
    if "inline" in os.environ.get("MPLBACKEND", ""):
        os.environ["MPLBACKEND"] = "Agg"
    os.environ.setdefault("MPLBACKEND", "Agg")

    # Headless offscreen rendering wants EGL, but only Linux has it: importing
    # mujoco with MUJOCO_GL=egl on macOS raises outright. Default only where the
    # backend exists, and never override an explicit choice.
    if sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    if os.environ.get("MUJOCO_GL"):
        os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])


_harden_environment()

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pi05_mi.counterfactual_store import DeltaStore, sort_layer_names  # noqa: E402
from pi05_mi.langfuse_tracing import make_langfuse_tracer, summarize_action_tensor  # noqa: E402
from pi05_mi.patch_pi05 import Pi05TranscoderContext, install_pi05_action_expert_wrappers  # noqa: E402
from pi05_mi.pi05_weights import assert_weights_loaded, install_load_recorder  # noqa: E402
from pi05_mi.provenance import collect_provenance  # noqa: E402
from pi05_mi.scene_perturbation import (  # noqa: E402
    blend_geom_color,
    find_objects,
    image_delta_stats,
    list_scene_objects,
    resolve_geom_ids,
    resolve_mj_model,
    set_geom_color,
    shift_geom_hue,
)
from pi05_mi.transcoders import TimeConditionedTranscoder, TimeConditionedTranscoderConfig  # noqa: E402

DEFAULT_OUTPUT_DIR = Path("outputs/probes/counterfactual")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy-path", default="lerobot/pi05_libero_finetuned")
    parser.add_argument("--checkpoint", type=Path, help="Transcoder checkpoint. Required unless --list-objects.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--prompt", default=None, help="Override the task language prompt.")
    parser.add_argument(
        "--alt-prompt",
        default=None,
        help=(
            "A second prompt to repeat every measurement under, on pixel-identical images. "
            "Defaults to a sibling task's language that refers to the placebo's location, which "
            "separates task relevance from screen position. Use --no-alt-prompt to skip."
        ),
    )
    parser.add_argument("--no-alt-prompt", dest="use_alt_prompt", action="store_false", default=True)

    parser.add_argument("--list-objects", action="store_true", help="Print the scene's objects and exit.")
    parser.add_argument(
        "--target",
        default=None,
        help=(
            "Regex matching the body name to perturb. Defaults to the task's first "
            "obj_of_interest, read from its BDDL."
        ),
    )
    parser.add_argument(
        "--placebo-target",
        default=None,
        help=(
            "A second object to perturb in a separate pass. Comparing its response to the "
            "primary target separates 'features track this object' from 'features track any pixel change'. "
            "Defaults to another instance of the same object type when the scene has one."
        ),
    )
    parser.add_argument(
        "--perturbation",
        choices=["blend", "set", "hue"],
        default="blend",
        help="blend: interpolate toward --color by --dose (monotone). set: absolute color. hue: rotate hue.",
    )
    parser.add_argument("--color", default="1.0,0.2,0.1", help="Target RGB for blend/set.")
    parser.add_argument(
        "--dose",
        default="0.5,1.0",
        help="Comma-separated doses in [0,1] for blend, or hue shifts for hue. Each is measured separately.",
    )
    parser.add_argument("--detach-material", action="store_true", help="Also strip the texture from the target.")

    parser.add_argument("--states", type=int, default=2, help="How many distinct sim states to measure at.")
    parser.add_argument("--state-stride", type=int, default=5, help="Env steps between measured states.")
    parser.add_argument(
        "--noise-samples",
        type=int,
        default=1,
        help=(
            "Independent flow-matching noise draws per state. Every condition at a state is measured "
            "under each draw, so draws are exchangeable replicates and give the error bound."
        ),
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=None,
        help="Seed for the shared noise draws. Defaults to --seed, so a run is reproducible bit for bit.",
    )
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--top-features", type=int, default=25, help="Top changed features to record per layer.")
    parser.add_argument(
        "--no-full-deltas",
        dest="full_deltas",
        action="store_false",
        default=True,
        help=(
            "Skip writing the full per-feature delta store (latents/). The store is what makes the "
            "placebo response exact and lets a traced circuit be scored at full coverage."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--policy-dtype", default="bfloat16")
    parser.add_argument("--save-images", action="store_true", default=True)
    parser.add_argument("--no-save-images", dest="save_images", action="store_false")
    parser.add_argument(
        "--trace-langfuse",
        action="store_true",
        help=(
            "Emit one Langfuse span per forward pass with the batch shapes and images. "
            "Requires PI05_LANGFUSE_TRACE=1 plus credentials; degrades to a no-op otherwise. "
            "The local observation_shapes.json is written either way."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------- latent capture


class LatentAccumulator:
    """Collect transcoder latents from one forward pass, reduced over tokens.

    A full latent is (batch, tokens, features) per layer per denoise step.
    Keeping all of it for 18 layers x 10 steps is needlessly large, so each is
    reduced to a per-feature mean and max over the token axis. That preserves
    "which features fired" -- the question here -- at a fraction of the memory.
    """

    def __init__(self) -> None:
        self.mean: dict[tuple[str, int], np.ndarray] = {}
        self.max: dict[tuple[str, int], np.ndarray] = {}
        self._step_index: dict[str, int] = defaultdict(int)
        self.enabled = False

    def reset(self) -> None:
        self.mean.clear()
        self.max.clear()
        self._step_index.clear()

    def callback(self, name: str, layer_index: int, latent: torch.Tensor, timestep: torch.Tensor) -> None:
        if not self.enabled:
            return
        step = self._step_index[name]
        self._step_index[name] += 1
        with torch.no_grad():
            flat = latent.detach().float().reshape(-1, latent.shape[-1])
            self.mean[(name, step)] = flat.mean(dim=0).cpu().numpy()
            self.max[(name, step)] = flat.max(dim=0).values.cpu().numpy()


def latent_delta_rows(
    baseline: LatentAccumulator,
    other: LatentAccumulator,
    *,
    condition: str,
    top_features: int,
) -> list[dict[str, Any]]:
    """Per (layer, denoise step) difference between two forward passes."""
    rows = []
    for key in sorted(baseline.max.keys() & other.max.keys(), key=lambda k: (k[0], k[1])):
        layer_name, step = key
        base_max = baseline.max[key]
        other_max = other.max[key]
        diff = other_max - base_max
        abs_diff = np.abs(diff)
        base_norm = float(np.linalg.norm(base_max))
        order = np.argsort(-abs_diff)[:top_features]
        rows.append(
            {
                "condition": condition,
                "layer": layer_name,
                "denoise_step": step,
                "features": int(base_max.size),
                "l2_delta": float(np.linalg.norm(diff)),
                "relative_l2": None if base_norm == 0 else float(np.linalg.norm(diff) / base_norm),
                "max_abs_delta": float(abs_diff.max()),
                "mean_abs_delta": float(abs_diff.mean()),
                "changed_features": int((abs_diff > 1e-6).sum()),
                "baseline_active": int((base_max > 0).sum()),
                "other_active": int((other_max > 0).sum()),
                "top_feature_ids": order.tolist(),
                "top_feature_deltas": [float(value) for value in diff[order]],
            }
        )
    return rows


def full_delta(baseline: LatentAccumulator, other: LatentAccumulator) -> dict[tuple[str, int], np.ndarray]:
    """Signed per-feature difference of the reduced code, for every (layer, step)."""
    return {
        key: (other.max[key] - baseline.max[key]).astype(np.float32)
        for key in baseline.max.keys() & other.max.keys()
    }


def summarize_latent_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"entries": 0}
    l2 = np.array([row["l2_delta"] for row in rows], dtype=np.float64)
    rel = np.array([row["relative_l2"] for row in rows if row["relative_l2"] is not None], dtype=np.float64)
    changed = np.array([row["changed_features"] for row in rows], dtype=np.float64)
    return {
        "entries": len(rows),
        "l2_delta_mean": float(l2.mean()),
        "l2_delta_max": float(l2.max()),
        "relative_l2_mean": float(rel.mean()) if rel.size else None,
        "relative_l2_max": float(rel.max()) if rel.size else None,
        "changed_features_mean": float(changed.mean()),
    }


# ---------------------------------------------------------------- env plumbing


@dataclass
class Harness:
    vec_env: Any
    inner_env: Any
    prompt: str
    preprocess_observation: Callable[[dict], dict]
    # Absent on a --list-objects pass, which needs the scene but not the policy.
    policy: Any = None
    preprocessor: Any = None
    env_preprocessor: Any = None


def ensure_libero_config() -> Path | None:
    """Write LIBERO's config.yaml so importing it cannot block on input().

    ``libero/libero/__init__.py`` prompts interactively ("Do you want to specify
    a custom path for the dataset folder?") the first time it is imported if
    ``~/.libero/config.yaml`` is missing. In any non-interactive process that
    raises ``EOFError`` before a single useful line runs.

    Writing the same defaults LIBERO would have written makes the import silent.
    The keys mirror ``get_default_path_dict`` in that module.
    """
    import importlib.util

    config_dir = Path(os.environ.get("LIBERO_CONFIG_PATH") or (Path.home() / ".libero"))
    config_file = config_dir / "config.yaml"
    if config_file.exists():
        return config_file

    # find_spec on the top-level package does not execute the submodule that
    # holds the prompt, so this is safe to call before the real import.
    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.origin:
        return None
    benchmark_root = Path(spec.origin).parent / "libero"

    try:
        import yaml
    except ImportError:
        return None

    payload = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(benchmark_root.parent / "datasets"),
        "assets": str(benchmark_root / "assets"),
    }
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file.write_text(yaml.safe_dump(payload), encoding="utf-8")
    print(f"wrote LIBERO config {config_file} (benchmark_root={benchmark_root})", flush=True)
    return config_file


def build_harness(args: argparse.Namespace, *, load_policy: bool = True) -> Harness:
    """Build the LIBERO env, and the policy stack only when it will be used.

    Listing the scene's objects needs the simulator but not Pi0.5, and loading
    the policy means a multi-GB download, so the discovery pass skips it.
    """
    ensure_libero_config()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.scripts.lerobot_eval import preprocess_observation

    env_cfg = LiberoEnv(task=args.suite, task_ids=[args.task_id])
    print(f"building LIBERO env {args.suite} task {args.task_id}", flush=True)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    vec_env = envs[args.suite][args.task_id]
    inner_env = vec_env.envs[0] if hasattr(vec_env, "envs") else vec_env

    harness = Harness(
        vec_env=vec_env,
        inner_env=inner_env,
        prompt=args.prompt or "",
        preprocess_observation=preprocess_observation,
    )
    if not load_policy:
        print("skipping policy load (discovery pass)", flush=True)
        return harness

    print(f"loading policy {args.policy_path}", flush=True)
    policy_cfg = PreTrainedConfig.from_pretrained(
        args.policy_path,
        cache_dir=os.environ.get("HF_HUB_CACHE"),
        local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1",
    )
    policy_cfg.pretrained_path = Path(args.policy_path)
    policy_cfg.device = args.device
    policy_cfg.dtype = args.policy_dtype
    policy_cfg.compile_model = False
    policy_cfg.gradient_checkpointing = False
    policy_cfg.n_action_steps = 10

    # LeRobot swallows a failed state-dict load and hands back random weights;
    # record what load_state_dict reported and refuse to measure on anything
    # but a complete load.
    install_load_recorder()
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map={})
    assert_weights_loaded(policy, source=str(args.policy_path))
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.policy_path,
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": {}},
        },
    )
    harness.policy = policy
    harness.preprocessor = preprocessor
    harness.env_preprocessor, _ = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)
    return harness


_INSTANCE_RE = re.compile(r"^(?P<base>.+?)_(?P<index>\d+)(?P<suffix>_.*)?$")


def read_objects_of_interest(harness: Harness) -> list[str]:
    """Names the task's BDDL marks as ``obj_of_interest``.

    LIBERO states the task-relevant objects in the problem definition, so the
    probe can choose its target from the task itself rather than making the
    caller look them up.
    """
    bddl_path = getattr(harness.inner_env, "_task_bddl_file", None)
    if not bddl_path or not Path(bddl_path).exists():
        return []
    text = Path(bddl_path).read_text(errors="replace")
    match = re.search(r"\(:obj_of_interest(.*?)\)", text, re.S)
    if not match:
        return []
    return [token for token in match.group(1).split() if token]


def auto_select_targets(harness: Harness, mj_model: Any) -> tuple[str | None, str | None, list[str]]:
    """Pick a target and a matched placebo without the caller naming them.

    Target: the first ``obj_of_interest`` that resolves to a body in the scene.

    Placebo: another instance of the *same object type* when the scene has one.
    In libero_spatial that is the second identical black bowl, which is the
    tightest possible control -- same mesh, same colour, same size, differing
    only in position and task relevance. Falling back to a different object
    would confound "different object" with "different pixels".
    """
    notes: list[str] = []
    bodies = [obj.body_name for obj in list_scene_objects(mj_model, task_objects_only=True)]

    target_body: str | None = None
    for name in read_objects_of_interest(harness):
        matches = find_objects(mj_model, f"^{re.escape(name)}(_|$)")
        if matches:
            target_body = matches[0].body_name
            notes.append(f"target {target_body!r} from the task's obj_of_interest ({name})")
            break

    if target_body is None:
        if not bodies:
            return None, None, ["no task-like bodies found in the scene"]
        target_body = bodies[0]
        notes.append(f"no obj_of_interest matched a body; falling back to {target_body!r}")

    placebo_body: str | None = None
    parsed = _INSTANCE_RE.match(target_body)
    if parsed:
        base, index, suffix = parsed.group("base"), parsed.group("index"), parsed.group("suffix") or ""
        for candidate in bodies:
            other = _INSTANCE_RE.match(candidate)
            if (
                other
                and other.group("base") == base
                and other.group("index") != index
                and (other.group("suffix") or "") == suffix
            ):
                placebo_body = candidate
                notes.append(
                    f"placebo {placebo_body!r}: another instance of the same object type, "
                    "so the perturbation is matched and only task relevance differs"
                )
                break

    if placebo_body is None:
        others = [name for name in bodies if name != target_body]
        if others:
            placebo_body = others[0]
            notes.append(f"placebo {placebo_body!r}: no sibling instance, using a different object")

    to_pattern = lambda name: f"^{re.escape(name)}$"  # noqa: E731 - exact, so it cannot match a sibling
    return (
        to_pattern(target_body),
        to_pattern(placebo_body) if placebo_body else None,
        notes,
    )


def _init_region_of(bddl_text: str, body_name: str) -> str | None:
    """Where a BDDL places a given object at reset."""
    base = re.sub(r"_main$|_cabinet_.*$|_burner.*$|_button$", "", body_name)
    match = re.search(rf"\(On\s+{re.escape(base)}\s+(\S+?)\)", bddl_text)
    return match.group(1) if match else None


def find_prompt_referring_to(harness: Harness, placebo_body: str) -> tuple[str, str] | None:
    """Find a sibling task whose prompt names the placebo's location.

    This is the control that separates task relevance from screen position. The
    scene is unchanged and both bowls stay exactly where they are; only the
    language changes, so that the *other* bowl becomes the object the prompt
    refers to. Every pixel is identical between the two runs.

    In libero_spatial the suite is built by moving the same target object to
    different regions, so some sibling task's target sits where our placebo
    sits, and that task's language is the prompt we need.
    """
    bddl_path = getattr(harness.inner_env, "_task_bddl_file", None)
    if not bddl_path or not Path(bddl_path).exists():
        return None
    here = Path(bddl_path)
    text = here.read_text(errors="replace")
    placebo_region = _init_region_of(text, placebo_body)
    if not placebo_region:
        return None

    for sibling in sorted(here.parent.glob("*.bddl")):
        if sibling == here:
            continue
        sibling_text = sibling.read_text(errors="replace")
        interest = re.search(r"\(:obj_of_interest(.*?)\)", sibling_text, re.S)
        language = re.search(r"\(:language ([^)]*)\)", sibling_text)
        if not interest or not language:
            continue
        names = interest.group(1).split()
        if not names:
            continue
        if _init_region_of(sibling_text, names[0]) == placebo_region:
            return language.group(1).strip(), sibling.stem
    return None


def rerender_observation(harness: Harness) -> dict[str, Any]:
    """Re-render the cameras at the *current* sim state without stepping physics.

    This is the operation the whole design depends on: it must reflect a model
    mutation made a moment ago while leaving qpos/qvel untouched.
    """
    inner = harness.inner_env
    # force_update is essential. robosuite caches observable values and only
    # refreshes them inside step(); without it every "re-render" hands back the
    # same frame, so a perturbation silently appears to change nothing. Its own
    # docstring names this case: grabbing observations after setting simulation
    # state directly, without stepping.
    raw = inner._env.env._get_observations(force_update=True)
    formatted = inner._format_raw_obs(raw)
    return _add_batch_axis(formatted)


def _add_batch_axis(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _add_batch_axis(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value[None, ...]
    if isinstance(value, (int, float)):
        return np.asarray([value])
    return value


def observation_to_batch(harness: Harness, observation: dict[str, Any]) -> dict[str, Any]:
    obs = harness.preprocess_observation(observation)
    obs["task"] = [harness.prompt]
    batch = harness.env_preprocessor(obs)
    return harness.preprocessor(batch)


def load_transcoders(checkpoint_path: Path, device: torch.device) -> dict[str, TimeConditionedTranscoder]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    transcoders: dict[str, TimeConditionedTranscoder] = {}
    for name, raw_config in checkpoint["configs"].items():
        transcoder = TimeConditionedTranscoder(TimeConditionedTranscoderConfig(**raw_config))
        transcoder.load_state_dict(checkpoint["state_dicts"][name])
        transcoder.to(device=device, dtype=torch.float32).eval()
        for parameter in transcoder.parameters():
            parameter.requires_grad_(False)
        transcoders[name] = transcoder
    return transcoders


def noise_seed_for(noise_seed: int, state_index: int, noise_index: int) -> int:
    """One seed per (state, draw) so any single cell can be regenerated alone."""
    return int(noise_seed) + 1009 * int(state_index) + int(noise_index)


def sample_shared_noise(
    policy: Any, batch_size: int, device: torch.device, *, seed: int | None = None
) -> torch.Tensor:
    """Draw the flow-matching noise both passes of a pair will share.

    Seeded explicitly: without this the draw comes from the global RNG, the run
    cannot be reproduced, and any run-to-run spread is really an unrecorded
    noise-draw effect.
    """
    config = policy.model.config
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
    return torch.normal(
        mean=0.0,
        std=1.0,
        size=(batch_size, config.chunk_size, config.max_action_dim),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )


def actions_to_step(chunk: np.ndarray, stride: int) -> list[np.ndarray]:
    """The first ``stride`` actions of the unperturbed chunk, in order.

    Advancing with the policy's own plan puts the next measurement at a state
    the policy would actually visit; repeating one action does not.
    """
    chunk = np.asarray(chunk, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[0] == 0:
        raise ValueError(f"Expected an action chunk shaped (steps, dims), got {chunk.shape}")
    return [chunk[min(i, chunk.shape[0] - 1)][None, ...] for i in range(int(stride))]


# ---------------------------------------------------------------- perturbation


def apply_perturbation(model: Any, args: argparse.Namespace, target: str, dose: float):
    color = [float(part) for part in args.color.split(",")]
    if args.perturbation == "blend":
        return blend_geom_color(model, target, color, dose, label=f"{target}@blend{dose:g}")
    if args.perturbation == "set":
        return set_geom_color(
            model, target, color, detach_material=args.detach_material, label=f"{target}@set"
        )
    return shift_geom_hue(model, target, dose, label=f"{target}@hue{dose:g}")


def save_image(path: Path, array: np.ndarray) -> None:
    arr = np.asarray(array)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.moveaxis(arr, 0, -1)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255 if arr.max() <= 1.5 else arr, 0, 255).astype(np.uint8)
    try:
        import cv2

        cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    except Exception:
        from PIL import Image

        Image.fromarray(arr).save(path)


def first_camera_image(observation: dict[str, Any]) -> np.ndarray:
    """Return one camera frame as (H, W, C), without the batch axis.

    Observations carry a leading batch dim for the policy; the image metrics
    want a plain frame, and leaving the extra axis on made "changed pixels"
    count channel values instead of pixels.
    """
    pixels = observation["pixels"]
    image = np.asarray(next(iter(pixels.values())))
    while image.ndim > 3 and image.shape[0] == 1:
        image = image[0]
    return image


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _stringify(row.get(key)) for key in fieldnames})


def _stringify(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value))
    return value


# ---------------------------------------------------------------- main


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    harness = build_harness(args, load_policy=not args.list_objects)

    env_observation, _info = harness.vec_env.reset(seed=[args.seed])
    if not harness.prompt:
        harness.prompt = _infer_prompt(harness, env_observation)
    print(f"prompt: {harness.prompt!r}", flush=True)

    mj_model = resolve_mj_model(harness.inner_env._env)

    objects = list_scene_objects(mj_model, task_objects_only=True)
    object_report = [
        {
            "body_name": obj.body_name,
            "body_id": obj.body_id,
            "geoms": [
                {
                    "geom_id": geom.geom_id,
                    "geom_name": geom.geom_name,
                    "rgba": [round(float(c), 4) for c in geom.rgba],
                    "uses_material": geom.uses_material,
                }
                for geom in obj.geoms
            ],
        }
        for obj in objects
    ]
    (args.output_dir / "scene_objects.json").write_text(
        json.dumps(object_report, indent=2, default=_json_default), encoding="utf-8"
    )
    print(f"\nscene contains {len(objects)} task-like objects:", flush=True)
    for obj in objects:
        print(f"  {obj.body_name:<40} geoms={len(obj.geoms)}", flush=True)

    if args.list_objects:
        print(f"\nWrote {args.output_dir / 'scene_objects.json'}", flush=True)
        harness.vec_env.close()
        return

    if args.checkpoint is None:
        raise SystemExit("--checkpoint is required unless --list-objects")

    if args.target is None:
        auto_target, auto_placebo, notes = auto_select_targets(harness, mj_model)
        if auto_target is None:
            raise SystemExit(
                "Could not choose a target automatically. Rerun with --list-objects and pass --target."
            )
        args.target = auto_target
        if args.placebo_target is None:
            args.placebo_target = auto_placebo
        print("\nauto-selected objects:", flush=True)
        for note in notes:
            print(f"  {note}", flush=True)

    def report_target(label: str, pattern: str) -> tuple[str, ...]:
        """Name the bodies a pattern matched.

        LIBERO bodies carry dozens of geoms, so printing ids is unreadable, and
        a loose pattern silently matching two bodies would quietly invalidate
        the target/placebo comparison.
        """
        matched = find_objects(mj_model, pattern)
        if not matched:
            raise SystemExit(
                f"--{label} {pattern!r} matched no body. Rerun with --list-objects to see the names."
            )
        names = tuple(obj.body_name for obj in matched)
        detail = ", ".join(f"{obj.body_name}({len(obj.geoms)} geoms)" for obj in matched)
        print(f"{label}: {pattern!r} -> {detail}", flush=True)
        if len(matched) > 1:
            print(
                f"  WARNING: {label} matches {len(matched)} bodies and will perturb all of them. "
                "Tighten the pattern if you meant only one.",
                flush=True,
            )
        return names

    print("", flush=True)
    report_target("target", args.target)
    target_geoms = resolve_geom_ids(mj_model, args.target)
    if args.placebo_target:
        placebo_names = report_target("placebo-target", args.placebo_target)
        if set(placebo_names) & {obj.body_name for obj in find_objects(mj_model, args.target)}:
            raise SystemExit(
                "--placebo-target and --target resolve to the same body, so the control "
                "would be a duplicate of the treatment."
            )

    # --- validity self-check: the re-render path must reproduce the env's own
    # observation, otherwise every measurement below compares the wrong images.
    rerendered = rerender_observation(harness)
    env_image = first_camera_image(env_observation)
    rerender_image = first_camera_image(rerendered)
    check = image_delta_stats(env_image, rerender_image)
    print(
        f"re-render self-check: changed_pixel_fraction={check['changed_pixel_fraction']:.6f} "
        f"max_abs_delta={check['max_abs_delta']:.6f}",
        flush=True,
    )
    if check["changed_pixel_fraction"] > 0.01:
        raise RuntimeError(
            "The re-render path does not reproduce the environment's own observation "
            f"({check}). Measurements would compare mismatched images; aborting."
        )

    # --- liveness control. The identity check above cannot tell a correct
    # re-render from a stale cached frame: both give a zero delta. Prove the
    # path is live by perturbing the target to a maximally visible colour and
    # confirming the pixels actually move, then reverting.
    liveness = set_geom_color(mj_model, args.target, (1.0, 0.0, 1.0), label="liveness-probe")
    try:
        live_image = first_camera_image(rerender_observation(harness))
        live = image_delta_stats(rerender_image, live_image)
    finally:
        liveness.revert(mj_model)
    restored = image_delta_stats(rerender_image, first_camera_image(rerender_observation(harness)))
    print(
        f"re-render liveness: perturbing the target moved "
        f"{live['changed_pixel_fraction'] * 100:.3f}% of pixels; "
        f"revert residual {restored['changed_pixel_fraction'] * 100:.3f}%",
        flush=True,
    )
    if live["changed_pixel_fraction"] == 0.0:
        raise RuntimeError(
            "Recolouring the target changed no pixels even at full saturation. Either the "
            "re-render is serving a cached frame, or the object is not visible to this "
            "camera. Measurements would all read zero; aborting."
        )
    if restored["changed_pixel_fraction"] != 0.0:
        raise RuntimeError(
            f"Reverting the liveness probe left {restored['changed_pixel_count']} pixels changed. "
            "The baseline is not reproducible, so paired deltas would be contaminated; aborting."
        )

    print(f"loading transcoders from {args.checkpoint}", flush=True)
    transcoders = load_transcoders(args.checkpoint, device)

    # What answered, not just what was asked: commit, versions, device and a
    # content hash of the checkpoint, so a number can be tied to code and weights.
    provenance = collect_provenance(
        repo_root=Path(__file__).resolve().parents[1],
        device=device,
        checkpoint=args.checkpoint,
        policy_path=args.policy_path,
        policy_dtype=args.policy_dtype,
        extra={"argv": sys.argv[1:]},
    )
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, default=_json_default), encoding="utf-8"
    )
    digest = (provenance.get("transcoder_checkpoint") or {}).get("digest", "")
    print(
        f"provenance: commit {provenance['git'].get('commit')} dirty={provenance['git'].get('dirty')} "
        f"lerobot={provenance['packages'].get('lerobot')} torch={provenance['packages'].get('torch')} "
        f"device={provenance['device'].get('gpu_name', provenance['device'].get('device'))} "
        f"checkpoint sha256={digest[:16]}",
        flush=True,
    )
    accumulator = LatentAccumulator()
    context = Pi05TranscoderContext(
        mode="probe",
        capture_records=False,
        capture_latents=True,
        store_latent_summaries=False,
        latent_callback=accumulator.callback,
    )
    _ctx, wrapped = install_pi05_action_expert_wrappers(
        harness.policy, context=context, transcoders=transcoders, mode="probe"
    )
    print(f"wrapped {len(wrapped)} action-expert MLPs", flush=True)

    tracer = make_langfuse_tracer(feature="pi05-counterfactual-probe", output_root=args.output_dir)
    shape_report: dict[str, Any] = {}

    def forward(
        observation: dict[str, Any], noise: torch.Tensor, *, label: str
    ) -> tuple[LatentAccumulator, np.ndarray]:
        batch = observation_to_batch(harness, observation)

        # Record exactly what is handed to the policy. This is written locally
        # regardless of Langfuse so the shapes are always inspectable, and it
        # doubles as a guard: every condition must send the same shapes, or the
        # comparison is not measuring what it claims to.
        trace_input = tracer.batch_input(batch)
        shapes = trace_input["tensor_shapes"]
        if not shape_report:
            shape_report["batch_keys"] = trace_input["batch_keys"]
            shape_report["task"] = trace_input["task"]
            shape_report["tensor_shapes"] = shapes
            shape_report["noise_shape"] = list(noise.shape)
            shape_report["noise_dtype"] = str(noise.dtype).replace("torch.", "")
            shape_report["num_inference_steps"] = args.num_inference_steps
            print("\nobservation shapes handed to the policy:", flush=True)
            for key in sorted(shapes):
                print(f"  {key:<44} {shapes[key]['shape']}  {shapes[key]['dtype']}", flush=True)
            print(f"  {'noise':<44} {list(noise.shape)}  {shape_report['noise_dtype']}", flush=True)
            print(f"  task: {trace_input['task']!r}\n", flush=True)
        elif shapes != shape_report["tensor_shapes"]:
            raise RuntimeError(
                f"Batch shapes changed between conditions at {label!r}. Baseline sent "
                f"{shape_report['tensor_shapes']}, this pass sent {shapes}. The paired "
                "comparison would be invalid."
            )

        accumulator.reset()
        accumulator.enabled = True
        try:
            with tracer.trace(
                f"counterfactual/{label}",
                input=trace_input if args.trace_langfuse else None,
                metadata={"label": label, "num_inference_steps": args.num_inference_steps},
            ) as span:
                with torch.inference_mode():
                    actions = harness.policy.predict_action_chunk(
                        batch, num_steps=args.num_inference_steps, noise=noise
                    )
                if args.trace_langfuse:
                    span.update(output=summarize_action_tensor(actions))
        finally:
            accumulator.enabled = False
        snapshot = LatentAccumulator()
        snapshot.mean = dict(accumulator.mean)
        snapshot.max = dict(accumulator.max)
        chunk = actions.detach().float().cpu().numpy()[0]
        shape_report.setdefault("action_chunk_shape", list(chunk.shape))
        return snapshot, chunk

    doses = [float(part) for part in args.dose.split(",") if part.strip()]
    targets = [("target", args.target)]
    if args.placebo_target:
        targets.append(("placebo", args.placebo_target))

    prompt_variants: list[tuple[str, str]] = [("task", harness.prompt)]
    alt_prompt_source = None
    if args.use_alt_prompt:
        if args.alt_prompt:
            prompt_variants.append(("alt", args.alt_prompt))
            alt_prompt_source = "user-supplied"
        elif args.placebo_target:
            placebo_bodies = [obj.body_name for obj in find_objects(mj_model, args.placebo_target)]
            found = find_prompt_referring_to(harness, placebo_bodies[0]) if placebo_bodies else None
            if found:
                prompt_variants.append(("alt", found[0]))
                alt_prompt_source = found[1]
    if len(prompt_variants) > 1:
        print(
            f"\nalt prompt ({alt_prompt_source}): {prompt_variants[1][1]!r}\n"
            "  Same pixels, same noise, only the language differs. If selectivity follows the\n"
            "  prompt rather than the screen position, the response tracks task relevance.",
            flush=True,
        )
    elif args.use_alt_prompt:
        print("\nno alt prompt found; selectivity cannot be separated from screen position", flush=True)

    noise_seed = args.seed if args.noise_seed is None else args.noise_seed
    if args.noise_samples < 1:
        raise SystemExit("--noise-samples must be at least 1")
    store: DeltaStore | None = None

    def ensure_store(latents: LatentAccumulator) -> DeltaStore | None:
        nonlocal store
        if not args.full_deltas or store is not None:
            return store
        layer_names = sort_layer_names({name for name, _step in latents.max})
        num_features = int(next(iter(latents.max.values())).shape[0])
        store = DeltaStore.create(
            args.output_dir / "latents",
            layer_names=layer_names,
            num_steps=args.num_inference_steps,
            num_features=num_features,
        )
        print(
            f"full delta store: {len(layer_names)} layers x {args.num_inference_steps} steps x "
            f"{num_features} features per block -> {store.root}",
            flush=True,
        )
        return store
    last_baseline_actions = None
    # Keyed (state, noise draw, prompt): the unperturbed action and latents.
    baseline_by_prompt: dict[tuple[int, int, str], np.ndarray] = {}
    baseline_latents_by_prompt: dict[tuple[int, int, str], LatentAccumulator] = {}
    latent_rows: list[dict[str, Any]] = []
    measurements: list[dict[str, Any]] = []
    image_dir = args.output_dir / "images"
    if args.save_images:
        image_dir.mkdir(parents=True, exist_ok=True)

    for state_index in range(args.states):
        print(f"\n=== state {state_index} ===", flush=True)
        baseline_obs = rerender_observation(harness)
        baseline_image = first_camera_image(baseline_obs)

        if args.save_images:
            save_image(image_dir / f"state{state_index}_baseline.png", baseline_image)

        # The perturbed renders do not depend on the noise draw or the prompt,
        # so render each once per state and reuse it across every replicate.
        perturbed_renders: dict[tuple[str, float], tuple[dict[str, Any], np.ndarray, dict[str, float], str]] = {}
        for kind, target in targets:
            for dose in doses:
                perturbation = apply_perturbation(mj_model, args, target, dose)
                try:
                    perturbed_obs = rerender_observation(harness)
                finally:
                    perturbation.revert(mj_model)
                perturbed_image = first_camera_image(perturbed_obs)
                pixel = image_delta_stats(baseline_image, perturbed_image)
                if pixel["changed_pixel_fraction"] == 0.0:
                    raise RuntimeError(
                        f"Perturbation {perturbation.label!r} changed no pixels. The object is "
                        "probably occluded or outside this camera's view; pick another target."
                    )
                perturbed_renders[(kind, dose)] = (perturbed_obs, perturbed_image, pixel, perturbation.label)
                if args.save_images:
                    save_image(image_dir / f"state{state_index}_{kind}_dose{dose:g}.png", perturbed_image)
                    diff = np.abs(perturbed_image.astype(np.float64) - baseline_image.astype(np.float64))
                    if diff.max() > 0:
                        diff = diff / diff.max() * 255.0
                    save_image(image_dir / f"state{state_index}_{kind}_dose{dose:g}_diff.png", diff)
        # The baseline must be intact after every revert, or the pairs below
        # would be measured against a contaminated reference.
        residual = image_delta_stats(baseline_image, first_camera_image(rerender_observation(harness)))
        if residual["changed_pixel_fraction"] != 0.0:
            raise RuntimeError(
                f"Reverting the perturbations left {residual['changed_pixel_count']} pixels changed at "
                f"state {state_index}; the baseline is not reproducible, aborting."
            )

        for noise_index in range(args.noise_samples):
            cell_seed = noise_seed_for(noise_seed, state_index, noise_index)
            noise = sample_shared_noise(harness.policy, 1, device, seed=cell_seed)
            if args.noise_samples > 1:
                print(f"  --- noise draw {noise_index} (seed {cell_seed}) ---", flush=True)

            # Blocks are written after both prompts are measured, so that the
            # task block can also carry the prompt-swap positive control.
            pending_blocks: dict[str, tuple[LatentAccumulator, dict[tuple[str, float], dict[tuple[str, int], np.ndarray]]]] = {}

            # Every prompt variant sees pixel-identical images and the same noise,
            # so the only thing that changes between variants is the language.
            for prompt_label, prompt_text in prompt_variants:
                harness.prompt = prompt_text
                if len(prompt_variants) > 1:
                    print(f"  [prompt:{prompt_label}] {prompt_text!r}", flush=True)
                tag = f"s{state_index}/n{noise_index}/{prompt_label}"

                baseline_latents, baseline_actions = forward(baseline_obs, noise, label=f"{tag}/baseline")
                null_latents, null_actions = forward(baseline_obs, noise, label=f"{tag}/null")
                if prompt_label == prompt_variants[0][0] and noise_index == 0:
                    last_baseline_actions = baseline_actions
                # Keep the unperturbed action per prompt: comparing these says whether
                # the policy responds to the language at all. If it does not, the
                # prompt-swap control manipulated nothing and proves nothing.
                baseline_by_prompt[(state_index, noise_index, prompt_label)] = baseline_actions
                baseline_latents_by_prompt[(state_index, noise_index, prompt_label)] = baseline_latents
                ensure_store(baseline_latents)
                block_deltas: dict[tuple[str, float], dict[tuple[str, int], np.ndarray]] = {}
                if store is not None:
                    block_deltas[("null", 0.0)] = full_delta(baseline_latents, null_latents)

                null_rows = latent_delta_rows(
                    baseline_latents, null_latents, condition="null", top_features=args.top_features
                )
                for row in null_rows:
                    row["state_index"] = state_index
                    row["noise_index"] = noise_index
                    row["dose"] = 0.0
                    row["target"] = "none"
                    row["prompt"] = prompt_label
                latent_rows.extend(null_rows)
                null_summary = summarize_latent_rows(null_rows)
                print(
                    f"    null control  latent L2 mean={null_summary.get('l2_delta_mean'):.6g} "
                    f"action rel_l2={_rel_l2(baseline_actions, null_actions):.6g}",
                    flush=True,
                )
                measurements.append(
                    {
                        "state_index": state_index,
                        "noise_index": noise_index,
                        "prompt": prompt_label,
                        "kind": "null",
                        "target": "none",
                        "dose": 0.0,
                        "pixel": image_delta_stats(baseline_image, baseline_image),
                        "latent": null_summary,
                        "action_relative_l2": _rel_l2(baseline_actions, null_actions),
                    }
                )

                for kind, target in targets:
                    for dose in doses:
                        perturbed_obs, perturbed_image, pixel, perturbation_label = perturbed_renders[(kind, dose)]
                        perturbed_latents, perturbed_actions = forward(
                            perturbed_obs, noise, label=f"{tag}/{kind}/dose{dose:g}"
                        )
                        rows = latent_delta_rows(
                            baseline_latents, perturbed_latents, condition=kind, top_features=args.top_features
                        )
                        if store is not None:
                            block_deltas[(kind, dose)] = full_delta(baseline_latents, perturbed_latents)
                        for row in rows:
                            row["state_index"] = state_index
                            row["noise_index"] = noise_index
                            row["dose"] = dose
                            row["target"] = target
                            row["prompt"] = prompt_label
                        latent_rows.extend(rows)
                        summary = summarize_latent_rows(rows)
                        action_rel = _rel_l2(baseline_actions, perturbed_actions)
                        print(
                            f"    {kind:<8} {target:<28} dose={dose:<5g} "
                            f"pixels={pixel['changed_pixel_fraction']:.4f} "
                            f"latentL2={summary.get('l2_delta_mean'):.6g} "
                            f"action_rel_l2={action_rel:.6g}",
                            flush=True,
                        )
                        measurements.append(
                            {
                                "state_index": state_index,
                                "noise_index": noise_index,
                                "prompt": prompt_label,
                                "kind": kind,
                                "target": target,
                                "dose": dose,
                                "perturbation": perturbation_label,
                                "pixel": pixel,
                                "latent": summary,
                                "action_relative_l2": action_rel,
                            }
                        )

                pending_blocks[prompt_label] = (baseline_latents, block_deltas)

            # Positive control. Swapping the prompt is a manipulation known to
            # change behaviour (its action effect is g, Eq. grounding), and its
            # latent response is available for free from the two baselines. It
            # anchors the scale: a recolour response is read as a fraction of
            # what a behaviour-changing manipulation produces on these layers.
            task_label = prompt_variants[0][0]
            if task_label in pending_blocks and "alt" in pending_blocks:
                task_latents, task_block = pending_blocks[task_label]
                alt_latents, _alt_block = pending_blocks["alt"]
                swap_rows = latent_delta_rows(
                    task_latents, alt_latents, condition="prompt_swap", top_features=args.top_features
                )
                for row in swap_rows:
                    row["state_index"] = state_index
                    row["noise_index"] = noise_index
                    row["dose"] = 0.0
                    row["target"] = "none"
                    row["prompt"] = task_label
                latent_rows.extend(swap_rows)
                swap_summary = summarize_latent_rows(swap_rows)
                grounding_here = _rel_l2(
                    baseline_by_prompt[(state_index, noise_index, task_label)],
                    baseline_by_prompt[(state_index, noise_index, "alt")],
                )
                print(
                    f"    positive control (prompt swap) latent L2 mean={swap_summary.get('l2_delta_mean'):.6g} "
                    f"action rel_l2={grounding_here:.6g}",
                    flush=True,
                )
                measurements.append(
                    {
                        "state_index": state_index,
                        "noise_index": noise_index,
                        "prompt": task_label,
                        "kind": "prompt_swap",
                        "target": "none",
                        "dose": 0.0,
                        "pixel": image_delta_stats(baseline_image, baseline_image),
                        "latent": swap_summary,
                        "action_relative_l2": grounding_here,
                    }
                )
                if store is not None:
                    task_block[("prompt_swap", 0.0)] = full_delta(task_latents, alt_latents)

            if store is not None:
                for prompt_label, (block_latents, block_deltas) in pending_blocks.items():
                    store.write_block(
                        state=state_index,
                        noise=noise_index,
                        prompt=prompt_label,
                        baseline_max=block_latents.max,
                        deltas=block_deltas,
                    )

        harness.prompt = prompt_variants[0][1]

        # Advance the unperturbed trajectory so the next measurement sits at a
        # genuinely different state, by executing the first few actions of the
        # policy's own unperturbed plan (task prompt, first noise draw). An
        # alternate prompt is a measurement condition, not a steering input.
        for step_action in actions_to_step(last_baseline_actions, args.state_stride):
            harness.vec_env.step(step_action)

    payload = {
        "config": {
            "suite": args.suite,
            "task_id": args.task_id,
            "seed": args.seed,
            "prompt": harness.prompt,
            "prompt_variants": {label: text for label, text in prompt_variants},
            "alt_prompt_source": alt_prompt_source,
            "target": args.target,
            "placebo_target": args.placebo_target,
            "perturbation": args.perturbation,
            "color": args.color,
            "doses": doses,
            "states": args.states,
            "state_stride": args.state_stride,
            "noise_samples": args.noise_samples,
            "noise_seed": noise_seed,
            "num_inference_steps": args.num_inference_steps,
            "top_features": args.top_features,
            "full_delta_store": None if store is None else str(store.root),
            "checkpoint": str(args.checkpoint),
        },
        "rerender_self_check": check,
        "provenance": provenance,
        "prompt_grounding_rel_l2": [
            _rel_l2(baseline_by_prompt[(i, n, "task")], baseline_by_prompt[(i, n, "alt")])
            for i in range(args.states)
            for n in range(args.noise_samples)
            if (i, n, "task") in baseline_by_prompt and (i, n, "alt") in baseline_by_prompt
        ],
        "observation_shapes": shape_report,
        "measurements": measurements,
        "interpretation": (
            "latent deltas are per (layer, denoise step) differences in per-feature max activation, "
            "measured at a frozen sim state with shared diffusion noise. Read every perturbation "
            "delta against the 'null' rows, which repeat the identical observation and therefore "
            "give the nondeterminism floor."
        ),
    }
    (args.output_dir / "counterfactual_summary.json").write_text(
        json.dumps(payload, indent=2, default=_json_default), encoding="utf-8"
    )
    (args.output_dir / "observation_shapes.json").write_text(
        json.dumps(shape_report, indent=2, default=_json_default), encoding="utf-8"
    )
    tracer.flush()
    write_csv(args.output_dir / "latent_deltas.csv", latent_rows)
    write_csv(
        args.output_dir / "measurements.csv",
        [
            {
                "state_index": m["state_index"],
                "noise_index": m.get("noise_index", 0),
                "prompt": m.get("prompt"),
                "kind": m["kind"],
                "target": m["target"],
                "dose": m["dose"],
                "changed_pixel_fraction": m["pixel"]["changed_pixel_fraction"],
                "pixel_relative_l2": m["pixel"]["relative_l2"],
                "latent_l2_mean": m["latent"].get("l2_delta_mean"),
                "latent_relative_l2_mean": m["latent"].get("relative_l2_mean"),
                "changed_features_mean": m["latent"].get("changed_features_mean"),
                "action_relative_l2": m["action_relative_l2"],
            }
            for m in measurements
        ],
    )

    _print_verdict(
        measurements, baseline_by_prompt=baseline_by_prompt, states=args.states, noise_samples=args.noise_samples
    )
    decision = compute_decision_metrics(measurements, baseline_by_prompt=baseline_by_prompt)
    _print_decision_metrics(decision)
    payload["decision_metrics"] = decision
    (args.output_dir / "counterfactual_summary.json").write_text(
        json.dumps(payload, indent=2, default=_json_default), encoding="utf-8"
    )
    (args.output_dir / "decision_metrics.json").write_text(
        json.dumps(decision, indent=2, default=_json_default), encoding="utf-8"
    )
    write_csv(args.output_dir / "h1_cells.csv", decision["h1"]["cells"])
    print(f"\nArtifacts in {args.output_dir}", flush=True)
    harness.vec_env.close()


def _rel_l2(baseline: np.ndarray, other: np.ndarray) -> float:
    denominator = float(np.linalg.norm(baseline))
    if denominator == 0.0:
        return float("nan")
    return float(np.linalg.norm(other - baseline) / denominator)


def _infer_prompt(harness: Harness, observation: dict[str, Any]) -> str:
    for candidate in (observation.get("task"), getattr(harness.inner_env, "task_description", None)):
        if isinstance(candidate, (list, tuple)) and candidate:
            return str(candidate[0])
        if isinstance(candidate, str) and candidate:
            return candidate
    language = getattr(getattr(harness.inner_env, "_env", None), "language_instruction", None)
    return str(language) if language else ""


# ---------------------------------------------------------------- decision metrics
#
# Everything below computes the quantities the write-up's decision rules are
# stated in, so that the verdict is read off numbers the run itself emits
# rather than derived by hand from the CSV afterwards.
#
#   D(c)      layer response, Eq. layer-response: ||delta||_2 over features,
#             averaged over layers and denoise steps, i.e. `l2_delta_mean`.
#   S(c)      perturbation size in image space, two choices: changed-pixel
#             fraction |P| and image-difference norm ||dI||_2.
#   Sel       D(target) / D(placebo)
#   Sel_adj   Sel / (S(target) / S(placebo)), the footprint adjustment.
#   g         prompt grounding, Eq. grounding.

GROUNDING_THRESHOLD = 0.01  # relative action change below which a prompt swap is treated as inert
NEAR_ZERO_FLOOR_FRACTION = 0.05  # null floor must be below this fraction of the placebo response


def _spread(values: list[float]) -> dict[str, Any]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _ratio_of_means(numerators: list[float], denominators: list[float]) -> float | None:
    if not numerators or not denominators:
        return None
    den = float(np.mean(denominators))
    return float(np.mean(numerators)) / den if den > 0 else None


def _bootstrap_ratio_ci(
    cells: list[dict[str, Any]], numerator: str, denominator: str, *, adjust: tuple[str, str] | None,
    draws: int = 2000, seed: int = 0,
) -> dict[str, Any]:
    """Percentile CI for a pooled ratio, resampling cells with replacement.

    Cells are the replication unit (state x noise draw x dose). With few cells
    the interval is wide and coarse; that is the honest state of the evidence,
    and n is reported next to it.
    """
    if len(cells) < 2:
        return {"n_cells": len(cells), "draws": 0, "low": None, "high": None}
    rng = np.random.default_rng(seed)
    num = np.asarray([c[numerator] for c in cells], dtype=np.float64)
    den = np.asarray([c[denominator] for c in cells], dtype=np.float64)
    if adjust is not None:
        s_num = np.asarray([c[adjust[0]] for c in cells], dtype=np.float64)
        s_den = np.asarray([c[adjust[1]] for c in cells], dtype=np.float64)
    samples = []
    n = len(cells)
    for _ in range(draws):
        idx = rng.integers(0, n, size=n)
        d = den[idx].mean()
        if d <= 0:
            continue
        value = num[idx].mean() / d
        if adjust is not None:
            sd = s_den[idx].mean()
            sn = s_num[idx].mean()
            if sd <= 0 or sn <= 0:
                continue
            value = value / (sn / sd)
        samples.append(value)
    if not samples:
        return {"n_cells": n, "draws": 0, "low": None, "high": None}
    arr = np.asarray(samples)
    return {
        "n_cells": n,
        "draws": int(arr.size),
        "low": float(np.percentile(arr, 2.5)),
        "high": float(np.percentile(arr, 97.5)),
    }


def _cell_key(m: dict[str, Any]) -> tuple[int, int, float]:
    return (int(m["state_index"]), int(m.get("noise_index", 0)), float(m["dose"]))


def h1_cells(measurements: list[dict[str, Any]], *, prompt: str) -> list[dict[str, Any]]:
    """One row per (state, noise draw, dose) with both target and placebo measured."""
    by_key: dict[tuple[int, int, float], dict[str, dict[str, Any]]] = defaultdict(dict)
    for m in measurements:
        if m.get("prompt", "task") != prompt or m["kind"] not in ("target", "placebo"):
            continue
        by_key[_cell_key(m)][m["kind"]] = m
    cells = []
    for key in sorted(by_key):
        pair = by_key[key]
        if "target" not in pair or "placebo" not in pair:
            continue
        t, p_ = pair["target"], pair["placebo"]
        cell = {
            "state_index": key[0],
            "noise_index": key[1],
            "dose": key[2],
            "D_target": float(t["latent"].get("l2_delta_mean") or 0.0),
            "D_placebo": float(p_["latent"].get("l2_delta_mean") or 0.0),
            "S_px_target": float(t["pixel"]["changed_pixel_fraction"]),
            "S_px_placebo": float(p_["pixel"]["changed_pixel_fraction"]),
            "S_l2_target": float(t["pixel"].get("l2_delta") or 0.0),
            "S_l2_placebo": float(p_["pixel"].get("l2_delta") or 0.0),
            "A_target": float(t["action_relative_l2"]),
            "A_placebo": float(p_["action_relative_l2"]),
        }
        cell["sel_raw"] = cell["D_target"] / cell["D_placebo"] if cell["D_placebo"] > 0 else None
        for name in ("px", "l2"):
            s_ratio = (
                cell[f"S_{name}_target"] / cell[f"S_{name}_placebo"] if cell[f"S_{name}_placebo"] > 0 else None
            )
            cell[f"S_{name}_ratio"] = s_ratio
            cell[f"sel_adj_{name}"] = (
                cell["sel_raw"] / s_ratio if (cell["sel_raw"] is not None and s_ratio) else None
            )
        cell["action_sel"] = cell["A_target"] / cell["A_placebo"] if cell["A_placebo"] > 0 else None
        cells.append(cell)
    return cells


def dose_response(measurements: list[dict[str, Any]], *, prompt: str, kind: str) -> dict[str, Any]:
    """Does the response grow with the dose, and how close to linearly?

    Elasticity is log(D_hi/D_lo) / log(S_hi/S_lo) between consecutive doses,
    with S the image-difference norm (the changed-pixel count barely moves with
    a blend dose, so it cannot carry this check). Linear response gives 1.
    """
    per_dose: dict[float, dict[str, list[float]]] = defaultdict(lambda: {"D": [], "S": [], "A": []})
    for m in measurements:
        if m.get("prompt", "task") != prompt or m["kind"] != kind:
            continue
        bucket = per_dose[float(m["dose"])]
        bucket["D"].append(float(m["latent"].get("l2_delta_mean") or 0.0))
        bucket["S"].append(float(m["pixel"].get("l2_delta") or 0.0))
        bucket["A"].append(float(m["action_relative_l2"]))
    doses = sorted(per_dose)
    rows = [
        {
            "dose": d,
            "D": float(np.mean(per_dose[d]["D"])),
            "S_l2": float(np.mean(per_dose[d]["S"])),
            "A": float(np.mean(per_dose[d]["A"])),
            "n": len(per_dose[d]["D"]),
        }
        for d in doses
    ]
    elasticities = []
    for lo, hi in zip(rows, rows[1:]):
        if lo["D"] > 0 and hi["D"] > 0 and lo["S_l2"] > 0 and hi["S_l2"] > 0 and hi["S_l2"] != lo["S_l2"]:
            elasticities.append(float(np.log(hi["D"] / lo["D"]) / np.log(hi["S_l2"] / lo["S_l2"])))
    monotone = all(hi["D"] > lo["D"] for lo, hi in zip(rows, rows[1:])) if len(rows) > 1 else None
    return {
        "kind": kind,
        "prompt": prompt,
        "rows": rows,
        "elasticity": elasticities,
        "elasticity_mean": float(np.mean(elasticities)) if elasticities else None,
        "monotone_in_dose": monotone,
    }


def compute_decision_metrics(
    measurements: list[dict[str, Any]],
    *,
    baseline_by_prompt: dict[tuple[int, int, str], np.ndarray] | None = None,
    grounding_threshold: float = GROUNDING_THRESHOLD,
) -> dict[str, Any]:
    """Every number the pre-registered decision rules are stated in.

    H1 is decided under the task prompt on the footprint-adjusted selectivity
    with S = ||dI||_2 as the primary normaliser (it reflects how far pixels
    moved, which the pixel count does not); |P| is reported alongside.

    H2 is decided on the raw within-prompt selectivity, because the images and
    hence S are identical across prompts, gated on prompt grounding g.
    """
    baseline_by_prompt = baseline_by_prompt or {}
    prompts = sorted({m.get("prompt", "task") for m in measurements})
    primary = "task" if "task" in prompts else (prompts[0] if prompts else "task")

    # ---- H1
    cells = h1_cells(measurements, prompt=primary)
    nulls = [
        float(m["latent"].get("l2_delta_mean") or 0.0)
        for m in measurements
        if m["kind"] == "null" and m.get("prompt", "task") == primary
    ]
    null_floor = _spread(nulls)
    pooled = {
        "D_target": float(np.mean([c["D_target"] for c in cells])) if cells else None,
        "D_placebo": float(np.mean([c["D_placebo"] for c in cells])) if cells else None,
        "S_px_target": float(np.mean([c["S_px_target"] for c in cells])) if cells else None,
        "S_px_placebo": float(np.mean([c["S_px_placebo"] for c in cells])) if cells else None,
        "S_l2_target": float(np.mean([c["S_l2_target"] for c in cells])) if cells else None,
        "S_l2_placebo": float(np.mean([c["S_l2_placebo"] for c in cells])) if cells else None,
        "A_target": float(np.mean([c["A_target"] for c in cells])) if cells else None,
        "A_placebo": float(np.mean([c["A_placebo"] for c in cells])) if cells else None,
    }
    pooled["sel_raw"] = _ratio_of_means([c["D_target"] for c in cells], [c["D_placebo"] for c in cells])
    for name in ("px", "l2"):
        s_ratio = _ratio_of_means([c[f"S_{name}_target"] for c in cells], [c[f"S_{name}_placebo"] for c in cells])
        pooled[f"S_{name}_ratio"] = s_ratio
        pooled[f"sel_adj_{name}"] = (pooled["sel_raw"] / s_ratio) if (pooled["sel_raw"] is not None and s_ratio) else None
    pooled["action_sel"] = _ratio_of_means([c["A_target"] for c in cells], [c["A_placebo"] for c in cells])

    spread = {
        key: _spread([c[key] for c in cells])
        for key in ("sel_raw", "sel_adj_px", "sel_adj_l2", "action_sel", "D_target", "D_placebo")
    }
    ci = {
        "sel_raw": _bootstrap_ratio_ci(cells, "D_target", "D_placebo", adjust=None),
        "sel_adj_px": _bootstrap_ratio_ci(cells, "D_target", "D_placebo", adjust=("S_px_target", "S_px_placebo")),
        "sel_adj_l2": _bootstrap_ratio_ci(cells, "D_target", "D_placebo", adjust=("S_l2_target", "S_l2_placebo")),
    }
    doses = {
        kind: dose_response(measurements, prompt=primary, kind=kind)
        for kind in ("target", "placebo")
        if any(m["kind"] == kind for m in measurements)
    }

    # Positive control: the prompt swap's latent response, per (state, draw),
    # as the scale a behaviour-changing manipulation produces on these layers.
    swaps = [
        float(m["latent"].get("l2_delta_mean") or 0.0)
        for m in measurements
        if m["kind"] == "prompt_swap" and m.get("prompt", "task") == primary
    ]
    positive = {
        "kind": "prompt_swap",
        "n": len(swaps),
        "D": float(np.mean(swaps)) if swaps else None,
        "spread": _spread(swaps),
        "target_over_positive": None,
        "placebo_over_positive": None,
    }
    if swaps and positive["D"]:
        if pooled["D_target"] is not None:
            positive["target_over_positive"] = pooled["D_target"] / positive["D"]
        if pooled["D_placebo"] is not None:
            positive["placebo_over_positive"] = pooled["D_placebo"] / positive["D"]

    floor_ok = (
        null_floor["max"] is not None
        and pooled["D_placebo"] is not None
        and null_floor["max"] <= NEAR_ZERO_FLOOR_FRACTION * pooled["D_placebo"]
    )
    adj = pooled.get("sel_adj_l2")
    adj_min = spread["sel_adj_l2"]["min"]
    if not cells or adj is None:
        h1_verdict = "not measured"
    elif not floor_ok:
        h1_verdict = "invalid: null floor is not near zero"
    elif adj > 1.0 and adj_min is not None and adj_min > 1.0:
        h1_verdict = "supported"
    elif adj > 1.0:
        h1_verdict = "weak: pooled adjusted selectivity above 1 but not in every cell"
    else:
        h1_verdict = "falsified"

    h1 = {
        "prompt": primary,
        "primary_normaliser": "S_l2",
        "cells": cells,
        "null_floor": null_floor,
        "null_floor_near_zero": bool(floor_ok),
        "pooled": pooled,
        "spread": spread,
        "bootstrap_ci_95": ci,
        "dose_response": doses,
        "positive_control": positive,
        "decision_rule": (
            "supported if the null floor is at most 5% of the placebo response, the pooled "
            "footprint-adjusted selectivity (S = image-difference norm) exceeds 1, and every "
            "cell's adjusted selectivity exceeds 1; weak if only the pooled value does; "
            "falsified otherwise"
        ),
        "verdict": h1_verdict,
    }

    # ---- H2
    grounding_values = []
    for (state_index, noise_index, label), action in baseline_by_prompt.items():
        if label != "task":
            continue
        alt = baseline_by_prompt.get((state_index, noise_index, "alt"))
        if alt is not None:
            grounding_values.append(_rel_l2(action, alt))
    grounding = _spread(grounding_values)
    per_prompt = {}
    for prompt in prompts:
        p_cells = h1_cells(measurements, prompt=prompt)
        per_prompt[prompt] = {
            "n_cells": len(p_cells),
            "D_target": float(np.mean([c["D_target"] for c in p_cells])) if p_cells else None,
            "D_placebo": float(np.mean([c["D_placebo"] for c in p_cells])) if p_cells else None,
            "sel_raw": _ratio_of_means([c["D_target"] for c in p_cells], [c["D_placebo"] for c in p_cells]),
            "sel_raw_spread": _spread([c["sel_raw"] for c in p_cells]),
            "A_target": float(np.mean([c["A_target"] for c in p_cells])) if p_cells else None,
            "A_placebo": float(np.mean([c["A_placebo"] for c in p_cells])) if p_cells else None,
        }
    sel_task = per_prompt.get("task", {}).get("sel_raw")
    sel_alt = per_prompt.get("alt", {}).get("sel_raw")
    if "alt" not in prompts:
        h2_verdict = "untestable: no alternate prompt"
    elif grounding["n"] == 0 or grounding["mean"] is None or grounding["mean"] < grounding_threshold:
        h2_verdict = "untestable: prompt grounding below threshold"
    elif sel_task is None or sel_alt is None:
        h2_verdict = "not measured"
    elif sel_task <= 1.0:
        h2_verdict = "not applicable: no selectivity under the task prompt to follow the referent"
    elif sel_alt < 1.0:
        h2_verdict = "supported: selectivity inverts under the sibling prompt"
    elif sel_alt >= sel_task:
        h2_verdict = "falsified: selectivity unchanged or stronger for the same object"
    else:
        h2_verdict = "partial: selectivity weakens but does not invert"
    h2 = {
        "grounding_threshold": grounding_threshold,
        "grounding": grounding,
        "grounding_values": grounding_values,
        "perturbation_action_effect": pooled.get("A_target"),
        "per_prompt": per_prompt,
        "decision_rule": (
            "untestable if g < threshold; supported if selectivity under the sibling prompt falls "
            "below 1 while the task prompt's exceeds 1; falsified if it is unchanged or stronger; "
            "partial if it weakens without inverting"
        ),
        "verdict": h2_verdict,
    }
    return {"h1": h1, "h2": h2}


def _fmt(value: Any, spec: str = ".4g") -> str:
    if value is None:
        return "n/a"
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def _print_decision_metrics(metrics: dict[str, Any]) -> None:
    h1, h2 = metrics["h1"], metrics["h2"]
    print("\n--- decision metrics (task prompt) ---", flush=True)
    print(f"cells (state x noise draw x dose): {len(h1['cells'])}", flush=True)
    nf = h1["null_floor"]
    print(
        f"null floor D: mean={_fmt(nf['mean'])} max={_fmt(nf['max'])} "
        f"({'near zero' if h1['null_floor_near_zero'] else 'NOT near zero'})",
        flush=True,
    )
    p = h1["pooled"]
    print(f"{'':<22}{'target':>12}{'placebo':>12}{'ratio':>10}", flush=True)
    print(f"{'D (latent L2)':<22}{_fmt(p['D_target']):>12}{_fmt(p['D_placebo']):>12}{_fmt(p['sel_raw']):>10}", flush=True)
    print(f"{'S |P| (px frac)':<22}{_fmt(p['S_px_target']):>12}{_fmt(p['S_px_placebo']):>12}{_fmt(p['S_px_ratio']):>10}", flush=True)
    print(f"{'S ||dI||_2':<22}{_fmt(p['S_l2_target']):>12}{_fmt(p['S_l2_placebo']):>12}{_fmt(p['S_l2_ratio']):>10}", flush=True)
    print(f"{'action rel L2':<22}{_fmt(p['A_target']):>12}{_fmt(p['A_placebo']):>12}{_fmt(p['action_sel']):>10}", flush=True)
    for key, label in (("sel_raw", "Sel raw"), ("sel_adj_px", "Sel adj |P|"), ("sel_adj_l2", "Sel adj ||dI||")):
        sp, ci = h1["spread"][key], h1["bootstrap_ci_95"][key]
        print(
            f"{label:<16} pooled={_fmt(p[key])}  cells mean={_fmt(sp['mean'])} sd={_fmt(sp['sd'])} "
            f"min={_fmt(sp['min'])} max={_fmt(sp['max'])}  95% CI [{_fmt(ci['low'])}, {_fmt(ci['high'])}]",
            flush=True,
        )
    for kind, dr in h1["dose_response"].items():
        rows = "  ".join(f"d={r['dose']:g}: D={_fmt(r['D'])} S={_fmt(r['S_l2'])}" for r in dr["rows"])
        print(
            f"dose response [{kind}]: {rows}  elasticity={_fmt(dr['elasticity_mean'])} "
            f"monotone={dr['monotone_in_dose']}",
            flush=True,
        )
    pc = h1.get("positive_control", {})
    if pc.get("n"):
        print(
            f"positive control (prompt swap) D={_fmt(pc['D'])} (n={pc['n']}); target is "
            f"{_fmt(pc['target_over_positive'])} of it, placebo {_fmt(pc['placebo_over_positive'])}",
            flush=True,
        )
    print(f"H1 verdict: {h1['verdict']}", flush=True)

    print("\n--- decision metrics (prompt swap) ---", flush=True)
    g = h2["grounding"]
    print(
        f"prompt grounding g: mean={_fmt(g['mean'])} min={_fmt(g['min'])} max={_fmt(g['max'])} "
        f"(n={g['n']}, threshold {h2['grounding_threshold']}); perturbation action effect "
        f"{_fmt(h2['perturbation_action_effect'])}",
        flush=True,
    )
    for prompt, row in h2["per_prompt"].items():
        print(
            f"  {prompt:>5}: D_target={_fmt(row['D_target'])} D_placebo={_fmt(row['D_placebo'])} "
            f"Sel={_fmt(row['sel_raw'])} (cells {row['n_cells']})",
            flush=True,
        )
    print(f"H2 verdict: {h2['verdict']}", flush=True)


def _print_verdict(
    measurements: list[dict[str, Any]],
    *,
    baseline_by_prompt: dict[tuple[int, int, str], np.ndarray] | None = None,
    states: int = 0,
    noise_samples: int = 1,
) -> None:
    """Summarise the run.

    ``baseline_by_prompt`` holds the unperturbed action per (state, noise draw,
    prompt); it is what decides whether the prompt swap manipulated anything.
    """
    baseline_by_prompt = baseline_by_prompt or {}
    nulls = [m for m in measurements if m["kind"] == "null"]
    targets = [m for m in measurements if m["kind"] == "target"]
    if not nulls or not targets:
        return
    floor = float(np.mean([m["latent"].get("l2_delta_mean") or 0.0 for m in nulls]))
    signal = float(np.mean([m["latent"].get("l2_delta_mean") or 0.0 for m in targets]))
    print("\n--- verdict ---", flush=True)
    print(f"null-control latent L2 (nondeterminism floor): {floor:.6g}", flush=True)
    print(f"target-perturbation latent L2:                 {signal:.6g}", flush=True)
    if floor == 0.0:
        print("Forward passes are bit-deterministic, so the entire target delta is signal.", flush=True)
    else:
        print(f"signal-to-floor ratio: {signal / floor:.3g}", flush=True)
    # Prompt-swap control: identical pixels, only the language differs.
    prompts = {m.get("prompt") for m in measurements if m.get("prompt")}
    if len(prompts) > 1:
        def mean_for(prompt: str, kind: str) -> float:
            vals = [
                m["latent"].get("l2_delta_mean") or 0.0
                for m in measurements
                if m.get("prompt") == prompt and m["kind"] == kind
            ]
            return float(np.mean(vals)) if vals else 0.0

        # Does the policy respond to the language at all on these images? Without
        # this, a "selectivity did not follow the prompt" result is ambiguous
        # between "features track position" and "the prompt was ignored".
        grounding = []
        for state_index in range(states):
            for noise_index in range(noise_samples):
                a = baseline_by_prompt.get((state_index, noise_index, "task"))
                b = baseline_by_prompt.get((state_index, noise_index, "alt"))
                if a is not None and b is not None:
                    grounding.append(_rel_l2(a, b))
        perturbation_effect = float(
            np.mean([m["action_relative_l2"] for m in measurements if m["kind"] == "target"])
        ) if any(m["kind"] == "target" for m in measurements) else 0.0

        print("\nprompt grounding check (unperturbed action, task prompt vs alt prompt):", flush=True)
        if grounding:
            mean_grounding = float(np.mean(grounding))
            print(
                f"  swapping the prompt alone moves the action by rel_l2={mean_grounding:.4g}; "
                f"recolouring the target moves it by {perturbation_effect:.4g}",
                flush=True,
            )
            if mean_grounding < 0.01:
                print(
                    "  The policy barely reacts to the language here, so the prompt swap did not "
                    "actually change what the task refers to. Treat the control below as VOID.",
                    flush=True,
                )
            else:
                print(
                    "  The policy does react to the language, so the prompt swap is a real "
                    "manipulation and the control below is interpretable.",
                    flush=True,
                )

        print("\nprompt-swap control (same pixels, different language):", flush=True)
        print(f"  {'prompt':>6} {'target obj':>12} {'placebo obj':>13} {'selectivity':>12}", flush=True)
        ratios = {}
        for prompt in sorted(prompts):
            t, p_ = mean_for(prompt, "target"), mean_for(prompt, "placebo")
            ratios[prompt] = (t / p_) if p_ > 0 else None
            shown = "n/a" if ratios[prompt] is None else f"{ratios[prompt]:.2f}x"
            print(f"  {prompt:>6} {t:>12.4f} {p_:>13.4f} {shown:>12}", flush=True)
        task_ratio, alt_ratio = ratios.get("task"), ratios.get("alt")
        grounded = bool(grounding) and float(np.mean(grounding)) >= 0.01
        if task_ratio and alt_ratio:
            if alt_ratio < 1.0 < task_ratio:
                print(
                    "  Selectivity FLIPS with the prompt: the response follows the object the "
                    "language refers to, not its screen position.",
                    flush=True,
                )
            elif alt_ratio >= task_ratio:
                print(
                    f"  Selectivity does not follow the prompt; it strengthens for the same object "
                    f"({task_ratio:.2f}x -> {alt_ratio:.2f}x) even though the alt prompt refers to "
                    "the other one.",
                    flush=True,
                )
                if grounded:
                    print(
                        "  Since the policy does react to the language, this is evidence the "
                        "response is tied to the object or its position rather than to the "
                        "linguistic referent.",
                        flush=True,
                    )
                else:
                    print(
                        "  But the policy barely reacts to the language here, so this does NOT "
                        "establish position over relevance: the manipulation may simply not have "
                        "landed.",
                        flush=True,
                    )
            else:
                print(
                    f"  Selectivity weakens under the alt prompt ({task_ratio:.2f}x -> {alt_ratio:.2f}x) "
                    "but does not invert, so language and position both contribute.",
                    flush=True,
                )

    placebos = [m for m in measurements if m["kind"] == "placebo"]
    if placebos:
        placebo = float(np.mean([m["latent"].get("l2_delta_mean") or 0.0 for m in placebos]))
        print(f"placebo-object latent L2:                      {placebo:.6g}", flush=True)
        print(
            "If target and placebo responses are similar, the features are tracking generic pixel "
            "change rather than this specific object.",
            flush=True,
        )


if __name__ == "__main__":
    main()

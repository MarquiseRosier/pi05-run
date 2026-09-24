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
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pi05_mi.langfuse_tracing import make_langfuse_tracer, summarize_action_tensor  # noqa: E402
from pi05_mi.patch_pi05 import Pi05TranscoderContext, install_pi05_action_expert_wrappers  # noqa: E402
from pi05_mi.scene_perturbation import (  # noqa: E402
    blend_geom_color,
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

    parser.add_argument("--list-objects", action="store_true", help="Print the scene's objects and exit.")
    parser.add_argument("--target", default=None, help="Regex matching the body name to perturb.")
    parser.add_argument(
        "--placebo-target",
        default=None,
        help=(
            "A second object to perturb in a separate pass. Comparing its response to the "
            "primary target separates 'features track this object' from 'features track any pixel change'."
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
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--top-features", type=int, default=25, help="Top changed features to record per layer.")
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
    policy: Any
    preprocessor: Any
    env_preprocessor: Any
    prompt: str
    preprocess_observation: Callable[[dict], dict]


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


def build_harness(args: argparse.Namespace) -> Harness:
    ensure_libero_config()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.scripts.lerobot_eval import preprocess_observation

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

    env_cfg = LiberoEnv(task=args.suite, task_ids=[args.task_id])
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    vec_env = envs[args.suite][args.task_id]

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map={})
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
    env_preprocessor, _ = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)

    inner_env = vec_env.envs[0] if hasattr(vec_env, "envs") else vec_env
    return Harness(
        vec_env=vec_env,
        inner_env=inner_env,
        policy=policy,
        preprocessor=preprocessor,
        env_preprocessor=env_preprocessor,
        prompt=args.prompt or "",
        preprocess_observation=preprocess_observation,
    )


def rerender_observation(harness: Harness) -> dict[str, Any]:
    """Re-render the cameras at the *current* sim state without stepping physics.

    This is the operation the whole design depends on: it must reflect a model
    mutation made a moment ago while leaving qpos/qvel untouched.
    """
    inner = harness.inner_env
    raw = inner._env.env._get_observations()
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


def sample_shared_noise(policy: Any, batch_size: int, device: torch.device) -> torch.Tensor:
    config = policy.model.config
    return torch.normal(
        mean=0.0,
        std=1.0,
        size=(batch_size, config.chunk_size, config.max_action_dim),
        dtype=torch.float32,
        device=device,
    )


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
    pixels = observation["pixels"]
    return np.asarray(next(iter(pixels.values())))


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

    print("building LIBERO env and Pi0.5 policy", flush=True)
    harness = build_harness(args)

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

    if args.target is None:
        raise SystemExit("--target is required (run with --list-objects first to see the object names)")
    if args.checkpoint is None:
        raise SystemExit("--checkpoint is required unless --list-objects")

    target_geoms = resolve_geom_ids(mj_model, args.target)
    print(f"\nperturbing {args.target!r} -> geom ids {target_geoms}", flush=True)

    # --- validity self-check: the re-render path must reproduce the env's own
    # observation, otherwise every measurement below compares the wrong images.
    rerendered = rerender_observation(harness)
    env_image = np.asarray(next(iter(env_observation["pixels"].values())))
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

    print(f"loading transcoders from {args.checkpoint}", flush=True)
    transcoders = load_transcoders(args.checkpoint, device)
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
        return snapshot, actions.detach().float().cpu().numpy()[0]

    doses = [float(part) for part in args.dose.split(",") if part.strip()]
    targets = [("target", args.target)]
    if args.placebo_target:
        targets.append(("placebo", args.placebo_target))

    latent_rows: list[dict[str, Any]] = []
    measurements: list[dict[str, Any]] = []
    image_dir = args.output_dir / "images"
    if args.save_images:
        image_dir.mkdir(parents=True, exist_ok=True)

    for state_index in range(args.states):
        print(f"\n=== state {state_index} ===", flush=True)
        baseline_obs = rerender_observation(harness)
        baseline_image = first_camera_image(baseline_obs)
        noise = sample_shared_noise(harness.policy, 1, device)

        baseline_latents, baseline_actions = forward(baseline_obs, noise, label=f"s{state_index}/baseline")
        null_latents, null_actions = forward(baseline_obs, noise, label=f"s{state_index}/null")

        null_rows = latent_delta_rows(
            baseline_latents, null_latents, condition="null", top_features=args.top_features
        )
        for row in null_rows:
            row["state_index"] = state_index
            row["dose"] = 0.0
            row["target"] = "none"
        latent_rows.extend(null_rows)
        null_summary = summarize_latent_rows(null_rows)
        print(
            f"  null control  latent L2 mean={null_summary.get('l2_delta_mean'):.6g} "
            f"action rel_l2={_rel_l2(baseline_actions, null_actions):.6g}",
            flush=True,
        )
        measurements.append(
            {
                "state_index": state_index,
                "kind": "null",
                "target": "none",
                "dose": 0.0,
                "pixel": image_delta_stats(baseline_image, baseline_image),
                "latent": null_summary,
                "action_relative_l2": _rel_l2(baseline_actions, null_actions),
            }
        )

        if args.save_images:
            save_image(image_dir / f"state{state_index}_baseline.png", baseline_image)

        for kind, target in targets:
            for dose in doses:
                perturbation = apply_perturbation(mj_model, args, target, dose)
                try:
                    perturbed_obs = rerender_observation(harness)
                    perturbed_image = first_camera_image(perturbed_obs)
                    pixel = image_delta_stats(baseline_image, perturbed_image)
                    if pixel["changed_pixel_fraction"] == 0.0:
                        raise RuntimeError(
                            f"Perturbation {perturbation.label!r} changed no pixels. The object is "
                            "probably occluded or outside this camera's view; pick another target."
                        )
                    perturbed_latents, perturbed_actions = forward(
                        perturbed_obs, noise, label=f"s{state_index}/{kind}/{target}/dose{dose:g}"
                    )
                finally:
                    perturbation.revert(mj_model)

                rows = latent_delta_rows(
                    baseline_latents, perturbed_latents, condition=kind, top_features=args.top_features
                )
                for row in rows:
                    row["state_index"] = state_index
                    row["dose"] = dose
                    row["target"] = target
                latent_rows.extend(rows)
                summary = summarize_latent_rows(rows)
                action_rel = _rel_l2(baseline_actions, perturbed_actions)
                print(
                    f"  {kind:<8} {target:<28} dose={dose:<5g} "
                    f"pixels={pixel['changed_pixel_fraction']:.4f} "
                    f"latentL2={summary.get('l2_delta_mean'):.6g} "
                    f"action_rel_l2={action_rel:.6g}",
                    flush=True,
                )
                measurements.append(
                    {
                        "state_index": state_index,
                        "kind": kind,
                        "target": target,
                        "dose": dose,
                        "perturbation": perturbation.label,
                        "pixel": pixel,
                        "latent": summary,
                        "action_relative_l2": action_rel,
                    }
                )
                if args.save_images:
                    save_image(image_dir / f"state{state_index}_{kind}_dose{dose:g}.png", perturbed_image)
                    diff = np.abs(
                        perturbed_image.astype(np.float64) - baseline_image.astype(np.float64)
                    )
                    if diff.max() > 0:
                        diff = diff / diff.max() * 255.0
                    save_image(image_dir / f"state{state_index}_{kind}_dose{dose:g}_diff.png", diff)

        # Advance the (unperturbed) trajectory so the next measurement sits at a
        # genuinely different state. The perturbation is already reverted here.
        for _ in range(args.state_stride):
            step_action = np.asarray(baseline_actions[0], dtype=np.float32)[None, ...]
            harness.vec_env.step(step_action)

    payload = {
        "config": {
            "suite": args.suite,
            "task_id": args.task_id,
            "seed": args.seed,
            "prompt": harness.prompt,
            "target": args.target,
            "placebo_target": args.placebo_target,
            "perturbation": args.perturbation,
            "color": args.color,
            "doses": doses,
            "states": args.states,
            "state_stride": args.state_stride,
            "num_inference_steps": args.num_inference_steps,
            "checkpoint": str(args.checkpoint),
        },
        "rerender_self_check": check,
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

    _print_verdict(measurements)
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


def _print_verdict(measurements: list[dict[str, Any]]) -> None:
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

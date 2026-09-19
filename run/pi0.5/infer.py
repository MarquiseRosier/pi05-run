#!/usr/bin/env python3
"""Suite/task/prompt-controlled Pi0.5 LIBERO inference.

Scene is chosen by ``suite`` + ``task_id`` (0-9). The VLA language can be
any string; leave it empty to use the official LIBERO sentence. Success is
still the official LIBERO predicate, not a judgment of the custom prompt.

Runs on Windows (MuJoCo ``wgl``) and Linux/Colab (``egl``).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import site
import sys
import time
import traceback
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
OFFICIAL_TASKS_PATH = HERE / "official_tasks.yaml"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
DEFAULT_MAX_STEPS = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}


def _repo_sys_path() -> None:
    for path in (
        REPO_ROOT / "src",
        REPO_ROOT / "cloud" / "libero" / "transcoder_runtime",
    ):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def load_official_tasks() -> dict[str, dict[int, str]]:
    raw = OFFICIAL_TASKS_PATH.read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError:
        yaml = None
    if yaml is not None:
        parsed = yaml.safe_load(raw) or {}
    else:
        parsed = _parse_official_tasks_fallback(raw)
    out: dict[str, dict[int, str]] = {}
    for suite, tasks in parsed.items():
        out[str(suite)] = {int(task_id): str(prompt) for task_id, prompt in tasks.items()}
    return out


def _parse_official_tasks_fallback(raw: str) -> dict[str, dict[int, str]]:
    suite = None
    parsed: dict[str, dict[int, str]] = {}
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not line.startswith(" ") and stripped.endswith(":"):
            suite = stripped[:-1]
            parsed[suite] = {}
            continue
        if suite is None or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        parsed[suite][int(key)] = value.strip()
    return parsed


def official_language(suite: str, task_id: int) -> str:
    tasks = load_official_tasks()
    if suite not in tasks or task_id not in tasks[suite]:
        raise KeyError(f"Unknown suite/task: {suite} / {task_id}")
    return tasks[suite][task_id]


def load_config(path: Path | None) -> dict[str, Any]:
    defaults = {
        "suite": "libero_spatial",
        "task_ids": [0],
        "episodes": 1,
        "seed": 1000,
        "prompt": "",
        "mode": "probe",
        "policy_path": "lerobot/pi05_libero_finetuned",
        "transcoder_checkpoint": os.environ.get("PI05_TRANSCODER_CHECKPOINT", ""),
        "n_action_steps": 10,
        "max_steps": None,
        "device": "cuda",
        "dtype": "bfloat16",
        "save_video": True,
        "capture_transcoder_latents": True,
        "transcoder_top_k": 64,
        "transcoder_max_chunks": 80,
        "save_full_latents": False,
        "atlas_save_features": True,
        "ablate_features": "",
        "ablate_layers": "",
        "steer_features": "",
        "steer_strength": 0.0,
        "output_root": str(REPO_ROOT / "outputs" / "pi05_infer"),
    }
    if path is None:
        return defaults
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required to read config.yaml") from exc
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    defaults.update(loaded)
    return defaults


def _parse_task_ids(value: Any) -> list[int]:
    if value is None or value == "":
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def cmd_list_tasks(suite: str | None) -> int:
    tasks = load_official_tasks()
    suites = [suite] if suite else list(SUITES)
    for name in suites:
        if name not in tasks:
            print(f"Unknown suite: {name}", file=sys.stderr)
            return 2
        print(f"[{name}]  scene is fixed by task_id; prompt below is official default")
        for task_id in range(10):
            print(f"  {task_id}: {tasks[name][task_id]}")
        print()
    return 0


def find_libero_root() -> Path:
    roots = [Path(path) for path in site.getsitepackages()]
    user_site = site.getusersitepackages()
    if user_site:
        roots.append(Path(user_site))
    for root in roots:
        candidate = root / "libero" / "libero"
        if candidate.exists():
            return candidate
    raise RuntimeError("Could not find installed LIBERO package path")


def ensure_libero_config(libero_root: Path) -> None:
    config_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", Path.home() / ".libero"))
    dataset_dir = Path(os.environ.get("LIBERO_DATASET_DIR", REPO_ROOT / "data" / "libero" / "datasets"))
    config_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.yaml"
    if config_path.exists():
        return
    config_path.write_text(
        "\n".join(
            [
                f"assets: {libero_root / 'assets'}",
                f"bddl_files: {libero_root / 'bddl_files'}",
                f"benchmark_root: {libero_root}",
                f"datasets: {dataset_dir}",
                f"init_states: {libero_root / 'init_files'}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def ensure_libero_assets() -> Path:
    libero_root = find_libero_root()
    ensure_libero_config(libero_root)
    assets_dir = libero_root / "assets"
    required = assets_dir / "scenes" / "libero_tabletop_base_style.xml"
    if required.exists():
        print(f"libero_assets OK {required}", flush=True)
        return libero_root

    from huggingface_hub import snapshot_download

    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    cache_dir = os.environ.get("HF_HUB_CACHE") or str(
        Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"
    )
    print("LIBERO assets missing; installing lerobot/libero-assets", flush=True)
    snapshot = Path(
        snapshot_download(
            repo_id="lerobot/libero-assets",
            repo_type="dataset",
            cache_dir=cache_dir,
            local_files_only=offline,
            token=os.environ.get("HF_TOKEN") or None,
        )
    )
    assets_dir.mkdir(parents=True, exist_ok=True)
    for child in snapshot.iterdir():
        if child.name == ".gitattributes":
            continue
        target = assets_dir / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            shutil.copy2(child, target)
    if not required.exists():
        raise FileNotFoundError(f"LIBERO asset install failed; missing {required}")
    print(f"libero_assets OK {required}", flush=True)
    return libero_root


def setup_runtime_env(cfg: dict[str, Any], run_dir: Path) -> None:
    if os.name == "nt":
        os.environ.setdefault("MUJOCO_GL", "wgl")
    else:
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    mode = str(cfg["mode"]).strip().lower()
    os.environ["PI05_TRANSCODER_MODE"] = mode
    os.environ["PI05_TRANSCODER_CHECKPOINT"] = str(cfg.get("transcoder_checkpoint") or "")
    os.environ["PI05_TRANSCODER_CAPTURE_DIR"] = str(run_dir / "transcoder_capture")
    os.environ["PI05_TRANSCODER_CAPTURE_LATENTS"] = "1" if cfg.get("capture_transcoder_latents") else "0"
    os.environ["PI05_TRANSCODER_TOP_K"] = str(cfg.get("transcoder_top_k", 64))
    os.environ["PI05_TRANSCODER_MAX_CHUNKS"] = str(cfg.get("transcoder_max_chunks", 80))
    os.environ["PI05_TRANSCODER_SAVE_FULL_LATENTS"] = "1" if cfg.get("save_full_latents") else "0"
    os.environ["PI05_TRANSCODER_DTYPE"] = str(cfg.get("dtype", "auto"))
    os.environ["PI05_ATLAS_SAVE_FEATURES"] = "1" if cfg.get("atlas_save_features", True) else "0"
    os.environ["PI05_ATLAS_SUITE"] = str(cfg.get("suite") or "")
    os.environ["PI05_TRANSCODER_ABLATE_FEATURES"] = str(cfg.get("ablate_features") or "")
    os.environ["PI05_TRANSCODER_ABLATE_LAYERS"] = str(cfg.get("ablate_layers") or "")
    os.environ["PI05_TRANSCODER_STEER_FEATURES"] = str(cfg.get("steer_features") or "")
    os.environ["PI05_TRANSCODER_STEER_STRENGTH"] = str(cfg.get("steer_strength") or 0)


def maybe_start_transcoder(mode: str) -> None:
    if mode not in {"probe", "replace"}:
        return
    _repo_sys_path()
    import pi05_transcoder_runtime  # noqa: F401


def write_image(path: Path, image: Any) -> None:
    import cv2
    import numpy as np

    arr = image
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.size and float(arr.max()) <= 1.5:
        arr = np.clip(arr, 0, 1) * 255
    arr = np.clip(arr, 0, 255).astype("uint8")
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


def render_frame(env: Any) -> Any:
    if hasattr(env, "envs"):
        return env.envs[0].render()
    frames = env.call("render")
    return frames[0]


def to_env_action(action: Any):
    import numpy as np

    if isinstance(action, dict):
        action = action.get("action", next(iter(action.values())))
    if hasattr(action, "detach"):
        action = action.detach().cpu().numpy()
    action = np.asarray(action)
    if action.ndim == 1:
        action = action[None, ...]
    return action


def first_item(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def as_bool(value: Any) -> bool:
    value = first_item(value)
    if value is None:
        return False
    if hasattr(value, "item"):
        return bool(value.item())
    return bool(value)


def extract_success(info: Any, terminated: Any) -> bool:
    info = first_item(info) or {}
    if isinstance(info, dict):
        for key in ("success", "is_success", "task_success"):
            if key in info:
                return as_bool(info[key])
        final_info = first_item(info.get("final_info"))
        if isinstance(final_info, dict):
            for key in ("success", "is_success", "task_success"):
                if key in final_info:
                    return as_bool(final_info[key])
    return as_bool(terminated)


def extract_task_text(obs: dict[str, Any]) -> str:
    task = obs.get("task")
    task = first_item(task)
    return str(task) if task is not None else ""


def write_video(path: Path, frames: list[Any], fps: int = 10) -> None:
    if not frames:
        return
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = []
    for frame in frames:
        arr = frame
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        arr = np.asarray(arr)
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.moveaxis(arr, 0, -1)
        if arr.size and float(arr.max()) <= 1.5:
            arr = np.clip(arr, 0, 1) * 255
        cleaned.append(np.clip(arr, 0, 255).astype("uint8"))
    try:
        import imageio.v2 as imageio

        imageio.mimsave(path, cleaned, fps=fps)
        return
    except Exception:
        pass
    import cv2

    height, width = cleaned[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in cleaned:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def rollout_episode(
    *,
    env: Any,
    policy: Any,
    env_preprocessor: Any,
    preprocessor: Any,
    postprocessor: Any,
    prompt: str,
    seed: int,
    max_steps: int,
    save_video: bool,
    episode_dir: Path,
) -> dict[str, Any]:
    from lerobot.scripts.lerobot_eval import preprocess_observation
    import numpy as np
    import torch

    if hasattr(policy, "reset"):
        policy.reset()

    obs, info = env.reset(seed=[seed])
    env_language = extract_task_text(obs)
    frames: list[Any] = []
    if save_video:
        try:
            frames.append(render_frame(env))
        except Exception as exc:
            print(f"render failed at reset: {type(exc).__name__}: {exc}", flush=True)

    write_image(episode_dir / "reset.png", frames[0] if frames else render_frame(env))

    success = False
    reward_sum = 0.0
    step = 0
    terminated = False
    truncated = False
    while step < max_steps:
        obs = preprocess_observation(obs)
        obs["task"] = [prompt]
        batch = env_preprocessor(obs)
        batch = preprocessor(batch)
        with torch.inference_mode():
            action = policy.select_action(batch)
        action = postprocessor(action)
        obs, reward, terminated, truncated, info = env.step(to_env_action(action))
        reward_sum += float(np.asarray(reward).reshape(-1)[0])
        step += 1
        if save_video:
            try:
                frames.append(render_frame(env))
            except Exception:
                pass
        success = extract_success(info, terminated)
        done = bool(np.asarray(terminated).any()) or bool(np.asarray(truncated).any())
        if done:
            break

    video_path = None
    if save_video and frames:
        video_path = episode_dir / "video.mp4"
        save_video_file = video_path
        write_video(save_video_file, frames)
        video_path = str(video_path.relative_to(episode_dir.parent.parent))

    summary = {
        "seed": seed,
        "steps": step,
        "max_steps": max_steps,
        "success": bool(success),
        "reward_sum": reward_sum,
        "env_language": env_language,
        "vla_prompt": prompt,
        "terminated": as_bool(terminated),
        "truncated": as_bool(truncated),
        "video": video_path,
    }
    (episode_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def make_run_dir(output_root: Path) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    run_dir = output_root / stamp
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"{stamp}-{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def cmd_run(cfg: dict[str, Any]) -> int:
    suite = str(cfg["suite"])
    if suite not in SUITES:
        raise SystemExit(f"suite must be one of {SUITES}, got {suite!r}")
    task_ids = _parse_task_ids(cfg.get("task_ids"))
    if not task_ids:
        raise SystemExit("task_ids is empty; pass e.g. --task-id 3 or task_ids: [0, 1]")
    for task_id in task_ids:
        if task_id < 0 or task_id > 9:
            raise SystemExit(f"task_id must be 0-9, got {task_id}")

    mode = str(cfg["mode"]).strip().lower()
    if mode not in {"original", "probe", "replace"}:
        raise SystemExit(f"mode must be original|probe|replace, got {mode!r}")
    if mode in {"probe", "replace"} and not cfg.get("transcoder_checkpoint"):
        raise SystemExit("probe/replace require --transcoder-checkpoint or config transcoder_checkpoint")

    prompt_override = str(cfg.get("prompt") or "").strip()
    episodes = int(cfg["episodes"])
    seed0 = int(cfg["seed"])
    max_steps = int(cfg["max_steps"] or DEFAULT_MAX_STEPS[suite])
    output_root = Path(cfg["output_root"]).expanduser()
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    run_dir = make_run_dir(output_root)
    (run_dir / "config.resolved.json").write_text(
        json.dumps({**cfg, "task_ids": task_ids, "max_steps": max_steps, "run_dir": str(run_dir)}, indent=2),
        encoding="utf-8",
    )

    print(f"run_dir={run_dir}", flush=True)
    print(f"suite={suite} task_ids={task_ids} episodes={episodes} mode={mode}", flush=True)
    print(f"prompt_override={prompt_override or '(official LIBERO language)'}", flush=True)

    setup_runtime_env(cfg, run_dir)
    print(f"MUJOCO_GL={os.environ.get('MUJOCO_GL')}", flush=True)
    ensure_libero_assets()
    maybe_start_transcoder(mode)

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    policy_path = str(cfg["policy_path"])
    policy_cfg = PreTrainedConfig.from_pretrained(
        policy_path,
        cache_dir=os.environ.get("HF_HUB_CACHE"),
        local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1",
    )
    policy_cfg.pretrained_path = Path(policy_path)
    policy_cfg.device = str(cfg["device"])
    policy_cfg.dtype = str(cfg["dtype"])
    policy_cfg.compile_model = False
    policy_cfg.gradient_checkpointing = False
    policy_cfg.n_action_steps = int(cfg["n_action_steps"])

    # Build policy against the first task; observation space is shared in LIBERO.
    first_env_cfg = LiberoEnv(task=suite, task_ids=[task_ids[0]])
    policy = make_policy(cfg=policy_cfg, env_cfg=first_env_cfg, rename_map={})
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_path,
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": {}},
        },
    )
    env_preprocessor, _env_postprocessor = make_env_pre_post_processors(
        env_cfg=first_env_cfg,
        policy_cfg=policy_cfg,
    )

    episode_summaries: list[dict[str, Any]] = []
    per_task: dict[str, dict[str, Any]] = {}
    n_success = 0
    try:
        for task_id in task_ids:
            official = official_language(suite, task_id)
            prompt = prompt_override or official
            env_cfg = LiberoEnv(task=suite, task_ids=[task_id])
            envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
            env = envs[suite][task_id]
            task_success = 0
            try:
                for ep in range(episodes):
                    seed = seed0 + ep
                    episode_dir = run_dir / "episodes" / f"task_{task_id}_ep{ep}"
                    episode_dir.mkdir(parents=True, exist_ok=True)
                    print(
                        f"rollout suite={suite} task_id={task_id} ep={ep} seed={seed}\n"
                        f"  official: {official}\n"
                        f"  vla_prompt: {prompt}",
                        flush=True,
                    )
                    if mode in {"probe", "replace"}:
                        import pi05_transcoder_runtime as transcoder_runtime

                        transcoder_runtime.atlas_begin_episode(suite=suite, task_id=task_id, episode=ep)
                    summary = rollout_episode(
                        env=env,
                        policy=policy,
                        env_preprocessor=env_preprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        prompt=prompt,
                        seed=seed,
                        max_steps=max_steps,
                        save_video=bool(cfg.get("save_video", True)),
                        episode_dir=episode_dir,
                    )
                    summary.update(
                        {
                            "suite": suite,
                            "task_id": task_id,
                            "episode": ep,
                            "official_language": official,
                        }
                    )
                    episode_summaries.append(summary)
                    if summary["success"]:
                        task_success += 1
                        n_success += 1
                    print(
                        f"  done success={summary['success']} steps={summary['steps']}",
                        flush=True,
                    )
                    if mode in {"probe", "replace"}:
                        import pi05_transcoder_runtime as transcoder_runtime

                        transcoder_runtime.atlas_end_episode()
            finally:
                try:
                    env.close()
                except Exception as exc:
                    print(f"env.close failed: {type(exc).__name__}: {exc}", flush=True)
            n_eps = episodes
            per_task[str(task_id)] = {
                "task_id": task_id,
                "official_language": official,
                "vla_prompt": prompt,
                "n_episodes": n_eps,
                "n_successes": task_success,
                "pc_success": 100.0 * task_success / n_eps if n_eps else 0.0,
            }
    except Exception:
        traceback.print_exc()
        (run_dir / "crash.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return 1

    n_episodes = len(episode_summaries)
    eval_info = {
        "suite": suite,
        "task_ids": task_ids,
        "mode": mode,
        "prompt_override": prompt_override,
        "policy_path": cfg["policy_path"],
        "transcoder_checkpoint": cfg.get("transcoder_checkpoint") or "",
        "n_action_steps": int(cfg["n_action_steps"]),
        "max_steps": max_steps,
        "overall": {
            "n_episodes": n_episodes,
            "n_successes": n_success,
            "pc_success": 100.0 * n_success / n_episodes if n_episodes else 0.0,
        },
        "per_task": per_task,
        "episodes": episode_summaries,
        "run_dir": str(run_dir),
    }
    (run_dir / "eval_info.json").write_text(json.dumps(eval_info, indent=2), encoding="utf-8")
    print(json.dumps(eval_info["overall"], indent=2), flush=True)
    print(f"saved {run_dir / 'eval_info.json'}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    listing = sub.add_parser("list-tasks", help="Print official LIBERO prompts for a suite")
    listing.add_argument("--suite", choices=SUITES, default=None)

    run = sub.add_parser("run", help="Roll out Pi0.5 on one or more LIBERO scenes")
    run.add_argument("--config", type=Path, default=HERE / "config.yaml")
    run.add_argument("--suite", choices=SUITES)
    run.add_argument("--task-id", dest="task_ids", action="append", type=int, help="Repeatable. Example: --task-id 1 --task-id 3")
    run.add_argument("--task-ids", dest="task_ids_csv", default=None, help="Comma list, e.g. 0,1,3")
    run.add_argument("--prompt", default=None, help="VLA language. Empty/omit = official LIBERO sentence")
    run.add_argument("--episodes", type=int)
    run.add_argument("--seed", type=int)
    run.add_argument("--mode", choices=("original", "probe", "replace"))
    run.add_argument("--policy-path")
    run.add_argument("--transcoder-checkpoint")
    run.add_argument("--max-steps", type=int)
    run.add_argument("--n-action-steps", type=int)
    run.add_argument("--device")
    run.add_argument("--dtype")
    run.add_argument("--output-root", type=Path)
    run.add_argument("--no-video", action="store_true")
    run.add_argument("--ablate-features", help="Comma feature ids to zero in transcoder z")
    run.add_argument("--ablate-layers", help="Comma action-expert layer indices, empty=all")
    run.add_argument("--steer-features", help="Comma feature ids to scale in transcoder z")
    run.add_argument("--steer-strength", type=float, help="Multiply steered features by 1+strength")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list-tasks":
        return cmd_list_tasks(args.suite)

    cfg = load_config(args.config)
    if args.suite:
        cfg["suite"] = args.suite
    if args.task_ids_csv:
        cfg["task_ids"] = _parse_task_ids(args.task_ids_csv)
    elif args.task_ids:
        cfg["task_ids"] = args.task_ids
    if args.prompt is not None:
        cfg["prompt"] = args.prompt
    if args.episodes is not None:
        cfg["episodes"] = args.episodes
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.mode:
        cfg["mode"] = args.mode
    if args.policy_path:
        cfg["policy_path"] = args.policy_path
    if args.transcoder_checkpoint:
        cfg["transcoder_checkpoint"] = args.transcoder_checkpoint
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    if args.n_action_steps is not None:
        cfg["n_action_steps"] = args.n_action_steps
    if args.device:
        cfg["device"] = args.device
    if args.dtype:
        cfg["dtype"] = args.dtype
    if args.output_root:
        cfg["output_root"] = str(args.output_root)
    if args.no_video:
        cfg["save_video"] = False
    if args.ablate_features is not None:
        cfg["ablate_features"] = args.ablate_features
    if args.ablate_layers is not None:
        cfg["ablate_layers"] = args.ablate_layers
    if args.steer_features is not None:
        cfg["steer_features"] = args.steer_features
    if args.steer_strength is not None:
        cfg["steer_strength"] = args.steer_strength
    return cmd_run(cfg)


if __name__ == "__main__":
    raise SystemExit(main())

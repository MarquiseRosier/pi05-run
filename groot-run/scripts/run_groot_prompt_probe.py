#!/usr/bin/env python3
"""Reset one LIBERO task, swap the language prompt, request one GR00T action chunk."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def _save_image(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = array
    while arr.ndim > 3:
        arr = arr[0]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    try:
        import cv2

        cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    except Exception:
        np.save(path.with_suffix(".npy"), arr)


def _modality_keys(modality) -> list[str]:
    keys = getattr(modality, "modality_keys", None)
    if keys is None:
        keys = modality["modality_keys"]
    return list(keys)


def _delta_len(modality) -> int:
    indices = getattr(modality, "delta_indices", None)
    if indices is None:
        indices = modality["delta_indices"]
    return len(list(indices))


def _as_hwc_uint8(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value)
    while arr.ndim > 3:
        arr = arr[0]
    if arr.ndim != 3:
        raise SystemExit(f"Expected HWC image, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _as_state_vector(value) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def _batch_for_policy(observation: dict, modality_config, language: str) -> dict:
    """Match official eval: MultiStepWrapper (T) then SyncVectorEnv (B=1)."""
    out = dict(observation)
    video_t = max(1, _delta_len(modality_config["video"]))
    state_t = max(1, _delta_len(modality_config["state"]))

    for key in _modality_keys(modality_config["video"]):
        flat = f"video.{key}"
        if flat not in out:
            raise SystemExit(f"Missing video key {flat}")
        frame = _as_hwc_uint8(out[flat])
        out[flat] = np.stack([frame] * video_t, axis=0)[np.newaxis, ...]

    for key in _modality_keys(modality_config["state"]):
        flat = f"state.{key}"
        if flat not in out:
            raise SystemExit(f"Missing state key {flat}")
        vec = _as_state_vector(out[flat])
        out[flat] = np.stack([vec] * state_t, axis=0)[np.newaxis, ...]

    lang_keys = _modality_keys(modality_config["language"])
    aliases = (
        "task",
        "annotation.human.coarse_action",
        "annotation.human.action.task_description",
    )
    for key in [*lang_keys, *aliases]:
        if key in out or key in lang_keys:
            out[key] = (language,)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="GR00T one-chunk plaintext prompt probe.")
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--policy-client-host", default="127.0.0.1")
    parser.add_argument("--policy-client-port", type=int, default=5555)
    parser.add_argument("--n-action-steps", type=int, default=8)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MPLBACKEND", "Agg")

    from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs
    from gr00t.policy.server_client import PolicyClient
    import gymnasium as gym

    register_libero_envs()
    env = gym.make(args.env_name)
    observation, info = env.reset(seed=args.seed)
    if isinstance(observation, tuple):
        observation = observation[0]

    if not isinstance(observation, dict):
        raise SystemExit(f"Unexpected observation type: {type(observation)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    render = None
    try:
        render = env.render()
    except Exception:
        render = info.get("agentview_image") if isinstance(info, dict) else None
    if isinstance(render, np.ndarray):
        _save_image(out_dir / "render.png", render)

    image_index = 0
    for key, value in observation.items():
        if not isinstance(value, np.ndarray):
            continue
        if "video" not in str(key).lower() and "image" not in str(key).lower():
            continue
        name = "observation_images_image" if image_index == 0 else "observation_images_image2"
        _save_image(out_dir / f"{name}.png", value)
        image_index += 1
        if image_index >= 2:
            break

    client = PolicyClient(host=args.policy_client_host, port=args.policy_client_port)
    observation = _batch_for_policy(observation, client.get_modality_config(), args.language)
    action, _info = client.get_action(observation)
    payload = {}
    for key, value in action.items():
        if isinstance(value, np.ndarray):
            arr = value
            while arr.ndim > 2:
                arr = arr[0]
            payload[key] = [[round(float(x), 6) for x in row] for row in arr[: args.n_action_steps].tolist()]
        else:
            payload[key] = value
    (out_dir / "action_chunk.json").write_text(
        json.dumps(
            {
                "env_name": args.env_name,
                "language": args.language,
                "seed": args.seed,
                "n_action_steps": args.n_action_steps,
                "action": payload,
            },
            indent=2,
        )
    )
    env.close()
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

    language_key = "task" if "task" in observation else "annotation.human.coarse_action"
    if language_key in observation:
        value = observation[language_key]
        if isinstance(value, (list, tuple)):
            observation[language_key] = type(value)([args.language] * len(value))
        else:
            observation[language_key] = args.language

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

#!/usr/bin/env python3
"""Language intervention probe: same cameras/state, two prompts, compare each 8-step chunk.

The scene follows the original LIBERO instruction. At every replan the client also
asks what the policy would have output under --language, then writes the per-step
delta. That contrast is the probe; a single swapped-prompt inference is not.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

ACTION_KEYS = (
    "action.x",
    "action.y",
    "action.z",
    "action.roll",
    "action.pitch",
    "action.yaw",
    "action.gripper",
)

LANGUAGE_KEYS = (
    "annotation.human.action.task_description",
    "annotation.human.task_description",
    "annotation.human.coarse_action",
    "task",
)


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
    for key in [*lang_keys, *LANGUAGE_KEYS]:
        if key in out or key in lang_keys:
            out[key] = (language,)
    return out


def _native_language(observation: dict) -> str:
    for key in LANGUAGE_KEYS:
        if key not in observation:
            continue
        value = observation[key]
        if isinstance(value, (list, tuple)):
            value = value[0]
        if isinstance(value, np.ndarray):
            value = value.reshape(-1)[0]
        return str(value)
    raise SystemExit("Observation has no language key")


def _flatten_chunk(action: dict, n_steps: int) -> dict[str, list[float]]:
    payload: dict[str, list[float]] = {}
    for key in ACTION_KEYS:
        if key not in action:
            raise SystemExit(f"Policy action missing {key}")
        arr = np.asarray(action[key], dtype=np.float64)
        while arr.ndim > 1:
            if arr.shape[0] == 1:
                arr = arr[0]
            else:
                arr = arr.reshape(arr.shape[0], -1)[:, 0]
                break
        arr = np.asarray(arr, dtype=np.float64).reshape(-1)
        if arr.size < n_steps:
            raise SystemExit(f"{key} has {arr.size} steps, need {n_steps}")
        payload[key] = [round(float(x), 6) for x in arr[:n_steps]]
    return payload


def _query_chunk(client, modality_config, observation: dict, language: str, n_steps: int) -> dict[str, list[float]]:
    batched = _batch_for_policy(observation, modality_config, language)
    action, _info = client.get_action(batched)
    return _flatten_chunk(action, n_steps)


def _chunk_delta(original: dict[str, list[float]], probe: dict[str, list[float]]) -> tuple[dict, dict, float]:
    delta = {}
    abs_mean = {}
    for key in ACTION_KEYS:
        values = [round(probe[key][i] - original[key][i], 6) for i in range(len(original[key]))]
        delta[key] = values
        abs_mean[key] = round(float(np.mean(np.abs(values))), 6)
    stacked = np.asarray([delta[key] for key in ACTION_KEYS], dtype=np.float64)
    return delta, abs_mean, round(float(np.linalg.norm(stacked)), 6)


def _env_action(chunk: dict[str, list[float]], step: int) -> dict[str, np.ndarray]:
    return {key: np.asarray([chunk[key][step]], dtype=np.float32) for key in ACTION_KEYS}


def _write_compare_md(path: Path, payload: dict) -> None:
    lines = [
        "# Language intervention probe",
        "",
        f"- env: `{payload['env_name']}`",
        f"- original language: `{payload['original_language']}`",
        f"- intervention language: `{payload['probe_language']}`",
        f"- seed: {payload['seed']}",
        f"- chunk length: {payload['n_action_steps']} actions per replan",
        "- execution: original-language chunks, so the cameras follow the real task",
        "- contrast: same image/state, only the language changes",
        "",
    ]
    if payload["same_language"]:
        lines += [
            "WARNING: intervention language equals the original task prompt.",
            "Deltas should be near zero. Change `PROBE_LANGUAGE` to probe anything.",
            "",
        ]
    for row in payload["replans"]:
        start = row["replan"] * payload["n_action_steps"]
        end = start + payload["n_action_steps"] - 1
        lines += [
            f"## Replan {row['replan']} (env steps {start}–{end})",
            "",
            f"- chunk L2 |original − intervention|: `{row['l2']}`",
            "- mean |Δ| per dim: "
            + ", ".join(f"`{key.split('.')[-1]}={row['mean_abs_delta'][key]}`" for key in ACTION_KEYS),
            "",
            "| dim | original 8 steps | intervention 8 steps | Δ |",
            "|---|---|---|---|",
        ]
        for key in ACTION_KEYS:
            name = key.split(".")[-1]
            orig = ", ".join(f"{x:.4f}" for x in row["original"][key])
            probe = ", ".join(f"{x:.4f}" for x in row["probe"][key])
            delta = ", ".join(f"{x:+.4f}" for x in row["delta"][key])
            lines.append(f"| {name} | `{orig}` | `{probe}` | `{delta}` |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="GR00T language-intervention probe across replans.")
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--language", required=True, help="Intervention prompt compared against the env's native language.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--policy-client-host", default="127.0.0.1")
    parser.add_argument("--policy-client-port", type=int, default=5555)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--n-replans", type=int, default=4, help="How many 8-step planning windows to compare.")
    args = parser.parse_args()
    if args.n_replans < 1:
        raise SystemExit("--n-replans must be >= 1")

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

    original_language = _native_language(observation)
    probe_language = args.language
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
    modality_config = client.get_modality_config()
    replans = []
    for replan in range(args.n_replans):
        original_chunk = _query_chunk(
            client, modality_config, observation, original_language, args.n_action_steps
        )
        probe_chunk = _query_chunk(
            client, modality_config, observation, probe_language, args.n_action_steps
        )
        delta, abs_mean, l2 = _chunk_delta(original_chunk, probe_chunk)
        replans.append(
            {
                "replan": replan,
                "original": original_chunk,
                "probe": probe_chunk,
                "delta": delta,
                "mean_abs_delta": abs_mean,
                "l2": l2,
            }
        )
        done = False
        for step in range(args.n_action_steps):
            observation, _reward, terminated, truncated, info = env.step(_env_action(original_chunk, step))
            if isinstance(observation, tuple):
                observation = observation[0]
            done = bool(terminated or truncated or (isinstance(info, dict) and info.get("success")))
            if done:
                break
        if done:
            break

    payload = {
        "env_name": args.env_name,
        "original_language": original_language,
        "probe_language": probe_language,
        "same_language": original_language.strip().lower() == probe_language.strip().lower(),
        "seed": args.seed,
        "n_action_steps": args.n_action_steps,
        "n_replans_requested": args.n_replans,
        "n_replans_recorded": len(replans),
        "replans": replans,
    }
    (out_dir / "language_probe.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_compare_md(out_dir / "compare.md", payload)
    # Keep the first intervention chunk under the old name so leftover notebook cells still find a file.
    (out_dir / "action_chunk.json").write_text(
        json.dumps(
            {
                "env_name": args.env_name,
                "language": probe_language,
                "compared_against": original_language,
                "seed": args.seed,
                "n_action_steps": args.n_action_steps,
                "action": {key: [[v] for v in replans[0]["probe"][key]] for key in ACTION_KEYS},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    env.close()
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

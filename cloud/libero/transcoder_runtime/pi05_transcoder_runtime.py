"""Optional Pi0.5 transcoder runtime for LIBERO evaluation.

Loaded by ``sitecustomize`` when ``PI05_TRANSCODER_MODE`` is ``probe`` or
``replace``. The module patches LeRobot's ``make_policy`` function so normal
``lerobot-eval`` rollouts can run with action-expert MLP transcoders installed.
"""

from __future__ import annotations

import atexit
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

import torch

from pi05_mi.atlas_bridge import (
    atlas_activation_path,
    flatten_token_rows,
    load_sparse_features,
    pack_sparse_features,
    parse_int_list,
    save_sparse_features,
)
from pi05_mi.atlas_concepts import task_id_from_prompt
from pi05_mi.patch_pi05 import Pi05TranscoderContext, install_pi05_action_expert_wrappers
from pi05_mi.transcoders import TimeConditionedTranscoder, TimeConditionedTranscoderConfig


def _mode() -> str:
    return os.environ.get("PI05_TRANSCODER_MODE", "original").strip().lower()


def _enabled() -> bool:
    return _mode() in {"probe", "replace"}


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _dtype_for_policy(policy: Any) -> torch.dtype:
    requested = os.environ.get("PI05_TRANSCODER_DTYPE", "auto").strip().lower()
    if requested in {"float32", "fp32"}:
        return torch.float32
    if requested in {"float16", "fp16"}:
        return torch.float16
    if requested in {"bfloat16", "bf16"}:
        return torch.bfloat16

    policy_dtype = str(getattr(getattr(policy, "config", None), "dtype", "")).lower()
    if "bfloat16" in policy_dtype or "bf16" in policy_dtype:
        return torch.bfloat16
    if "float16" in policy_dtype or "fp16" in policy_dtype:
        return torch.float16
    return torch.float32


def _device_for_policy(policy: Any) -> torch.device:
    raw_device = getattr(getattr(policy, "config", None), "device", None)
    if raw_device is None:
        raw_device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(str(raw_device))


def _load_transcoders(checkpoint_path: Path, *, device: torch.device, dtype: torch.dtype) -> dict[str, TimeConditionedTranscoder]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    transcoders: dict[str, TimeConditionedTranscoder] = {}
    for name, raw_config in checkpoint["configs"].items():
        config = TimeConditionedTranscoderConfig(**raw_config)
        transcoder = TimeConditionedTranscoder(config)
        transcoder.load_state_dict(checkpoint["state_dicts"][name])
        transcoder.to(device=device, dtype=dtype)
        transcoder.eval()
        for parameter in transcoder.parameters():
            parameter.requires_grad_(False)
        transcoders[name] = transcoder
    return transcoders


def _round_list(tensor: torch.Tensor, limit: int | None = None) -> list[Any]:
    values = tensor.detach().cpu()
    if limit is not None:
        values = values.reshape(-1)[:limit]
    return json.loads(json.dumps(values.tolist(), allow_nan=False))


class TranscoderRecorder:
    def __init__(self) -> None:
        capture_dir = os.environ.get("PI05_TRANSCODER_CAPTURE_DIR", "transcoder_capture")
        self.root = Path(capture_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.full_dir = self.root / "full_latents"
        self.events_path = self.root / "events.jsonl"
        self.file = self.events_path.open("a", buffering=1)
        self.capture_latents = _bool_env("PI05_TRANSCODER_CAPTURE_LATENTS", True)
        self.max_chunks = _int_env("PI05_TRANSCODER_MAX_CHUNKS", 40)
        self.top_k = _int_env("PI05_TRANSCODER_TOP_K", 64)
        self.save_full_latents = _bool_env("PI05_TRANSCODER_SAVE_FULL_LATENTS", False)
        self.save_atlas_features = _bool_env("PI05_ATLAS_SAVE_FEATURES", True)
        self.atlas_suite = os.environ.get("PI05_ATLAS_SUITE", "").strip()
        self.atlas_root = self.root / "atlas_activations"
        if self.save_full_latents:
            self.full_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_index = -1
        self.captured_chunks = 0
        self.active = False
        self.call_index = 0
        self.current_task = ""
        self.episode_task_id: int | None = None
        self.episode_index: int | None = None
        self._prompt_to_task: dict[str, int] = {}
        self._atlas_records: dict[int, list[dict[str, Any]]] = {}
        self._write(
            {
                "type": "transcoder_capture_start",
                "time": time.time(),
                "mode": _mode(),
                "capture_latents": self.capture_latents,
                "top_k": self.top_k,
                "max_chunks": self.max_chunks,
                "save_full_latents": self.save_full_latents,
                "save_atlas_features": self.save_atlas_features,
                "atlas_suite": self.atlas_suite,
            }
        )
        atexit.register(self.close)

    def begin_episode(self, *, suite: str = "", task_id: int | None = None, episode: int | None = None) -> None:
        self.flush_atlas_episode()
        if suite:
            self.atlas_suite = suite
        self.episode_task_id = task_id
        self.episode_index = episode
        self._write(
            {
                "type": "atlas_episode_start",
                "time": time.time(),
                "suite": self.atlas_suite,
                "task_id": task_id,
                "episode": episode,
            }
        )

    def end_episode(self) -> None:
        self.flush_atlas_episode()
        self._write(
            {
                "type": "atlas_episode_end",
                "time": time.time(),
                "suite": self.atlas_suite,
                "task_id": self.episode_task_id,
                "episode": self.episode_index,
            }
        )
        self.episode_task_id = None
        self.episode_index = None

    def close(self) -> None:
        self.flush_atlas_episode()
        if not self.file.closed:
            self._write({"type": "transcoder_capture_end", "time": time.time()})
            self.file.close()

    def _write(self, payload: dict[str, Any]) -> None:
        self.file.write(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")

    def begin_chunk(self, batch: dict[str, Any], context: Pi05TranscoderContext) -> None:
        self.chunk_index += 1
        self.call_index = 0
        context.clear_records()
        self.active = self.capture_latents and (self.max_chunks <= 0 or self.captured_chunks < self.max_chunks)
        if not self.active:
            return
        self.captured_chunks += 1
        task = batch.get("task")
        new_task = ""
        if isinstance(task, str):
            new_task = task
        elif isinstance(task, (list, tuple)) and task and isinstance(task[0], str):
            new_task = task[0]
        if self.save_atlas_features and self._atlas_records and new_task and new_task != self.current_task:
            self.flush_atlas_episode()
            self.episode_task_id = None
        payload: dict[str, Any] = {
            "type": "chunk_start",
            "chunk": self.chunk_index,
            "time": time.time(),
            "mode": _mode(),
        }
        if new_task:
            payload["task"] = new_task
            self.current_task = new_task
        self._write(payload)

    def end_chunk(self, context: Pi05TranscoderContext, actions: torch.Tensor | None = None) -> None:
        if not self.active:
            context.clear_records()
            return
        if actions is not None:
            self._record_action_chunk(actions)
        self._record_latents(context)
        self._write({"type": "chunk_end", "chunk": self.chunk_index, "time": time.time()})
        context.clear_records()
        self.active = False

    def _record_action_chunk(self, actions: torch.Tensor) -> None:
        try:
            arr = actions.detach()[0].to("cpu").float()
            values = arr[:, : min(arr.shape[-1], 12)]
            self._write(
                {
                    "type": "action_chunk",
                    "chunk": self.chunk_index,
                    "shape": list(actions.shape),
                    "values": [[round(float(x), 6) for x in row] for row in values.tolist()],
                    "norm_per_step": [round(float(x), 6) for x in arr.norm(dim=-1).tolist()],
                    "abs_per_dim": [round(float(x), 6) for x in arr.abs().mean(dim=0).tolist()],
                }
            )
        except Exception as exc:
            self._write({"type": "action_error", "chunk": self.chunk_index, "error": str(exc)})

    def _record_latents(self, context: Pi05TranscoderContext) -> None:
        for records in context.latents.values():
            for record in records:
                payload = {
                    "type": "transcoder_latent",
                    "chunk": self.chunk_index,
                    "call": self.call_index,
                    "mode": record.mode,
                    "module": record.name,
                    "layer": record.layer_index,
                    "timestep": _round_list(record.timestep),
                    "shape": list(record.latent_shape),
                    "l0_mean": round(float(record.latent_l0.float().mean()), 6),
                    "l0_max": round(float(record.latent_l0.float().max()), 6),
                    "l1_mean": round(float(record.latent_l1.float().mean()), 6),
                    "l1_max": round(float(record.latent_l1.float().max()), 6),
                    "max_mean": round(float(record.latent_max.float().mean()), 6),
                    "max_value": round(float(record.latent_max.float().max()), 6),
                    "token_l0": _round_list(record.latent_l0),
                    "token_l1": _round_list(record.latent_l1),
                    "token_max": _round_list(record.latent_max),
                    "top": [
                        {
                            "index": [int(index) for index in indices],
                            "value": round(float(value), 6),
                        }
                        for indices, value in zip(record.top_indices.tolist(), record.top_values.tolist(), strict=False)
                    ],
                }
                if record.full_latent is not None:
                    path = self.full_dir / f"chunk_{self.chunk_index:06d}_call_{self.call_index:06d}_layer_{record.layer_index:02d}.pt"
                    torch.save(record.full_latent, path)
                    payload["full_latent_path"] = str(path.relative_to(self.root))
                self._write(payload)
                if self.save_atlas_features and record.token_top_indices is not None and record.token_top_values is not None:
                    self._buffer_atlas_record(record)
                self.call_index += 1

    def _buffer_atlas_record(self, record: Any) -> None:
        indices, values, timesteps = flatten_token_rows(
            record.token_top_indices,
            record.token_top_values,
            record.timestep,
        )
        packed = pack_sparse_features(
            layer_index=record.layer_index,
            d_features=record.d_features or int(record.token_top_indices.shape[-1]),
            k=int(record.token_top_indices.shape[-1]),
            indices=indices,
            values=values,
            timesteps=timesteps,
        )
        self._atlas_records.setdefault(int(record.layer_index), []).append(packed)

    def _configured_task_ids(self) -> list[int]:
        return parse_int_list(os.environ.get("PI05_ATLAS_TASK_IDS", ""))

    def _resolve_task_id(self) -> int:
        if self.episode_task_id is not None:
            return int(self.episode_task_id)
        if self.atlas_suite and self.current_task:
            matched = task_id_from_prompt(self.atlas_suite, self.current_task, space="lerobot")
            if matched is not None:
                return matched
        if self.current_task:
            if self.current_task not in self._prompt_to_task:
                configured = self._configured_task_ids()
                next_index = len(self._prompt_to_task)
                if configured and next_index < len(configured):
                    self._prompt_to_task[self.current_task] = configured[next_index]
                else:
                    self._prompt_to_task[self.current_task] = next_index
            return self._prompt_to_task[self.current_task]
        configured = self._configured_task_ids()
        if configured:
            return configured[0]
        env_task = os.environ.get("PI05_ATLAS_TASK_ID", "").strip()
        if env_task.isdigit():
            return int(env_task)
        return 0

    def _resolve_episode(self) -> int:
        if self.episode_index is not None:
            return int(self.episode_index)
        env_episode = os.environ.get("PI05_ATLAS_EPISODE", "").strip()
        if env_episode.isdigit():
            return int(env_episode)
        return 0

    def flush_atlas_episode(self) -> None:
        if not self._atlas_records:
            return
        task_id = self._resolve_task_id()
        episode = self._resolve_episode()
        written = []
        for layer_index, records in self._atlas_records.items():
            merged = {
                "dictionary": "transcoder",
                "layer": records[0]["layer"],
                "layer_index": layer_index,
                "d_features": records[0]["d_features"],
                "k": records[0]["k"],
                "n_tokens": int(sum(item["n_tokens"] for item in records)),
                "indices": torch.cat([item["indices"] for item in records], dim=0),
                "values": torch.cat([item["values"] for item in records], dim=0),
                "timesteps": torch.cat([item["timesteps"] for item in records], dim=0),
            }
            path = atlas_activation_path(self.atlas_root, task_id, episode, layer_index)
            if path.exists():
                previous = load_sparse_features(path)
                merged["indices"] = torch.cat([previous["indices"], merged["indices"]], dim=0)
                merged["values"] = torch.cat([previous["values"], merged["values"]], dim=0)
                merged["timesteps"] = torch.cat([previous["timesteps"], merged["timesteps"]], dim=0)
                merged["n_tokens"] = int(merged["indices"].shape[0])
            save_sparse_features(path, merged)
            written.append(str(path.relative_to(self.root)))
        self._write(
            {
                "type": "atlas_features_flushed",
                "time": time.time(),
                "suite": self.atlas_suite,
                "task_id": task_id,
                "episode": episode,
                "layers": sorted(self._atlas_records),
                "files": written,
            }
        )
        self._atlas_records = {}


_RECORDER = TranscoderRecorder()


def _install_on_policy(policy: Any) -> None:
    if getattr(policy, "_pi05_transcoder_runtime_installed", False):
        return
    checkpoint = os.environ.get("PI05_TRANSCODER_CHECKPOINT", "").strip()
    if not checkpoint:
        raise RuntimeError("PI05_TRANSCODER_CHECKPOINT is required for probe/replace mode")
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Transcoder checkpoint not found: {checkpoint_path}")

    mode = _mode()
    if mode not in {"probe", "replace"}:
        raise ValueError(f"Unsupported PI05_TRANSCODER_MODE={mode!r}")

    device = _device_for_policy(policy)
    dtype = _dtype_for_policy(policy)
    transcoders = _load_transcoders(checkpoint_path, device=device, dtype=dtype)
    context = Pi05TranscoderContext(
        mode=mode,  # type: ignore[arg-type]
        capture_records=False,
        capture_latents=_RECORDER.capture_latents,
        latent_top_k=_RECORDER.top_k,
        save_full_latents=_RECORDER.save_full_latents,
    )
    context.set_intervention(
        ablate_features=os.environ.get("PI05_TRANSCODER_ABLATE_FEATURES", ""),
        ablate_layers=os.environ.get("PI05_TRANSCODER_ABLATE_LAYERS", ""),
        steer_features=os.environ.get("PI05_TRANSCODER_STEER_FEATURES", ""),
        steer_strength=float(os.environ.get("PI05_TRANSCODER_STEER_STRENGTH", "0") or 0),
    )
    context, wrapped_names = install_pi05_action_expert_wrappers(
        policy,
        context=context,
        transcoders=transcoders,
        mode=mode,  # type: ignore[arg-type]
    )

    original_predict = policy.predict_action_chunk

    def wrapped_predict_action_chunk(self: Any, batch: dict[str, Any], *args: Any, **kwargs: Any):
        _RECORDER.begin_chunk(batch, context)
        actions = None
        try:
            actions = original_predict(batch, *args, **kwargs)
            return actions
        finally:
            _RECORDER.end_chunk(context, actions)

    policy.predict_action_chunk = types.MethodType(wrapped_predict_action_chunk, policy)
    policy._pi05_transcoder_runtime_installed = True
    _RECORDER._write(
        {
            "type": "transcoder_runtime_installed",
            "time": time.time(),
            "mode": mode,
            "checkpoint": str(checkpoint_path),
            "dtype": str(dtype).replace("torch.", ""),
            "device": str(device),
            "wrapped_count": len(wrapped_names),
            "wrapped_modules": wrapped_names,
        }
    )
    print(
        f"[pi05_transcoder] installed mode={mode} checkpoint={checkpoint_path} "
        f"wrapped={len(wrapped_names)} dtype={dtype} device={device}",
        file=sys.stderr,
    )


def _patch_make_policy(module: Any, attr: str) -> None:
    original = getattr(module, attr)
    if getattr(original, "_pi05_transcoder_patched", False):
        return

    def _patched_make_policy(*args: Any, **kwargs: Any):
        policy = original(*args, **kwargs)
        _install_on_policy(policy)
        return policy

    _patched_make_policy._pi05_transcoder_patched = True
    setattr(module, attr, _patched_make_policy)


def atlas_begin_episode(*, suite: str = "", task_id: int | None = None, episode: int | None = None) -> None:
    _RECORDER.begin_episode(suite=suite, task_id=task_id, episode=episode)


def atlas_end_episode() -> None:
    _RECORDER.end_episode()


if _enabled():
    try:
        import lerobot.policies as _policies
        import lerobot.policies.factory as _policy_factory

        _patch_make_policy(_policies, "make_policy")
        _patch_make_policy(_policy_factory, "make_policy")
    except Exception as exc:  # pragma: no cover - best effort patching
        print(f"[pi05_transcoder] failed to patch make_policy: {exc}", file=sys.stderr)

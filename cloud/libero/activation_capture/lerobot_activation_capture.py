"""Optional LeRobot PI0.5 activation capture.

Loaded automatically when this directory is on PYTHONPATH. The runner enables it
only when CAPTURE_ACTIVATIONS=1.
"""

from __future__ import annotations

import atexit
import json
import math
import os
import re
import sys
import time
import types
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _enabled() -> bool:
    value = os.environ.get("LEROBOT_CAPTURE_ACTIVATIONS") or os.environ.get("CAPTURE_ACTIVATIONS")
    return str(value).lower() in {"1", "true", "yes", "on"}


try:
    import numpy as np
    import torch
except Exception as exc:  # pragma: no cover - best effort startup guard
    print(f"[activation_capture] disabled: {exc}", file=sys.stderr)
    raise RuntimeError("activation capture dependencies are unavailable") from exc


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).lower() in {"1", "true", "yes", "on"}


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "value"


def _json_number(value: Any, digits: int = 6) -> float | int | None:
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if not math.isfinite(value):
            return None
        return round(value, digits)
    if isinstance(value, (np.bool_, bool)):
        return int(value)
    return None


def _numeric_list(values: Any, max_items: int, digits: int = 6) -> list[float | int | None]:
    arr = np.asarray(values)
    if arr.size == 0 or max_items <= 0:
        return []
    flat = arr.reshape(-1)[:max_items]
    return [_json_number(value, digits=digits) for value in flat]


def _as_numpy(value: Any) -> np.ndarray | None:
    try:
        if torch.is_tensor(value):
            return value.detach().to("cpu").numpy()
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, (list, tuple)):
            return np.asarray(value)
    except Exception:
        return None
    return None


def _summarize_numeric(value: Any, max_values: int = 32) -> dict[str, Any] | None:
    arr = _as_numpy(value)
    if arr is None:
        return None
    payload: dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }
    if arr.size == 0:
        payload["values"] = []
        return payload
    if not (np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_)):
        return payload

    numeric = arr.astype(np.float32, copy=False).reshape(-1)
    finite = numeric[np.isfinite(numeric)]
    if finite.size:
        payload.update(
            {
                "mean": _json_number(float(finite.mean())),
                "std": _json_number(float(finite.std())),
                "min": _json_number(float(finite.min())),
                "max": _json_number(float(finite.max())),
                "abs_mean": _json_number(float(np.abs(finite).mean())),
            }
        )
    payload["values"] = _numeric_list(arr, max_values)
    if arr.size > max_values:
        payload["values_truncated"] = int(arr.size - max_values)
    return payload


def _shape_summary(value: Any) -> dict[str, Any] | None:
    arr = _as_numpy(value)
    if arr is None:
        return None
    return {"shape": list(arr.shape), "dtype": str(arr.dtype)}


def _first_batch_values(value: Any, max_items: int) -> list[float | int | None]:
    arr = _as_numpy(value)
    if arr is None or arr.size == 0:
        return []
    if arr.ndim >= 2:
        arr = arr[0]
    return _numeric_list(arr, max_items)


def _first_batch_matrix(value: Any, max_rows: int = 1, max_cols: int = 12) -> list[list[float | int | None]]:
    arr = _as_numpy(value)
    if arr is None or arr.size == 0:
        return []
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    rows = []
    for row in arr[:max_rows]:
        rows.append(_numeric_list(row[:max_cols], max_cols))
    return rows


def _extract_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _extract_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _extract_tensor(item)
            if tensor is not None:
                return tensor
    for attr in ("last_hidden_state", "pooler_output"):
        item = getattr(value, attr, None)
        if torch.is_tensor(item):
            return item
    return None


def _downsample(values: torch.Tensor, bins: int) -> list[float]:
    flat = values.detach().float().flatten()
    if flat.numel() == 0:
        return []
    if flat.numel() <= bins:
        return [round(float(x), 6) for x in flat.cpu().tolist()]
    chunks = torch.chunk(flat, bins)
    return [round(float(chunk.mean().cpu()), 6) for chunk in chunks]


def _task_from_batch(batch: dict[str, Any]) -> str | None:
    task = batch.get("task")
    if isinstance(task, str):
        return task
    if isinstance(task, (list, tuple)) and task and isinstance(task[0], str):
        return task[0]
    return None


ACTION_DIM_LABELS = ("dx", "dy", "dz", "dRx", "dRy", "dRz", "grip")
OBS_STATE = "observation.state"
OBS_LANGUAGE_TOKENS = "observation.language.tokens"
OBS_LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"


_TOKENIZER: Any | None = None


def _decode_tokens(token_ids: list[int]) -> str | None:
    global _TOKENIZER
    if not token_ids or not _bool_env("CAPTURE_DECODE_LANGUAGE", False):
        return None
    try:
        if _TOKENIZER is None:
            from transformers import AutoTokenizer

            _TOKENIZER = AutoTokenizer.from_pretrained(
                os.environ.get("CAPTURE_TOKENIZER_NAME", "google/paligemma-3b-pt-224"),
                local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1",
            )
        return _TOKENIZER.decode(token_ids, skip_special_tokens=False)
    except Exception as exc:
        return f"<decode failed: {type(exc).__name__}: {exc}>"


def _language_payload(batch: dict[str, Any], *, include_token_ids: bool, max_token_ids: int) -> dict[str, Any] | None:
    tokens = batch.get(OBS_LANGUAGE_TOKENS)
    if tokens is None:
        return None
    token_arr = _as_numpy(tokens)
    if token_arr is None or token_arr.size == 0:
        return None
    if token_arr.ndim == 1:
        first_tokens = token_arr
    else:
        first_tokens = token_arr[0]

    mask_arr = _as_numpy(batch.get(OBS_LANGUAGE_ATTENTION_MASK))
    if mask_arr is not None and mask_arr.size:
        first_mask = mask_arr if mask_arr.ndim == 1 else mask_arr[0]
        active = first_tokens[np.asarray(first_mask, dtype=bool)]
    else:
        active = first_tokens

    active_ids = [int(x) for x in active.reshape(-1).tolist()]
    payload: dict[str, Any] = {
        "tokens_shape": list(token_arr.shape),
        "active_token_count": len(active_ids),
        "attention_mask_shape": list(mask_arr.shape) if mask_arr is not None else None,
    }
    if include_token_ids:
        payload["active_token_ids"] = active_ids[:max_token_ids]
        if len(active_ids) > max_token_ids:
            payload["active_token_ids_truncated"] = len(active_ids) - max_token_ids
    decoded = _decode_tokens(active_ids)
    if decoded is not None:
        payload["decoded"] = decoded
    return payload


def _format_action_feedback(values: np.ndarray) -> str:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    parts = []
    for label, value in zip(ACTION_DIM_LABELS, flat, strict=False):
        parts.append(f"{label}={float(value):+.3f}")
    return ", ".join(parts)


def _inject_visible_feedback(prompt: str, feedback: str) -> str:
    if not feedback:
        return prompt
    marker = "\nAction: "
    insertion = f" Feedback: {feedback};"
    if marker in prompt:
        return prompt.replace(marker, f"{insertion}{marker}", 1)
    return f"{prompt.rstrip()}{insertion}\nAction: "


class ActivationRecorder:
    def __init__(self) -> None:
        capture_dir = os.environ.get("LEROBOT_CAPTURE_DIR", "activation_capture")
        self.root = Path(capture_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.root / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.root / "events.jsonl"
        self.file = self.events_path.open("a", buffering=1)
        self.layer_stride = max(1, _int_env("CAPTURE_LAYER_STRIDE", 1))
        self.max_bins = max(4, _int_env("CAPTURE_MAX_BINS", 64))
        self.max_chunks = _int_env("CAPTURE_MAX_CHUNKS", 80)
        self.every_n_chunks = max(1, _int_env("CAPTURE_EVERY_N_CHUNKS", 1))
        self.max_images = _int_env("CAPTURE_MAX_IMAGES", 80)
        self.sample_rate = max(0.0, min(1.0, _float_env("CAPTURE_SAMPLE_RATE", 1.0)))
        self.capture_param_stats = _int_env("CAPTURE_PARAM_STATS", 0) == 1
        self.capture_feedback_trace = _bool_env("CAPTURE_FEEDBACK_TRACE", True)
        self.capture_env_steps = _bool_env("CAPTURE_ENV_STEPS", True)
        self.capture_env_step_images = _bool_env("CAPTURE_ENV_STEP_IMAGES", False)
        self.env_step_image_every_n = max(1, _int_env("CAPTURE_ENV_STEP_IMAGE_EVERY_N", 10))
        self.max_env_step_images = _int_env("CAPTURE_MAX_ENV_STEP_IMAGES", 80)
        self.capture_token_ids = _bool_env("CAPTURE_TOKEN_IDS", True)
        self.capture_batch_tensor_summary = _bool_env("CAPTURE_BATCH_TENSOR_SUMMARY", True)
        self.capture_denoise_trace = _bool_env("CAPTURE_DENOISE_TRACE", True)
        self.max_tensor_values = max(0, _int_env("CAPTURE_MAX_TENSOR_VALUES", 64))
        self.prompt_feedback_mode = os.environ.get("PI05_PROMPT_FEEDBACK_MODE", "off").strip().lower()
        families = os.environ.get("CAPTURE_FAMILIES", "vision,prefix,expert,projection")
        self.families = {item.strip() for item in families.split(",") if item.strip()}
        self.chunk_index = -1
        self.policy_step_index = -1
        self.env_step_index = -1
        self.reset_index = -1
        self.captured_chunks = 0
        self.active = False
        self.hook_call = 0
        self.image_count = 0
        self.env_step_image_count = 0
        self.hooks = []
        self.policy_chunk_size: int | None = None
        self.policy_n_action_steps: int | None = None
        self.policy_action_dim: int | None = None
        self.last_selected_action: dict[str, Any] | None = None
        self.last_applied_action_values: np.ndarray | None = None
        self.recent_applied_actions: deque[np.ndarray] = deque(maxlen=50)
        self._write(
            {
                "type": "capture_start",
                "time": time.time(),
                "families": sorted(self.families),
                "layer_stride": self.layer_stride,
                "max_bins": self.max_bins,
                "max_chunks": self.max_chunks,
                "every_n_chunks": self.every_n_chunks,
                "sample_rate": self.sample_rate,
                "capture_param_stats": self.capture_param_stats,
                "capture_feedback_trace": self.capture_feedback_trace,
                "capture_env_steps": self.capture_env_steps,
                "capture_env_step_images": self.capture_env_step_images,
                "capture_token_ids": self.capture_token_ids,
                "capture_batch_tensor_summary": self.capture_batch_tensor_summary,
                "capture_denoise_trace": self.capture_denoise_trace,
                "prompt_feedback_mode": self.prompt_feedback_mode,
            }
        )
        atexit.register(self.close)

    def close(self) -> None:
        for handle in self.hooks:
            try:
                handle.remove()
            except Exception:
                pass
        self.hooks.clear()
        if not self.file.closed:
            self._write({"type": "capture_end", "time": time.time()})
            self.file.close()

    def _write(self, payload: dict[str, Any]) -> None:
        self.file.write(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")

    def configure_policy(self, policy: Any) -> None:
        config = getattr(policy, "config", None)
        self.policy_chunk_size = getattr(config, "chunk_size", None)
        self.policy_n_action_steps = getattr(config, "n_action_steps", None)
        output_features = getattr(config, "output_features", {}) or {}
        action_feature = output_features.get("action") if isinstance(output_features, dict) else None
        action_shape = getattr(action_feature, "shape", None)
        if action_shape:
            self.policy_action_dim = int(action_shape[0])
        self._write(
            {
                "type": "policy_config",
                "time": time.time(),
                "policy": type(policy).__name__,
                "chunk_size": self.policy_chunk_size,
                "n_action_steps": self.policy_n_action_steps,
                "action_dim": self.policy_action_dim,
                "use_relative_actions": getattr(config, "use_relative_actions", None),
                "num_inference_steps": getattr(config, "num_inference_steps", None),
            }
        )

    def _should_capture_chunk(self) -> bool:
        if self.chunk_index % self.every_n_chunks != 0:
            return False
        if self.max_chunks > 0 and self.captured_chunks >= self.max_chunks:
            return False
        if self.sample_rate < 1.0 and np.random.random() > self.sample_rate:
            return False
        return True

    def begin_chunk(self, batch: dict[str, Any]) -> None:
        self.chunk_index += 1
        self.hook_call = 0
        self.active = self._should_capture_chunk()
        if not self.active:
            return
        self.captured_chunks += 1
        payload: dict[str, Any] = {
            "type": "chunk_start",
            "chunk": self.chunk_index,
            "policy_step": self.policy_step_index,
            "time": time.time(),
            "batch_keys": sorted(str(key) for key in batch.keys()),
        }
        task = _task_from_batch(batch)
        if task is not None:
            payload["task"] = task
        if self.capture_feedback_trace:
            state = batch.get(OBS_STATE)
            state_summary = _summarize_numeric(state, self.max_tensor_values)
            if state_summary is not None:
                payload["state"] = state_summary

            language = _language_payload(
                batch,
                include_token_ids=self.capture_token_ids,
                max_token_ids=self.max_tensor_values,
            )
            if language is not None:
                payload["language"] = language

            if self.capture_batch_tensor_summary:
                tensor_summaries = {}
                for key, value in batch.items():
                    if str(key).startswith("observation.images."):
                        continue
                    if str(key) in {"task"}:
                        continue
                    if torch.is_tensor(value):
                        summary = _summarize_numeric(value, self.max_tensor_values)
                        if summary is not None:
                            tensor_summaries[str(key)] = summary
                if tensor_summaries:
                    payload["tensor_summaries"] = tensor_summaries
        self._write(payload)
        self._capture_images(batch)

    def end_chunk(self) -> None:
        if self.active:
            self._write({"type": "chunk_end", "chunk": self.chunk_index, "time": time.time()})
        self.active = False

    def _capture_images(self, batch: dict[str, Any]) -> None:
        if self.image_count >= self.max_images:
            return
        for key, value in batch.items():
            if not str(key).startswith("observation.images."):
                continue
            if not torch.is_tensor(value):
                continue
            try:
                arr = value.detach()[0].to("cpu").float().numpy()
                if arr.ndim != 3:
                    continue
                if arr.shape[0] in (1, 3):
                    arr = np.moveaxis(arr, 0, -1)
                if arr.shape[-1] == 1:
                    arr = np.repeat(arr, 3, axis=-1)
                if arr.min() < 0:
                    arr = (arr + 1.0) / 2.0
                if arr.max() > 2:
                    arr = arr / 255.0
                arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
                try:
                    import cv2

                    path = self.images_dir / f"chunk_{self.chunk_index:06d}_{_safe_name(str(key))}.jpg"
                    cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
                except Exception:
                    path = self.images_dir / f"chunk_{self.chunk_index:06d}_{_safe_name(str(key))}.npy"
                    np.save(path, arr)
                self.image_count += 1
                self._write(
                    {
                        "type": "image",
                        "chunk": self.chunk_index,
                        "key": str(key),
                        "path": str(path.relative_to(self.root)),
                        "shape": list(arr.shape),
                    }
                )
            except Exception as exc:
                self._write({"type": "image_error", "chunk": self.chunk_index, "key": str(key), "error": str(exc)})

    def record_action_chunk(self, actions: torch.Tensor) -> None:
        if not self.active:
            return
        try:
            arr = actions.detach()[0].to("cpu").float()
            values = arr[:, : min(arr.shape[-1], 12)]
            self._write(
                {
                    "type": "action_chunk",
                    "chunk": self.chunk_index,
                    "policy_step": self.policy_step_index,
                    "space": "policy_normalized_pre_postprocessor",
                    "chunk_size": self.policy_chunk_size,
                    "n_action_steps": self.policy_n_action_steps,
                    "shape": list(actions.shape),
                    "values": [[round(float(x), 6) for x in row] for row in values.tolist()],
                    "norm_per_step": [round(float(x), 6) for x in arr.norm(dim=-1).tolist()],
                    "abs_per_dim": [round(float(x), 6) for x in arr.abs().mean(dim=0).tolist()],
                }
            )
        except Exception as exc:
            self._write({"type": "action_error", "chunk": self.chunk_index, "error": str(exc)})

    def record_denoise_start(self, noise: torch.Tensor, num_steps: int) -> None:
        if not self.active or not self.capture_denoise_trace:
            return
        self._write(
            {
                "type": "denoise_start",
                "time": time.time(),
                "chunk": self.chunk_index,
                "policy_step": self.policy_step_index,
                "num_steps": int(num_steps),
                "noise": _summarize_numeric(noise, self.max_tensor_values),
            }
        )

    def record_denoise_step(self, step: int, timestep: float, x_t: torch.Tensor, v_t: torch.Tensor, next_x_t: torch.Tensor) -> None:
        if not self.active or not self.capture_denoise_trace:
            return
        try:
            x = x_t.detach().float()
            v = v_t.detach().float()
            next_x = next_x_t.detach().float()
            self._write(
                {
                    "type": "denoise_step",
                    "time": time.time(),
                    "chunk": self.chunk_index,
                    "policy_step": self.policy_step_index,
                    "step": int(step),
                    "timestep": round(float(timestep), 6),
                    "x_abs_mean": round(float(x.abs().mean().cpu()), 6),
                    "v_abs_mean": round(float(v.abs().mean().cpu()), 6),
                    "update_abs_mean": round(float((next_x - x).abs().mean().cpu()), 6),
                    "next_abs_mean": round(float(next_x.abs().mean().cpu()), 6),
                    "x_first_action": _first_batch_values(x[:, :1, : self.policy_action_dim or 7], self.max_tensor_values),
                    "v_first_action": _first_batch_values(v[:, :1, : self.policy_action_dim or 7], self.max_tensor_values),
                    "next_first_action": _first_batch_values(next_x[:, :1, : self.policy_action_dim or 7], self.max_tensor_values),
                }
            )
        except Exception as exc:
            self._write(
                {
                    "type": "denoise_trace_error",
                    "time": time.time(),
                    "chunk": self.chunk_index,
                    "step": int(step),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    def record_denoise_end(self, actions: torch.Tensor) -> None:
        if not self.active or not self.capture_denoise_trace:
            return
        self._write(
            {
                "type": "denoise_end",
                "time": time.time(),
                "chunk": self.chunk_index,
                "policy_step": self.policy_step_index,
                "actions": _summarize_numeric(actions, self.max_tensor_values),
                "first_action": _first_batch_values(actions[:, :1, : self.policy_action_dim or 7], self.max_tensor_values),
            }
        )

    def begin_select_action(self, policy: Any, batch: dict[str, Any]) -> dict[str, Any]:
        queue = getattr(policy, "_action_queue", None)
        queue_len_before = len(queue) if queue is not None else None
        n_action_steps = int(getattr(getattr(policy, "config", None), "n_action_steps", 1) or 1)
        self.policy_step_index += 1
        context = {
            "policy_step": self.policy_step_index,
            "queue_len_before": queue_len_before,
            "n_action_steps": n_action_steps,
            "will_predict_chunk": queue_len_before == 0,
        }
        if self.capture_feedback_trace:
            self._write(
                {
                    "type": "policy_step_start",
                    "time": time.time(),
                    "policy_step": self.policy_step_index,
                    "chunk": self.chunk_index + 1 if queue_len_before == 0 else self.chunk_index,
                    "queue_len_before": queue_len_before,
                    "will_predict_chunk": queue_len_before == 0,
                    "n_action_steps": n_action_steps,
                    "batch_keys": sorted(str(key) for key in batch.keys()),
                }
            )
        return context

    def record_selected_action(self, policy: Any, action: torch.Tensor, context: dict[str, Any]) -> None:
        queue = getattr(policy, "_action_queue", None)
        queue_len_after = len(queue) if queue is not None else None
        n_action_steps = int(context.get("n_action_steps") or self.policy_n_action_steps or 1)
        queue_len_before = context.get("queue_len_before")
        if queue_len_before == 0:
            phase = 0
        elif isinstance(queue_len_before, int):
            phase = max(0, n_action_steps - queue_len_before)
        else:
            phase = None
        selected = {
            "type": "policy_selected_action",
            "time": time.time(),
            "policy_step": context.get("policy_step"),
            "chunk": self.chunk_index,
            "phase": phase,
            "queue_len_before": queue_len_before,
            "queue_len_after": queue_len_after,
            "space": "policy_normalized_pre_postprocessor",
            "action": _summarize_numeric(action, self.max_tensor_values),
            "values": _first_batch_values(action, self.max_tensor_values),
        }
        self.last_selected_action = selected
        if self.capture_feedback_trace:
            self._write(selected)

    def record_select_error(self, context: dict[str, Any], exc: Exception) -> None:
        self._write(
            {
                "type": "policy_step_error",
                "time": time.time(),
                "policy_step": context.get("policy_step"),
                "chunk": self.chunk_index,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )

    def record_env_reset(
        self,
        observation: Any,
        info: Any,
        task_group: str | None = None,
        task_id: int | None = None,
    ) -> None:
        if not self.capture_env_steps:
            return
        self.reset_index += 1
        self.env_step_index = -1
        self.last_selected_action = None
        self.last_applied_action_values = None
        self.recent_applied_actions.clear()
        self._write(
            {
                "type": "env_reset",
                "time": time.time(),
                "reset": self.reset_index,
                "task_group": task_group,
                "task_id": task_id,
                "observation": self._observation_summary(observation),
                "info_keys": sorted(str(key) for key in info.keys()) if isinstance(info, dict) else [],
            }
        )

    def record_env_step(
        self,
        action: Any,
        observation: Any,
        reward: Any,
        terminated: Any,
        truncated: Any,
        info: Any,
        task_group: str | None = None,
        task_id: int | None = None,
    ) -> None:
        if not self.capture_env_steps:
            return
        self.env_step_index += 1
        action_arr = _as_numpy(action)
        if action_arr is not None and action_arr.size:
            first_action = action_arr[0] if action_arr.ndim >= 2 else action_arr
            self.last_applied_action_values = np.asarray(first_action, dtype=np.float32).reshape(-1)
            self.recent_applied_actions.append(self.last_applied_action_values)
        selected = self.last_selected_action or {}
        payload = {
            "type": "env_step",
            "time": time.time(),
            "reset": self.reset_index,
            "task_group": task_group,
            "task_id": task_id,
            "env_step": self.env_step_index,
            "policy_step": selected.get("policy_step"),
            "chunk": selected.get("chunk", self.chunk_index),
            "phase": selected.get("phase"),
            "applied_action": {
                "space": "env_postprocessed",
                "summary": _summarize_numeric(action, self.max_tensor_values),
                "values": _first_batch_values(action, self.max_tensor_values),
            },
            "reward": _first_batch_values(reward, self.max_tensor_values),
            "terminated": _first_batch_values(terminated, self.max_tensor_values),
            "truncated": _first_batch_values(truncated, self.max_tensor_values),
            "success": self._extract_success(info),
            "observation": self._observation_summary(observation),
        }
        self._capture_env_step_images(observation, payload)
        self._write(payload)

    def prompt_feedback_text(self) -> str:
        mode = self.prompt_feedback_mode
        if mode in {"", "off", "false", "0", "none"}:
            return ""
        if self.last_applied_action_values is None:
            return ""
        if mode in {"last_action", "action", "visible_last_action"}:
            return f"last applied action {_format_action_feedback(self.last_applied_action_values)}"
        if mode in {"chunk_summary", "recent_actions", "window"}:
            window = max(1, int(self.policy_n_action_steps or 10))
            recent = list(self.recent_applied_actions)[-window:]
            if not recent:
                return ""
            arr = np.stack(recent, axis=0)
            mean = arr.mean(axis=0)
            return f"recent {len(recent)} applied action mean {_format_action_feedback(mean)}"
        return f"last applied action {_format_action_feedback(self.last_applied_action_values)}"

    def record_prompt_feedback(self, feedback: str, prompts: list[str]) -> None:
        if not feedback or not self.capture_feedback_trace:
            return
        queue_len_after = None
        if isinstance(self.last_selected_action, dict):
            queue_len_after = self.last_selected_action.get("queue_len_after")
        self._write(
            {
                "type": "prompt_feedback",
                "time": time.time(),
                "policy_step": self.policy_step_index + 1,
                "env_step": self.env_step_index,
                "chunk": self.chunk_index + 1,
                "mode": self.prompt_feedback_mode,
                "will_affect_next_chunk": queue_len_after == 0,
                "feedback": feedback,
                "prompt_preview": prompts[0][:500] if prompts else "",
            }
        )

    def _observation_summary(self, observation: Any) -> dict[str, Any]:
        if not isinstance(observation, dict):
            return {"type": type(observation).__name__}
        result: dict[str, Any] = {"keys": sorted(str(key) for key in observation.keys())}
        summaries: dict[str, Any] = {}
        for key, value in observation.items():
            if key == "pixels" and isinstance(value, dict):
                summaries[str(key)] = {
                    str(cam): _shape_summary(img) for cam, img in value.items()
                }
                continue
            if isinstance(value, dict):
                nested = {}
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, dict):
                        nested[str(nested_key)] = {
                            str(inner_key): _summarize_numeric(inner_value, self.max_tensor_values)
                            for inner_key, inner_value in nested_value.items()
                        }
                    else:
                        nested[str(nested_key)] = _summarize_numeric(nested_value, self.max_tensor_values)
                summaries[str(key)] = nested
                continue
            summary = _summarize_numeric(value, self.max_tensor_values)
            if summary is not None:
                summaries[str(key)] = summary
        if summaries:
            result["summaries"] = summaries
        return result

    def _extract_success(self, info: Any) -> list[float | int | None]:
        if not isinstance(info, dict):
            return []
        if "is_success" in info:
            return _first_batch_values(info["is_success"], self.max_tensor_values)
        if "final_info" not in info:
            return []
        final_info = info["final_info"]
        if isinstance(final_info, dict) and "is_success" in final_info:
            return _first_batch_values(final_info["is_success"], self.max_tensor_values)
        if isinstance(final_info, (list, tuple)):
            values = []
            for item in final_info:
                if isinstance(item, dict) and "is_success" in item:
                    values.append(bool(item["is_success"]))
            return _numeric_list(values, self.max_tensor_values)
        return []

    def _capture_env_step_images(self, observation: Any, payload: dict[str, Any]) -> None:
        if not self.capture_env_step_images:
            return
        if self.env_step_image_count >= self.max_env_step_images:
            return
        if self.env_step_index % self.env_step_image_every_n != 0:
            return
        if not isinstance(observation, dict):
            return
        pixels = observation.get("pixels")
        if not isinstance(pixels, dict):
            return
        image_events = []
        for key, value in pixels.items():
            if self.env_step_image_count >= self.max_env_step_images:
                break
            arr = _as_numpy(value)
            if arr is None or arr.size == 0:
                continue
            if arr.ndim == 4:
                arr = arr[0]
            if arr.ndim != 3:
                continue
            try:
                if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
                    arr = np.moveaxis(arr, 0, -1)
                if arr.shape[-1] == 1:
                    arr = np.repeat(arr, 3, axis=-1)
                if arr.max() <= 1.5:
                    arr = np.clip(arr, 0, 1) * 255
                arr = np.clip(arr, 0, 255).astype(np.uint8)
                import cv2

                path = self.images_dir / (
                    f"env_step_{self.env_step_index:06d}_{_safe_name(str(key))}.jpg"
                )
                cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
                image_events.append(
                    {
                        "key": str(key),
                        "path": str(path.relative_to(self.root)),
                        "shape": list(arr.shape),
                    }
                )
                self.env_step_image_count += 1
            except Exception as exc:
                image_events.append({"key": str(key), "error": str(exc)})
        if image_events:
            payload["env_step_images"] = image_events

    def record_param_stats(self, name: str, family: str, layer_index: int | None, module: Any) -> None:
        if not self.capture_param_stats:
            return
        try:
            total = 0
            abs_sum = 0.0
            sq_sum = 0.0
            max_abs = 0.0
            with torch.no_grad():
                for param in module.parameters(recurse=True):
                    x = param.detach().float()
                    total += x.numel()
                    abs_sum += float(x.abs().sum().cpu())
                    sq_sum += float((x * x).sum().cpu())
                    max_abs = max(max_abs, float(x.abs().max().cpu()))
            if total == 0:
                return
            self._write(
                {
                    "type": "param_stats",
                    "family": family,
                    "layer": layer_index,
                    "module": name,
                    "numel": total,
                    "abs_mean": round(abs_sum / total, 8),
                    "rms": round((sq_sum / total) ** 0.5, 8),
                    "max_abs": round(max_abs, 8),
                }
            )
        except Exception as exc:
            self._write({"type": "param_stats_error", "module": name, "error": str(exc)})

    def hook(self, name: str, family: str, layer_index: int | None):
        def _hook(_module: Any, _inputs: Any, output: Any) -> None:
            if not self.active:
                return
            if layer_index is not None and layer_index % self.layer_stride != 0:
                return
            tensor = _extract_tensor(output)
            if tensor is None or tensor.numel() == 0:
                return
            try:
                x = tensor.detach().float()
                token_abs = None
                feature_abs = None
                if x.ndim >= 3:
                    dims = tuple(i for i in range(x.ndim) if i != 1)
                    token_abs = _downsample(x.abs().mean(dim=dims), self.max_bins)
                elif x.ndim == 2:
                    feature_abs = _downsample(x.abs().mean(dim=0), self.max_bins)
                self.hook_call += 1
                self._write(
                    {
                        "type": "activation",
                        "chunk": self.chunk_index,
                        "call": self.hook_call,
                        "family": family,
                        "layer": layer_index,
                        "module": name,
                        "shape": list(tensor.shape),
                        "mean": round(float(x.mean().cpu()), 6),
                        "std": round(float(x.std().cpu()), 6) if x.numel() > 1 else 0.0,
                        "abs_mean": round(float(x.abs().mean().cpu()), 6),
                        "max_abs": round(float(x.abs().max().cpu()), 6),
                        "token_abs": token_abs,
                        "feature_abs": feature_abs,
                    }
                )
            except Exception as exc:
                self._write(
                    {
                        "type": "activation_error",
                        "chunk": self.chunk_index,
                        "module": name,
                        "error": str(exc),
                    }
                )

        return _hook


_RECORDER = ActivationRecorder()


def _classify_module(name: str) -> tuple[str, int | None] | None:
    patterns = [
        (r"(?:^|\.)paligemma_with_expert\.paligemma\.model\.vision_tower\.vision_model\.encoder\.layers\.(\d+)$", "vision"),
        (r"(?:^|\.)paligemma_with_expert\.paligemma\.model\.language_model\.layers\.(\d+)$", "prefix"),
        (r"(?:^|\.)paligemma_with_expert\.gemma_expert\.model\.layers\.(\d+)$", "expert"),
    ]
    for pattern, family in patterns:
        match = re.search(pattern, name)
        if match:
            return family, int(match.group(1))
    projection_names = ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out")
    if any(name == item or name.endswith(f".{item}") for item in projection_names):
        return "projection", None
    return None


def _install_on_policy(policy: Any) -> None:
    if getattr(policy, "_activation_capture_installed", False):
        return
    _RECORDER.configure_policy(policy)
    selected = []
    selected_modules = []
    for name, module in policy.named_modules():
        classified = _classify_module(name)
        if classified is None:
            continue
        family, layer_index = classified
        if family not in _RECORDER.families:
            continue
        handle = module.register_forward_hook(_RECORDER.hook(name, family, layer_index))
        _RECORDER.hooks.append(handle)
        selected.append({"name": name, "family": family, "layer": layer_index})
        selected_modules.append((name, family, layer_index, module))

    for name, family, layer_index, module in selected_modules:
        _RECORDER.record_param_stats(name, family, layer_index, module)

    original_predict = policy.predict_action_chunk

    def wrapped_predict_action_chunk(self: Any, batch: dict[str, Any], *args: Any, **kwargs: Any):
        _RECORDER.begin_chunk(batch)
        try:
            actions = original_predict(batch, *args, **kwargs)
            _RECORDER.record_action_chunk(actions)
            return actions
        finally:
            _RECORDER.end_chunk()

    original_select = policy.select_action
    model = getattr(policy, "model", None)
    if model is not None and not getattr(model, "_activation_capture_sample_installed", False):
        original_sample_actions = model.sample_actions
        original_denoise_step = model.denoise_step

        def wrapped_sample_actions(
            self_model: Any,
            images: Any,
            img_masks: Any,
            tokens: torch.Tensor,
            masks: torch.Tensor,
            noise: torch.Tensor | None = None,
            num_steps: int | None = None,
            **kwargs: Any,
        ):
            actual_num_steps = int(num_steps or getattr(self_model.config, "num_inference_steps", 1))
            if noise is None:
                actions_shape = (
                    tokens.shape[0],
                    int(getattr(self_model.config, "chunk_size", 50)),
                    int(getattr(self_model.config, "max_action_dim", 32)),
                )
                noise = self_model.sample_noise(actions_shape, tokens.device)
            self_model._activation_capture_denoise_step_index = 0
            self_model._activation_capture_denoise_num_steps = actual_num_steps
            _RECORDER.record_denoise_start(noise, actual_num_steps)
            actions = original_sample_actions(
                images,
                img_masks,
                tokens,
                masks,
                noise=noise,
                num_steps=actual_num_steps,
                **kwargs,
            )
            _RECORDER.record_denoise_end(actions)
            return actions

        def wrapped_denoise_step(
            self_model: Any,
            prefix_pad_masks: torch.Tensor,
            past_key_values: Any,
            x_t: torch.Tensor,
            timestep: torch.Tensor,
        ):
            v_t = original_denoise_step(prefix_pad_masks, past_key_values, x_t, timestep)
            step = int(getattr(self_model, "_activation_capture_denoise_step_index", 0))
            num_steps = int(getattr(self_model, "_activation_capture_denoise_num_steps", 1) or 1)
            dt = -1.0 / num_steps
            next_x_t = x_t + dt * v_t
            time_value = float(timestep[0].detach().to("cpu")) if torch.is_tensor(timestep) else float(timestep)
            _RECORDER.record_denoise_step(step, time_value, x_t, v_t, next_x_t)
            self_model._activation_capture_denoise_step_index = step + 1
            return v_t

        model.sample_actions = types.MethodType(wrapped_sample_actions, model)
        model.denoise_step = types.MethodType(wrapped_denoise_step, model)
        model._activation_capture_sample_installed = True

    def wrapped_select_action(self: Any, batch: dict[str, Any], *args: Any, **kwargs: Any):
        context = _RECORDER.begin_select_action(self, batch)
        try:
            action = original_select(batch, *args, **kwargs)
            _RECORDER.record_selected_action(self, action, context)
            return action
        except Exception as exc:
            _RECORDER.record_select_error(context, exc)
            raise

    policy.predict_action_chunk = types.MethodType(wrapped_predict_action_chunk, policy)
    policy.select_action = types.MethodType(wrapped_select_action, policy)
    policy._activation_capture_installed = True
    _RECORDER._write({"type": "hooks_installed", "count": len(selected), "modules": selected})
    print(f"[activation_capture] installed {len(selected)} hooks", file=sys.stderr)


def _patch_make_policy(module: Any, attr: str) -> None:
    original = getattr(module, attr)
    if getattr(original, "_activation_capture_patched", False):
        return

    def _patched_make_policy(*args: Any, **kwargs: Any):
        policy = original(*args, **kwargs)
        _install_on_policy(policy)
        return policy

    _patched_make_policy._activation_capture_patched = True
    setattr(module, attr, _patched_make_policy)


class _CaptureVectorEnv:
    def __init__(self, env: Any, task_group: str | None = None, task_id: int | None = None) -> None:
        self.env = env
        self.task_group = task_group
        self.task_id = task_id
        self._activation_capture_env_wrapped = True

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    @property
    def unwrapped(self) -> Any:
        return getattr(self.env, "unwrapped", self.env)

    def reset(self, *args: Any, **kwargs: Any):
        observation, info = self.env.reset(*args, **kwargs)
        _RECORDER.record_env_reset(observation, info, self.task_group, self.task_id)
        return observation, info

    def step(self, action: Any):
        observation, reward, terminated, truncated, info = self.env.step(action)
        _RECORDER.record_env_step(
            action,
            observation,
            reward,
            terminated,
            truncated,
            info,
            self.task_group,
            self.task_id,
        )
        return observation, reward, terminated, truncated, info

    def call(self, *args: Any, **kwargs: Any):
        return self.env.call(*args, **kwargs)

    def get_attr(self, *args: Any, **kwargs: Any):
        return self.env.get_attr(*args, **kwargs)

    def close(self) -> None:
        return self.env.close()


def _wrap_envs(value: Any, task_group: str | None = None, task_id: int | None = None) -> Any:
    if isinstance(value, Mapping):
        wrapped = {}
        for key, item in value.items():
            if isinstance(item, Mapping):
                wrapped[key] = _wrap_envs(item, task_group=str(key), task_id=task_id)
            else:
                next_task_id = int(key) if isinstance(key, int) else task_id
                wrapped[key] = _wrap_envs(item, task_group=task_group, task_id=next_task_id)
        return wrapped
    if getattr(value, "_activation_capture_env_wrapped", False):
        return value
    if hasattr(value, "step") and hasattr(value, "reset"):
        return _CaptureVectorEnv(value, task_group=task_group, task_id=task_id)
    return value


def _patch_make_env(module: Any, attr: str) -> None:
    original = getattr(module, attr)
    if getattr(original, "_activation_capture_patched", False):
        return

    def _patched_make_env(*args: Any, **kwargs: Any):
        envs = original(*args, **kwargs)
        return _wrap_envs(envs)

    _patched_make_env._activation_capture_patched = True
    setattr(module, attr, _patched_make_env)


def _patch_pi05_prompt_feedback() -> None:
    try:
        import lerobot.policies.pi05.processor_pi05 as _pi05_processor
    except Exception as exc:
        print(f"[activation_capture] failed to import pi05 prompt processor: {exc}", file=sys.stderr)
        return

    step_cls = getattr(_pi05_processor, "Pi05PrepareStateTokenizerProcessorStep", None)
    if step_cls is None:
        return
    original = step_cls.__call__
    if getattr(original, "_activation_capture_patched", False):
        return

    def _patched_prepare_state_tokenizer(self: Any, transition: Any):
        result = original(self, transition)
        feedback = _RECORDER.prompt_feedback_text()
        if not feedback:
            return result
        try:
            from lerobot.lerobot_types import TransitionKey

            complementary = result.get(TransitionKey.COMPLEMENTARY_DATA) or {}
            task_key = getattr(self, "task_key", "task")
            prompts = complementary.get(task_key)
            if isinstance(prompts, str):
                updated = _inject_visible_feedback(prompts, feedback)
                complementary[task_key] = updated
                _RECORDER.record_prompt_feedback(feedback, [updated])
            elif isinstance(prompts, (list, tuple)):
                updated_list = [
                    _inject_visible_feedback(prompt, feedback) if isinstance(prompt, str) else prompt
                    for prompt in prompts
                ]
                complementary[task_key] = updated_list
                _RECORDER.record_prompt_feedback(
                    feedback,
                    [prompt for prompt in updated_list if isinstance(prompt, str)],
                )
            result[TransitionKey.COMPLEMENTARY_DATA] = complementary
        except Exception as exc:
            _RECORDER._write(
                {
                    "type": "prompt_feedback_error",
                    "time": time.time(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        return result

    _patched_prepare_state_tokenizer._activation_capture_patched = True
    step_cls.__call__ = _patched_prepare_state_tokenizer


try:
    import lerobot.policies as _policies
    import lerobot.envs as _envs
    import lerobot.envs.factory as _env_factory
    import lerobot.policies.factory as _policy_factory

    _patch_make_policy(_policies, "make_policy")
    _patch_make_policy(_policy_factory, "make_policy")
    _patch_make_env(_envs, "make_env")
    _patch_make_env(_env_factory, "make_env")
    _patch_pi05_prompt_feedback()
except Exception as exc:  # pragma: no cover - best effort patching
    print(f"[activation_capture] failed to patch runtime hooks: {exc}", file=sys.stderr)

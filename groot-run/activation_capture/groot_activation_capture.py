"""Optional GR00T activation capture.

Loaded automatically when this directory is on PYTHONPATH and
CAPTURE_ACTIVATIONS=1. Event schema matches the Pi0.5 capture so the
forked report scripts can read events.jsonl the same way.

Families are remapped onto GR00T N1.7 modules:
  vision     -> backbone vision / visual encoder layers
  prefix     -> backbone language / LLM layers
  expert     -> action_head DiT / diffusion blocks
  projection -> action/state encoders and decoders
"""

from __future__ import annotations

import atexit
import json
import os
import re
import sys
import time
import types
from pathlib import Path
from typing import Any


def _enabled() -> bool:
    value = os.environ.get("GROOT_CAPTURE_ACTIVATIONS") or os.environ.get("CAPTURE_ACTIVATIONS")
    return str(value).lower() in {"1", "true", "yes", "on"}


try:
    import numpy as np
    import torch
except Exception as exc:  # pragma: no cover
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


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "value"


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
    for attr in ("last_hidden_state", "pooler_output", "hidden_states"):
        item = getattr(value, attr, None)
        if torch.is_tensor(item):
            return item
        if isinstance(item, (tuple, list)) and item and torch.is_tensor(item[0]):
            return item[0]
    return None


def _downsample(values: torch.Tensor, bins: int) -> list[float]:
    flat = values.detach().float().flatten()
    if flat.numel() == 0:
        return []
    if flat.numel() <= bins:
        return [round(float(x), 6) for x in flat.cpu().tolist()]
    chunks = torch.chunk(flat, bins)
    return [round(float(chunk.mean().cpu()), 6) for chunk in chunks]


class ActivationRecorder:
    def __init__(self) -> None:
        capture_dir = os.environ.get("GROOT_CAPTURE_DIR") or os.environ.get(
            "LEROBOT_CAPTURE_DIR", "activation_capture"
        )
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
        families = os.environ.get("CAPTURE_FAMILIES", "vision,prefix,expert,projection")
        self.families = {item.strip() for item in families.split(",") if item.strip()}
        self.chunk_index = -1
        self.captured_chunks = 0
        self.active = False
        self.hook_call = 0
        self.image_count = 0
        self.hooks = []
        self._write(
            {
                "type": "capture_start",
                "time": time.time(),
                "model": "groot_n1.7",
                "families": sorted(self.families),
                "layer_stride": self.layer_stride,
                "max_bins": self.max_bins,
                "max_chunks": self.max_chunks,
                "every_n_chunks": self.every_n_chunks,
                "sample_rate": self.sample_rate,
                "capture_param_stats": self.capture_param_stats,
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
            "time": time.time(),
            "batch_keys": sorted(str(key) for key in batch.keys()),
        }
        task = _extract_task(batch)
        if task:
            payload["task"] = task
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
            arr = _to_image_array(key, value)
            if arr is None:
                continue
            try:
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
                self._write(
                    {"type": "image_error", "chunk": self.chunk_index, "key": str(key), "error": str(exc)}
                )

    def record_action_chunk(self, actions: Any) -> None:
        if not self.active:
            return
        try:
            arr = _actions_to_array(actions)
            if arr is None or arr.size == 0:
                return
            values = arr[:, : min(arr.shape[-1], 12)]
            self._write(
                {
                    "type": "action_chunk",
                    "chunk": self.chunk_index,
                    "shape": list(arr.shape),
                    "values": [[round(float(x), 6) for x in row] for row in values.tolist()],
                    "norm_per_step": [round(float(x), 6) for x in np.linalg.norm(arr, axis=-1).tolist()],
                    "abs_per_dim": [round(float(x), 6) for x in np.abs(arr).mean(axis=0).tolist()],
                }
            )
        except Exception as exc:
            self._write({"type": "action_error", "chunk": self.chunk_index, "error": str(exc)})

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


def _extract_task(batch: dict[str, Any]) -> str:
    for key in ("task", "language", "annotation.human.coarse_action"):
        value = batch.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)) and value:
            item = value[0]
            if isinstance(item, str):
                return item
            if isinstance(item, (list, tuple)) and item and isinstance(item[0], str):
                return item[0]
    language = batch.get("language")
    if isinstance(language, dict):
        for value in language.values():
            if isinstance(value, str):
                return value
            if isinstance(value, (list, tuple)) and value:
                item = value[0]
                if isinstance(item, str):
                    return item
                if isinstance(item, (list, tuple)) and item and isinstance(item[0], str):
                    return item[0]
    return ""


def _to_image_array(key: str, value: Any) -> np.ndarray | None:
    key_l = str(key).lower()
    if not any(token in key_l for token in ("video", "image", "camera", "agentview", "wrist")):
        return None
    arr: np.ndarray | None = None
    if torch.is_tensor(value):
        arr = value.detach().to("cpu").float().numpy()
    elif isinstance(value, np.ndarray):
        arr = value
    if arr is None:
        return None
    while arr.ndim > 3:
        arr = arr[0]
    if arr.ndim != 3:
        return None
    if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.dtype != np.uint8:
        if arr.max() > 2:
            arr = np.clip(arr, 0, 255)
        elif arr.min() < 0:
            arr = (arr + 1.0) / 2.0 * 255.0
        else:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _actions_to_array(actions: Any) -> np.ndarray | None:
    if torch.is_tensor(actions):
        arr = actions.detach().to("cpu").float().numpy()
    elif isinstance(actions, np.ndarray):
        arr = actions.astype(np.float32)
    elif isinstance(actions, dict):
        parts = []
        for key in sorted(actions):
            value = actions[key]
            if torch.is_tensor(value):
                value = value.detach().to("cpu").float().numpy()
            if isinstance(value, np.ndarray):
                parts.append(value)
        if not parts:
            return None
        arr = np.concatenate(parts, axis=-1)
    else:
        return None
    while arr.ndim > 2:
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr.astype(np.float32)


def _layer_index(name: str) -> int | None:
    match = re.search(r"(?:layers|blocks|layer|block)\.(\d+)", name)
    if match:
        return int(match.group(1))
    return None


def _classify_module(name: str) -> tuple[str, int | None] | None:
    lowered = name.lower()
    layer = _layer_index(name)

    if re.search(r"action_head\.(action_encoder|action_decoder|state_encoder)\b", name):
        return "projection", layer
    if "action_head" in lowered and re.search(r"(?:layers|blocks)\.\d+$", name):
        return "expert", layer
    if "action_head.model" in lowered and layer is not None:
        return "expert", layer

    vision_tokens = ("vision", "visual", "vision_tower", "vision_model", "siglip", "clip")
    if any(token in lowered for token in vision_tokens) and layer is not None:
        return "vision", layer
    if any(token in lowered for token in vision_tokens) and re.search(r"(encoder|patch_embed)$", name):
        return "vision", layer

    if "backbone" in lowered and layer is not None:
        if any(token in lowered for token in vision_tokens):
            return "vision", layer
        return "prefix", layer

    projection_names = (
        "action_encoder",
        "action_decoder",
        "state_encoder",
        "action_in_proj",
        "action_out_proj",
    )
    if any(name == item or name.endswith(f".{item}") for item in projection_names):
        return "projection", None
    return None


_RECORDER = ActivationRecorder()


def _install_on_policy(policy: Any) -> None:
    if getattr(policy, "_activation_capture_installed", False):
        return
    model = getattr(policy, "model", None)
    if model is None and hasattr(policy, "policy"):
        inner = getattr(policy, "policy")
        model = getattr(inner, "model", None)
        if model is not None:
            policy = inner
    if model is None:
        print("[activation_capture] no model attribute on policy", file=sys.stderr)
        return

    selected = []
    selected_modules = []
    for name, module in model.named_modules():
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

    original_get_action = policy.get_action

    def wrapped_get_action(self: Any, observation: dict[str, Any], *args: Any, **kwargs: Any):
        _RECORDER.begin_chunk(observation if isinstance(observation, dict) else {})
        try:
            result = original_get_action(observation, *args, **kwargs)
            actions = result[0] if isinstance(result, tuple) else result
            _RECORDER.record_action_chunk(actions)
            return result
        finally:
            _RECORDER.end_chunk()

    policy.get_action = types.MethodType(wrapped_get_action, policy)
    policy._activation_capture_installed = True
    _RECORDER._write({"type": "hooks_installed", "count": len(selected), "modules": selected})
    print(f"[activation_capture] installed {len(selected)} hooks", file=sys.stderr)


def _patch_class(module: Any, class_name: str) -> None:
    cls = getattr(module, class_name, None)
    if cls is None or getattr(cls, "_activation_capture_patched", False):
        return
    original_init = cls.__init__

    def patched_init(self: Any, *args: Any, **kwargs: Any):
        original_init(self, *args, **kwargs)
        try:
            _install_on_policy(self)
        except Exception as exc:
            print(f"[activation_capture] install failed: {exc}", file=sys.stderr)

    cls.__init__ = patched_init
    cls._activation_capture_patched = True


try:
    import gr00t.policy.gr00t_policy as _groot_policy

    _patch_class(_groot_policy, "Gr00tPolicy")
except Exception as exc:  # pragma: no cover
    print(f"[activation_capture] failed to patch GR00T policy: {exc}", file=sys.stderr)

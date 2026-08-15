"""Optional LeRobot PI0.5 activation capture.

Loaded automatically when this directory is on PYTHONPATH. The runner enables it
only when CAPTURE_ACTIVATIONS=1.
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
        task = batch.get("task")
        if isinstance(task, str):
            payload["task"] = task
        elif isinstance(task, (list, tuple)) and task and isinstance(task[0], str):
            payload["task"] = task[0]
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
                    "shape": list(actions.shape),
                    "values": [[round(float(x), 6) for x in row] for row in values.tolist()],
                    "norm_per_step": [round(float(x), 6) for x in arr.norm(dim=-1).tolist()],
                    "abs_per_dim": [round(float(x), 6) for x in arr.abs().mean(dim=0).tolist()],
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

    policy.predict_action_chunk = types.MethodType(wrapped_predict_action_chunk, policy)
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


try:
    import lerobot.policies as _policies
    import lerobot.policies.factory as _policy_factory

    _patch_make_policy(_policies, "make_policy")
    _patch_make_policy(_policy_factory, "make_policy")
except Exception as exc:  # pragma: no cover - best effort patching
    print(f"[activation_capture] failed to patch make_policy: {exc}", file=sys.stderr)

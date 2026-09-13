"""Optional Langfuse tracing helpers for Pi0.5 mechanistic probes."""

from __future__ import annotations

import contextlib
import json
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterator


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _round(value: Any, digits: int = 6) -> float | None:
    number = _float(value)
    return None if number is None else round(number, digits)


def _is_media(value: Any) -> bool:
    return value.__class__.__name__ in {"LangfuseMedia", "LangfuseMediaReference"}


def _jsonable(value: Any, *, max_list_items: int = 256) -> Any:
    if _is_media(value):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item, max_list_items=max_list_items) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        values = [_jsonable(item, max_list_items=max_list_items) for item in list(value)[:max_list_items]]
        if len(value) > max_list_items:
            values.append({"truncated_items": len(value) - max_list_items})
        return values
    if hasattr(value, "detach") and hasattr(value, "shape"):
        return summarize_tensor(value)
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return _jsonable(value.item(), max_list_items=max_list_items)
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        json.dumps(value, allow_nan=False)
        return value
    except Exception:
        return str(value)


def _shape(value: Any) -> list[int] | None:
    raw = getattr(value, "shape", None)
    if raw is None:
        return None
    try:
        return [int(item) for item in raw]
    except Exception:
        return None


def _tensor_cpu_float(value: Any) -> Any:
    tensor = value.detach() if hasattr(value, "detach") else value
    if hasattr(tensor, "to"):
        try:
            tensor = tensor.to("cpu")
        except Exception:
            tensor = tensor.cpu()
    if hasattr(tensor, "float"):
        tensor = tensor.float()
    return tensor


def _tensor_stats(value: Any) -> dict[str, Any]:
    shape = _shape(value)
    dtype = str(getattr(value, "dtype", "unknown")).replace("torch.", "")
    result: dict[str, Any] = {"shape": shape, "dtype": dtype}
    try:
        tensor = _tensor_cpu_float(value)
        if int(tensor.numel()) == 0:
            return result
        result.update(
            {
                "mean": _round(tensor.mean()),
                "std": _round(tensor.std()) if int(tensor.numel()) > 1 else 0.0,
                "min": _round(tensor.min()),
                "max": _round(tensor.max()),
                "abs_mean": _round(tensor.abs().mean()),
                "l2": _round(tensor.norm()),
            }
        )
    except Exception:
        pass
    return result


def _preview_tensor(value: Any, *, max_rows: int = 10, max_cols: int = 12, max_flat: int = 64) -> list[Any]:
    try:
        tensor = _tensor_cpu_float(value)
        if len(tensor.shape) >= 3:
            tensor = tensor[0, :max_rows, :max_cols]
        elif len(tensor.shape) == 2:
            tensor = tensor[:max_rows, :max_cols]
        else:
            tensor = tensor.reshape(-1)[:max_flat]
        return json.loads(json.dumps(tensor.tolist(), allow_nan=False))
    except Exception:
        return []


def summarize_tensor(value: Any, *, include_preview: bool = True) -> dict[str, Any]:
    summary = _tensor_stats(value)
    if include_preview:
        summary["preview"] = _preview_tensor(value)
    return summary


def summarize_action_tensor(actions: Any, *, executed_steps: int = 10, max_values: int = 500) -> dict[str, Any]:
    summary = _tensor_stats(actions)
    try:
        arr = _tensor_cpu_float(actions)
        if len(arr.shape) == 3:
            arr = arr[0]
        steps = int(arr.shape[0]) if len(arr.shape) >= 2 else 0
        dims = int(arr.shape[-1]) if len(arr.shape) >= 2 else 0
        summary["planned_steps"] = steps
        summary["action_dims"] = dims
        summary["executed_window_steps"] = min(max(0, executed_steps), steps)
        if steps and dims:
            summary["executed_window"] = _preview_tensor(arr[:executed_steps], max_rows=executed_steps, max_cols=dims)
            summary["norm_per_step"] = [
                _round(item) for item in arr.norm(dim=-1).reshape(-1)[: min(steps, 64)].tolist()
            ]
            summary["abs_per_dim"] = [_round(item) for item in arr.abs().mean(dim=0).reshape(-1)[: min(dims, 64)].tolist()]
            if steps * dims <= max_values:
                summary["planned_action_chunk"] = _preview_tensor(arr, max_rows=steps, max_cols=dims)
    except Exception:
        pass
    return summary


def summarize_diffusion_tensor(kind: str, value: Any, timestep: Any | None) -> dict[str, Any]:
    result = summarize_tensor(value, include_preview=False)
    result["kind"] = kind
    result["preview"] = _preview_tensor(value, max_rows=4, max_cols=12)
    if timestep is not None:
        result["timestep"] = _preview_tensor(timestep, max_flat=16)
    return result


def _task_from_batch(batch: dict[str, Any]) -> str | None:
    task = batch.get("task")
    if isinstance(task, str):
        return task
    if isinstance(task, (list, tuple)) and task and isinstance(task[0], str):
        return task[0]
    return None


def _tensor_shape_map(batch: dict[str, Any]) -> dict[str, Any]:
    shapes: dict[str, Any] = {}
    for key, value in batch.items():
        shape = _shape(value)
        if shape is not None:
            shapes[str(key)] = {"shape": shape, "dtype": str(getattr(value, "dtype", "unknown")).replace("torch.", "")}
    return shapes


class LangfuseRobotTracer:
    """Small optional wrapper around the Langfuse Python SDK.

    The tracing path must never break robot rollouts. If credentials or the SDK
    are missing, all methods degrade to no-ops while local artifacts are still
    written.
    """

    def __init__(self, *, feature: str, output_root: str | Path | None = None) -> None:
        self.feature = feature
        self.output_root = str(output_root) if output_root is not None else None
        self.enabled = False
        self.client: Any | None = None
        self._propagate_attributes: Any | None = None
        self._media_class: Any | None = None
        self._warned = False
        self.media_enabled = _bool_env("PI05_LANGFUSE_MEDIA", True)
        self.max_media_bytes = _int_env("PI05_LANGFUSE_MAX_MEDIA_BYTES", 2_000_000)
        self.max_images = _int_env("PI05_LANGFUSE_MAX_IMAGES", 2)

        if not _bool_env("PI05_LANGFUSE_TRACE", False):
            return
        if os.environ.get("LANGFUSE_BASE_URL") and not os.environ.get("LANGFUSE_HOST"):
            os.environ["LANGFUSE_HOST"] = os.environ["LANGFUSE_BASE_URL"]
        if not os.environ.get("LANGFUSE_PUBLIC_KEY") or not os.environ.get("LANGFUSE_SECRET_KEY"):
            self._warn("PI05_LANGFUSE_TRACE=1 but Langfuse credentials are not set; tracing disabled.")
            return
        try:
            from langfuse import get_client, propagate_attributes
            from langfuse.media import LangfuseMedia

            self.client = get_client()
            self._propagate_attributes = propagate_attributes
            self._media_class = LangfuseMedia
            self.enabled = True
        except Exception as exc:
            self._warn(f"Langfuse tracing disabled: {type(exc).__name__}: {exc}")

    def _warn(self, message: str) -> None:
        if self._warned:
            return
        self._warned = True
        print(f"[pi05_langfuse] {message}", file=sys.stderr)

    def _trace_tags(self, extra_tags: list[str] | None = None) -> list[str]:
        tags = ["pi05", "transcoder", self.feature]
        raw = os.environ.get("PI05_LANGFUSE_TAGS", "")
        tags.extend(item.strip() for item in raw.split(",") if item.strip())
        if extra_tags:
            tags.extend(extra_tags)
        seen = set()
        return [tag for tag in tags if not (tag in seen or seen.add(tag))]

    def _trace_metadata(self, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "feature": self.feature,
            "caller": os.environ.get("PI05_LANGFUSE_CALLER") or os.environ.get("USER") or "unknown",
        }
        if self.output_root:
            result["output_root"] = self.output_root
        for env_name, key in [
            ("RUN_ID", "run_id"),
            ("PI05_LANGFUSE_TENANT_ID", "tenant_id"),
            ("PI05_TENANT_ID", "tenant_id"),
            ("PI05_LANGFUSE_RUN_GROUP", "run_group"),
            ("PI05_TRANSCODER_MODE", "transcoder_mode"),
        ]:
            value = os.environ.get(env_name)
            if value and key not in result:
                result[key] = value
        if metadata:
            result.update(_jsonable(metadata))
        return result

    @contextlib.contextmanager
    def trace(
        self,
        name: str,
        *,
        input: Any | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> Iterator[Any]:
        trace_metadata = self._trace_metadata(metadata)
        if not self.enabled or self._propagate_attributes is None:
            with self.observation(
                name,
                as_type="span",
                input=input,
                metadata=trace_metadata,
            ) as obs:
                yield obs
            return

        propagate_kwargs: dict[str, Any] = {
            "trace_name": name,
            "metadata": trace_metadata,
            "tags": self._trace_tags(tags),
        }
        user_id = os.environ.get("PI05_LANGFUSE_USER_ID") or os.environ.get("USER")
        if user_id:
            propagate_kwargs["user_id"] = user_id
        session_id = (
            os.environ.get("PI05_LANGFUSE_SESSION_ID")
            or os.environ.get("RUN_ID")
            or os.environ.get("TRANSCODER_PROBE_NAME")
        )
        if session_id:
            propagate_kwargs["session_id"] = session_id
        environment = os.environ.get("PI05_LANGFUSE_ENVIRONMENT")
        if environment:
            propagate_kwargs["environment"] = environment
        stack = ExitStack()
        try:
            stack.enter_context(self._propagate_attributes(**propagate_kwargs))
        except TypeError:
            # Older SDKs may not accept every v4 propagation field.
            stack.close()
            stack = ExitStack()
            compact = {key: value for key, value in propagate_kwargs.items() if key in {"user_id", "session_id", "tags"}}
            stack.enter_context(self._propagate_attributes(**compact))
        with stack:
            with self.observation(
                name,
                as_type="span",
                input=input,
                metadata=trace_metadata,
            ) as obs:
                yield obs

    @contextlib.contextmanager
    def observation(
        self,
        name: str,
        *,
        as_type: str = "span",
        input: Any | None = None,
        output: Any | None = None,
        metadata: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> Iterator[Any]:
        if not self.enabled or self.client is None:
            yield _NoopObservation()
            return

        kwargs: dict[str, Any] = {"as_type": as_type, "name": name}
        if input is not None:
            kwargs["input"] = _jsonable(input)
        if metadata is not None:
            kwargs["metadata"] = _jsonable(metadata)
        if model is not None:
            kwargs["model"] = model

        manager = None
        obs = None
        try:
            manager = self.client.start_as_current_observation(**kwargs)
            obs = manager.__enter__()
        except Exception as exc:
            self.enabled = False
            self._warn(f"could not start Langfuse observation {name!r}: {type(exc).__name__}: {exc}")
            yield _NoopObservation()
            return

        error: BaseException | None = None
        try:
            yield obs
            if output is not None:
                self.update(obs, output=output)
        except BaseException as exc:
            error = exc
            self.update(obs, metadata={"error": f"{type(exc).__name__}: {exc}"})
            raise
        finally:
            try:
                if error is None:
                    manager.__exit__(None, None, None)
                else:
                    manager.__exit__(type(error), error, getattr(error, "__traceback__", None))
            except Exception as exc:
                self._warn(f"could not end Langfuse observation {name!r}: {type(exc).__name__}: {exc}")

    def update(
        self,
        observation: Any,
        *,
        input: Any | None = None,
        output: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        payload: dict[str, Any] = {}
        if input is not None:
            payload["input"] = _jsonable(input)
        if output is not None:
            payload["output"] = _jsonable(output)
        if metadata is not None:
            payload["metadata"] = _jsonable(metadata)
        if not payload:
            return
        try:
            observation.update(**payload)
        except Exception as exc:
            self._warn(f"could not update Langfuse observation: {type(exc).__name__}: {exc}")

    def media_from_path(self, path: str | Path, *, content_type: str | None = None) -> Any | None:
        if not self.enabled or not self.media_enabled or self._media_class is None:
            return None
        path = Path(path)
        try:
            if not path.exists() or path.stat().st_size > self.max_media_bytes:
                return None
            return self._media_class(content_bytes=path.read_bytes(), content_type=content_type or _guess_content_type(path))
        except Exception as exc:
            self._warn(f"could not attach Langfuse media {path}: {type(exc).__name__}: {exc}")
            return None

    def media_from_tensor(self, value: Any) -> Any | None:
        if not self.enabled or not self.media_enabled or self._media_class is None:
            return None
        try:
            import cv2
            import numpy as np

            arr = value.detach()[0].to("cpu").float().numpy() if hasattr(value, "detach") else np.asarray(value)
            if arr.ndim != 3:
                return None
            if arr.shape[0] in (1, 3):
                arr = np.moveaxis(arr, 0, -1)
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            if arr.min() < 0:
                arr = (arr + 1.0) / 2.0
            if arr.max() > 2:
                arr = arr / 255.0
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            max_side = _int_env("PI05_LANGFUSE_MAX_IMAGE_SIDE", 512)
            h, w = arr.shape[:2]
            if max(h, w) > max_side:
                scale = max_side / float(max(h, w))
                arr = cv2.resize(arr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if not ok:
                return None
            content = encoded.tobytes()
            if len(content) > self.max_media_bytes:
                return None
            return self._media_class(content_bytes=content, content_type="image/jpeg")
        except Exception as exc:
            self._warn(f"could not encode Langfuse image media: {type(exc).__name__}: {exc}")
            return None

    def batch_input(self, batch: dict[str, Any]) -> dict[str, Any]:
        image_payload: dict[str, Any] = {}
        attached = 0
        for key in sorted(str(key) for key in batch if str(key).startswith("observation.images.")):
            value = batch[key]
            item = _tensor_stats(value)
            if attached < self.max_images:
                media = self.media_from_tensor(value)
                if media is not None:
                    item["media"] = media
                    attached += 1
            image_payload[key] = item
        return {
            "task": _task_from_batch(batch),
            "batch_keys": sorted(str(key) for key in batch.keys()),
            "tensor_shapes": _tensor_shape_map(batch),
            "images": image_payload,
        }

    def flush(self) -> None:
        if not self.enabled or self.client is None:
            return
        try:
            self.client.flush()
        except Exception as exc:
            self._warn(f"Langfuse flush failed: {type(exc).__name__}: {exc}")


class _NoopObservation:
    def update(self, *args: Any, **kwargs: Any) -> "_NoopObservation":
        return self

    def end(self) -> None:
        return None


def _guess_content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".svg": "image/svg+xml",
        ".html": "text/html",
        ".csv": "text/csv",
        ".json": "application/json",
        ".jsonl": "application/x-ndjson",
        ".pdf": "application/pdf",
        ".mp4": "video/mp4",
        ".zip": "application/zip",
        ".gz": "application/gzip",
    }.get(suffix, "application/octet-stream")


def make_langfuse_tracer(*, feature: str, output_root: str | Path | None = None) -> LangfuseRobotTracer:
    return LangfuseRobotTracer(feature=feature, output_root=output_root)

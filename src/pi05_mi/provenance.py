"""Record what a run actually ran on, so a result can be tied to code and weights.

A configuration block says what was asked for. Provenance says what answered:
the commit the scripts came from, the library versions that shaped the
numbers, the device and dtype, and a content hash of the transcoder
checkpoint, because two files with the same name are not the same weights.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

TRACKED_PACKAGES = ("torch", "numpy", "lerobot", "transformers", "mujoco", "robosuite", "libero", "safetensors")


def git_state(repo_root: Path) -> dict[str, Any]:
    """Commit hash, branch and whether the tree was dirty when the run started."""
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", "-C", str(repo_root), *args], capture_output=True, text=True, check=True, timeout=10
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": commit,
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def package_versions(names: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str | None]:
    from importlib import metadata

    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def file_digest(path: Path, *, algorithm: str = "sha256", chunk_bytes: int = 1 << 24) -> dict[str, Any]:
    """Content hash and size of a file, streamed so multi-GB checkpoints fit."""
    path = Path(path)
    digest = hashlib.new(algorithm)
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return {"path": str(path), "algorithm": algorithm, "digest": digest.hexdigest(), "bytes": size}


def device_info(device: Any) -> dict[str, Any]:
    info: dict[str, Any] = {"device": str(device)}
    try:
        import torch

        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        if str(device).startswith("cuda") and torch.cuda.is_available():
            index = torch.device(device).index or 0
            info["gpu_name"] = torch.cuda.get_device_name(index)
            info["gpu_memory_bytes"] = int(torch.cuda.get_device_properties(index).total_memory)
        if hasattr(torch.backends, "cudnn"):
            info["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
            info["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    except Exception as exc:  # pragma: no cover - torch missing or broken
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def collect_provenance(
    *, repo_root: Path, device: Any, checkpoint: Path | None, policy_path: str, policy_dtype: str, extra: dict | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "git": git_state(repo_root),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
        "device": device_info(device),
        "policy": {"path": policy_path, "dtype": policy_dtype},
        "transcoder_checkpoint": None if checkpoint is None else file_digest(Path(checkpoint)),
    }
    if extra:
        payload.update(extra)
    return payload

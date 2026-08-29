"""Download nvidia/GR00T-N1.7-LIBERO/libero_10 from Hugging Face if needed.

Checkpoints stay local / Drive. They are not committed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO_ID = "nvidia/GR00T-N1.7-LIBERO"
SUBDIR = "libero_10"
REQUIRED_MODEL_TYPE = "Gr00tN1d7"
REQUIRED_NAMES = (
    "config.json",
    "model.safetensors.index.json",
    "processor_config.json",
    "statistics.json",
)


def _config_ok(model_dir: Path) -> bool:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return config.get("model_type") == REQUIRED_MODEL_TYPE


def _weights_ok(model_dir: Path) -> bool:
    if not _config_ok(model_dir):
        return False
    shards = list(model_dir.glob("model-*.safetensors"))
    return all((model_dir / name).is_file() for name in REQUIRED_NAMES) and bool(shards)


def download_libero_10(dest_root: Path, token: str | None, *, config_only: bool) -> Path:
    from huggingface_hub import snapshot_download

    dest_root.mkdir(parents=True, exist_ok=True)
    patterns = [f"{SUBDIR}/config.json"] if config_only else [f"{SUBDIR}/*"]
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(dest_root),
        allow_patterns=patterns,
        token=token or None,
    )
    return dest_root / SUBDIR


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest-root",
        type=Path,
        required=True,
        help="Directory that should contain libero_10/, e.g. checkpoints/GR00T-N1.7-LIBERO",
    )
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Download config.json only and check model_type. For local smoke tests.",
    )
    args = parser.parse_args()

    model_dir = args.dest_root / SUBDIR
    token = args.token.strip() or None
    if not args.force:
        if args.config_only and _config_ok(model_dir):
            print("OK existing config", model_dir / "config.json")
            return 0
        if (not args.config_only) and _weights_ok(model_dir):
            print("OK existing checkpoint", model_dir)
            return 0

    print(f"Downloading {REPO_ID}/{SUBDIR} -> {args.dest_root}")
    model_dir = download_libero_10(args.dest_root, token, config_only=args.config_only)
    if args.config_only:
        if not _config_ok(model_dir):
            raise SystemExit(f"Downloaded config is invalid: {model_dir / 'config.json'}")
        print("OK config model_type=", REQUIRED_MODEL_TYPE)
        return 0
    if not _weights_ok(model_dir):
        raise SystemExit(
            f"Download finished but checkpoint is incomplete: {model_dir}. "
            "Accept the gated HF repos and rerun with a valid HF_TOKEN."
        )
    print("OK checkpoint", model_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

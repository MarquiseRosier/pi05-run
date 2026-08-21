#!/usr/bin/env python
"""Check local Hugging Face cache entries needed for Pi0.5 transcoder runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import try_to_load_from_cache
from lerobot.utils.constants import HF_LEROBOT_HUB_CACHE


DEFAULT_POLICY_REPO = "lerobot/pi05_libero_finetuned"
DEFAULT_DATASET_REPO = "HuggingFaceVLA/libero"
DEFAULT_PALIGEMMA_REPO = "google/paligemma-3b-pt-224"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-repo", default=DEFAULT_POLICY_REPO)
    parser.add_argument("--dataset-repo", default=DEFAULT_DATASET_REPO)
    parser.add_argument("--paligemma-repo", default=DEFAULT_PALIGEMMA_REPO)
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser.parse_args()


def cache_status(repo_id: str, filename: str, *, repo_type: str | None, cache_dir: Path | None) -> tuple[bool, str]:
    path = try_to_load_from_cache(repo_id, filename, repo_type=repo_type, cache_dir=cache_dir)
    if isinstance(path, str):
        return True, path
    return False, "not found in local cache"


def print_check(label: str, repo_id: str, filename: str, *, repo_type: str | None, cache_dir: Path | None) -> bool:
    found, detail = cache_status(repo_id, filename, repo_type=repo_type, cache_dir=cache_dir)
    marker = "OK" if found else "MISSING"
    type_text = repo_type or "model"
    print(f"[{marker}] {label}: {type_text}:{repo_id}/{filename}")
    print(f"       {detail}")
    return found


def dataset_cache_dir(cache_dir: Path | None) -> Path | None:
    return cache_dir if cache_dir is not None else HF_LEROBOT_HUB_CACHE


def main() -> None:
    args = parse_args()
    checks = [
        print_check(
            "Pi0.5 policy config",
            args.policy_repo,
            "config.json",
            repo_type=None,
            cache_dir=args.cache_dir,
        ),
        print_check(
            "Pi0.5 training config",
            args.policy_repo,
            "train_config.json",
            repo_type=None,
            cache_dir=args.cache_dir,
        ),
        print_check(
            "Pi0.5 policy weights",
            args.policy_repo,
            "model.safetensors",
            repo_type=None,
            cache_dir=args.cache_dir,
        ),
        print_check(
            "LIBERO dataset metadata",
            args.dataset_repo,
            "meta/info.json",
            repo_type="dataset",
            cache_dir=dataset_cache_dir(args.cache_dir),
        ),
        print_check(
            "PaliGemma tokenizer/config",
            args.paligemma_repo,
            "config.json",
            repo_type=None,
            cache_dir=args.cache_dir,
        ),
    ]
    if all(checks):
        print("cache check passed")
    else:
        raise SystemExit("cache check failed; remove --local-files-only from the training command to allow downloads")


if __name__ == "__main__":
    main()

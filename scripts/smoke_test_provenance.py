#!/usr/bin/env python
"""Smoke test: a run records what it ran on."""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pi05_mi.provenance import collect_provenance, file_digest, git_state, package_versions  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def test_git_state_names_the_commit() -> None:
    state = git_state(REPO)
    assert state["commit"] and len(state["commit"]) == 40, state
    assert state["branch"] and isinstance(state["dirty"], bool)


def test_file_digest_matches_hashlib_and_streams() -> None:
    tmp = Path(tempfile.mkdtemp()) / "ckpt.bin"
    payload = bytes(range(256)) * 70000  # ~17.9 MB, crosses one 16 MB chunk boundary
    tmp.write_bytes(payload)
    out = file_digest(tmp)
    assert out["digest"] == hashlib.sha256(payload).hexdigest()
    assert out["bytes"] == len(payload) and out["algorithm"] == "sha256"


def test_package_versions_report_what_is_installed() -> None:
    versions = package_versions(("torch", "numpy", "definitely-not-a-package"))
    assert versions["torch"] and versions["numpy"]
    assert versions["definitely-not-a-package"] is None


def test_collect_provenance_has_every_section() -> None:
    tmp = Path(tempfile.mkdtemp()) / "ckpt.pt"
    tmp.write_bytes(b"weights")
    out = collect_provenance(
        repo_root=REPO, device="cpu", checkpoint=tmp, policy_path="org/policy", policy_dtype="bfloat16",
        extra={"note": "x"},
    )
    for key in ("git", "python", "platform", "packages", "device", "policy", "transcoder_checkpoint", "note"):
        assert key in out, key
    assert out["transcoder_checkpoint"]["digest"] == hashlib.sha256(b"weights").hexdigest()
    assert out["device"]["device"] == "cpu"


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok {test.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    main()

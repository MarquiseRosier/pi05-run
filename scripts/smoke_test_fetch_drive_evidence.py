#!/usr/bin/env python
"""Smoke test for the Drive evidence fetcher.

The network half needs credentials and is not tested here. The half that can
silently do the wrong thing is the filter: fetching a delta store would move
gigabytes, and skipping a decision artefact would produce an empty report that
looks like a finished one. Those, and the URL parsing that decides which folder
is read at all, are what these cover.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_drive_evidence import DEFAULT_MAX_BYTES, folder_id, wanted  # noqa: E402

FID = "1QEHe98wrQmTeU4715qLVm08JIMiZnrVO"


def _entry(name: str, size: int = 4096) -> dict:
    return {"rel": Path(name), "size": size}


def test_every_drive_url_shape_yields_the_same_id() -> None:
    for value in (
        FID,
        f"https://drive.google.com/drive/folders/{FID}",
        f"https://drive.google.com/drive/u/0/folders/{FID}",
        f"https://drive.google.com/drive/u/3/folders/{FID}?usp=sharing",
        f"https://drive.google.com/open?id={FID}",
    ):
        assert folder_id(value) == FID, value


def test_every_artefact_the_collector_reads_is_fetched() -> None:
    """If one of these is filtered out the report is silently incomplete."""
    for name in (
        "20260925/task00_seed1000/decision_metrics.json",
        "20260925/task00_seed1000/counterfactual_summary.json",
        "20260925/task00_seed1000/provenance.json",
        "20260925/task00_seed1000/nominated_targets.json",
        "20260925/task00_seed1000/h1_cells.csv",
        "20260925/aggregate/aggregate.json",
        "20260925/aggregate/per_task.csv",
        "20260925/h3/h3_rate.json",
        "20260925/h3/calibration_task00_L5/audit_calibration.json",
        "20260925/manifest.json",
        "20260925/task00_seed1000/probe.log",
    ):
        ok, why = wanted(_entry(name), DEFAULT_MAX_BYTES)
        assert ok, f"{name} would be skipped ({why})"


def test_bulk_artefacts_are_never_fetched() -> None:
    """These are the gigabytes, and no table is filled from any of them."""
    for name in ("latents/block_000.npz", "images/state0_baseline.png",
                 "circuit/graph.json.zip", "checkpoints/step_027233.pt",
                 "circuit/circuit_graph.svg", "circuit/circuit_report.html"):
        ok, why = wanted(_entry(name, size=900_000_000), DEFAULT_MAX_BYTES)
        assert not ok, name
        assert why, "a skip must say why"


def test_an_oversized_json_is_skipped_with_its_size_named() -> None:
    """graph.json is 12 MB on the pilot and carries no decision quantity."""
    ok, why = wanted(_entry("circuit/graph.json", size=12_476_031), DEFAULT_MAX_BYTES)
    assert not ok and "MB over the cap" in why, why
    # The same file under the cap is taken, so the rule is size and not name.
    ok, _ = wanted(_entry("circuit/graph.json", size=1024), DEFAULT_MAX_BYTES)
    assert ok


def test_the_cap_is_a_parameter_not_a_constant() -> None:
    entry = _entry("aggregate/aggregate.json", size=5_000_000)
    assert wanted(entry, DEFAULT_MAX_BYTES)[0]
    assert not wanted(entry, 1_000_000)[0]


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok {test.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    main()

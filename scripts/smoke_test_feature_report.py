#!/usr/bin/env python
"""Fast smoke test for the Pi0.5 feature inspection report."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from pi05_mi.feature_discovery import FeatureDiscoveryCollector, FeatureDiscoveryConfig


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="pi05_feature_report_"))
    layer_names = [
        "paligemma_with_expert.gemma_expert.model.layers.0.mlp",
        "paligemma_with_expert.gemma_expert.model.layers.1.mlp",
    ]
    collector = FeatureDiscoveryCollector(
        layer_names=layer_names,
        layer_indices={layer_names[0]: 0, layer_names[1]: 1},
        d_features=8,
        config=FeatureDiscoveryConfig(top_k=3, firing_threshold=0.1, top_m_active=2),
        observations_path=root / "observations.jsonl",
        camera_keys=["observation.images.front"],
    )

    raw_batch = {
        "episode_index": torch.tensor([0, 0, 1]),
        "frame_index": torch.tensor([10, 11, 2]),
        "task": ["pick up cup", "pick up cup", "open drawer"],
        "observation.images.front": torch.zeros(3, 3, 16, 16),
    }
    collector.begin_batch(raw_batch)
    z0 = torch.rand(3, 5, 8)
    z1 = torch.rand(3, 5, 8)
    z0[0, 2, 3] = 10.0
    z1[2, 1, 6] = 9.0
    collector.observe_latent(layer_names[0], 0, z0, torch.tensor([0.2, 0.3, 0.2]))
    collector.observe_latent(layer_names[1], 1, z1, torch.tensor([0.2, 0.3, 0.2]))
    collector.end_batch()
    collector.save(root)
    collector.close()

    script = Path(__file__).resolve().parent / "make_pi05_feature_report.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--feature-dir",
            str(root),
            "--max-features",
            "5",
            "--top-examples",
            "2",
            "--sort-by",
            "interesting",
        ],
        check=True,
    )

    assert (root / "feature_report.html").exists()
    assert (root / "feature_candidates.csv").exists()
    payload = json.loads((root / "feature_candidates.json").read_text())
    assert payload["candidates"]
    assert {"feature_key", "rank_score", "frequency"} <= set(payload["candidates"][0])
    print(f"feature report smoke test passed: {root}")


if __name__ == "__main__":
    main()

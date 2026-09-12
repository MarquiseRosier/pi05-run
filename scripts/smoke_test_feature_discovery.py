#!/usr/bin/env python
"""Fast smoke test for Pi0.5 transcoder feature-discovery summaries."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from pi05_mi.feature_discovery import FeatureDiscoveryCollector, FeatureDiscoveryConfig


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="pi05_feature_discovery_"))
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
        "episode_index": torch.tensor([0, 0]),
        "frame_index": torch.tensor([10, 11]),
        "task": ["pick up cup", "pick up cup"],
        "observation.images.front": torch.zeros(2, 3, 16, 16),
    }
    collector.begin_batch(raw_batch)
    collector.observe_latent(layer_names[0], 0, torch.rand(2, 5, 8), torch.tensor([0.2, 0.3]))
    collector.observe_latent(layer_names[1], 1, torch.rand(2, 5, 8), torch.tensor([0.2, 0.3]))
    updated = collector.end_batch()
    collector.save(root)
    collector.close()

    topk = torch.load(root / "feature_topk.pt", map_location="cpu", weights_only=False)
    stats = torch.load(root / "feature_stats.pt", map_location="cpu", weights_only=False)
    assert updated == {layer_names[0]: 2, layer_names[1]: 2}
    assert topk["format_version"] == 2
    assert sorted(topk["topk"][layer_names[0]]) == ["0.20000000", "0.30000001"]
    assert topk["topk"][layer_names[0]]["0.20000000"]["scores"].shape == (8, 3)
    assert stats["stats"][layer_names[0]]["0.20000000"]["mean"].shape == (8,)
    assert stats["stats"][layer_names[0]]["0.20000000"]["count"] == 1
    print(f"feature discovery smoke test passed: {root}")


if __name__ == "__main__":
    main()

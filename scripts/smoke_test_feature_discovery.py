#!/usr/bin/env python
"""Fast smoke test for Pi0.5 transcoder feature-discovery summaries."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from pi05_mi.feature_discovery import FeatureDiscoveryCollector, FeatureDiscoveryConfig, TokenSparsity


def check_token_sparsity() -> None:
    """Per-token L0 must be exact, and must not be the max-over-tokens bound.

    The distinction is the reason the accumulator exists: collapsing a code by a
    maximum over tokens counts a feature that fired anywhere, which overstates
    L0 by up to the token count when different tokens use different features.
    """
    store = TokenSparsity(d_features=8)
    # Four tokens with 1, 2, 3 and 4 active features.
    store.update(torch.tensor([[1, 2, 3, 4]]))
    state = store.state_dict()
    assert state["tokens"] == 4 and state["mean"] == 2.5
    assert state["median"] == 2.0 and state["min"] == 1.0 and state["max"] == 4.0
    assert state["p90"] == 4.0
    assert int(state["histogram"].sum()) == 4

    # Quantiles stay exact as the sample grows, because the histogram is exact.
    big = TokenSparsity(d_features=100)
    big.update(torch.arange(101))
    assert big.state_dict()["median"] == 50.0
    assert big.state_dict()["max"] == 100.0

    # Out-of-range values are clamped rather than raising.
    edge = TokenSparsity(d_features=4)
    edge.update(torch.tensor([9, -1]))
    assert edge.state_dict()["max"] == 4.0 and edge.state_dict()["min"] == 0.0

    # An empty store reports nan rather than dividing by zero.
    import math
    assert math.isnan(TokenSparsity(d_features=4).state_dict()["mean"])

    # The headline property: L0 per token is below the max-over-token count
    # whenever tokens use different features.
    disjoint = torch.zeros(1, 3, 9)
    disjoint[0, 0, 0:3] = 1.0
    disjoint[0, 1, 3:6] = 1.0
    disjoint[0, 2, 6:9] = 1.0
    per_token = (disjoint > 0).sum(dim=-1)
    max_over_tokens = int((disjoint.max(dim=1).values > 0).sum())
    assert per_token.float().mean() == 3.0 and max_over_tokens == 9, (
        "three active per token, but nine features fired somewhere"
    )


def main() -> None:
    check_token_sparsity()
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

    sparsity = torch.load(root / "token_sparsity.pt", map_location="cpu", weights_only=False)
    entry = sparsity["token_sparsity"][layer_names[0]]["0.20000000"]
    # One observation at this flow time, five action tokens.
    assert entry["tokens"] == 5, entry["tokens"]
    assert 0.0 <= entry["mean"] <= 8.0 and entry["max"] <= 8.0
    assert int(entry["histogram"].sum()) == entry["tokens"]
    print(f"feature discovery smoke test passed: {root}")


if __name__ == "__main__":
    main()

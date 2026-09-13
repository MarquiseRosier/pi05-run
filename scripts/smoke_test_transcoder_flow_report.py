#!/usr/bin/env python
"""Fast smoke test for the Pi0.5 transcoder layer-flow report."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from pi05_mi.feature_discovery import FeatureDiscoveryCollector, FeatureDiscoveryConfig


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="pi05_transcoder_flow_"))
    layer_names = [
        "paligemma_with_expert.gemma_expert.model.layers.0.mlp",
        "paligemma_with_expert.gemma_expert.model.layers.1.mlp",
        "paligemma_with_expert.gemma_expert.model.layers.2.mlp",
    ]
    collector = FeatureDiscoveryCollector(
        layer_names=layer_names,
        layer_indices={name: i for i, name in enumerate(layer_names)},
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
    for layer_index, name in enumerate(layer_names):
        latent = torch.rand(3, 5, 8)
        latent[:, :, layer_index + 2] += float(layer_index + 1)
        collector.observe_latent(name, layer_index, latent, torch.tensor([0.2, 0.3, 0.2]))
    collector.end_batch()
    collector.save(root, extra_config={"checkpoint": str(root / "fake_checkpoint.pt")})
    collector.close()

    checkpoint = {
        "state_dicts": {
            name: {"decoder.weight": torch.ones(4, 8) * (layer_index + 1)}
            for layer_index, name in enumerate(layer_names)
        }
    }
    torch.save(checkpoint, root / "fake_checkpoint.pt")

    script = Path(__file__).resolve().parent / "make_pi05_transcoder_flow_report.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--feature-dir",
            str(root),
            "--top-features-per-layer",
            "3",
        ],
        check=True,
    )

    assert (root / "transcoder_flow_report.html").exists()
    assert (root / "transcoder_flow_chart.svg").exists()
    assert (root / "transcoder_flow_layers.csv").exists()
    payload = json.loads((root / "transcoder_flow_summary.json").read_text())
    assert len(payload["layers"]) == 3
    assert payload["attribution_metric"] == "decoder_weighted_mean_z"
    assert payload["layers"][0]["top_features"]
    print(f"transcoder flow report smoke test passed: {root}")


if __name__ == "__main__":
    main()

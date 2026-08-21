#!/usr/bin/env python
"""Smoke test for Pi0.5 transcoder activation buffers."""

from __future__ import annotations

import torch

from pi05_mi.buffers import ActivationBuffer, MultiLayerActivationBuffer, flatten_activation_record
from pi05_mi.patch_pi05 import MLPActivationRecord, Pi05TranscoderContext


def main() -> None:
    torch.manual_seed(0)
    d_model = 4
    x = torch.randn(2, 3, d_model)
    y = torch.randn(2, 3, d_model)
    timestep = torch.tensor([0.25, 0.75])

    flat = flatten_activation_record(x, y, timestep)
    assert flat.x.shape == (6, d_model)
    assert flat.y.shape == (6, d_model)
    assert flat.timestep.tolist() == [0.25, 0.25, 0.25, 0.75, 0.75, 0.75]

    buffer = ActivationBuffer(d_model=d_model, capacity=5)
    added = buffer.add(x, y, timestep)
    assert added == 6
    assert buffer.size == 5
    assert buffer.variance.count == 6
    assert buffer.variance_denominator().item() > 0.0

    sample = buffer.sample(3, device="cpu", dtype=torch.float32)
    assert sample.x.shape == (3, d_model)
    assert sample.y.dtype == torch.float32
    assert sample.timestep.shape == (3,)

    full_buffer = ActivationBuffer(d_model=d_model, capacity=6)
    full_buffer.add(x, y, timestep)
    epoch_batches = list(full_buffer.epoch_batches(2, device="cpu", dtype=torch.float32, shuffle=False))
    assert len(epoch_batches) == 3
    assert sum(batch.x.shape[0] for batch in epoch_batches) == 6

    context = Pi05TranscoderContext()
    name = "paligemma_with_expert.gemma_expert.model.layers.0.mlp"
    context.records[name].append(
        MLPActivationRecord(
            name=name,
            layer_index=0,
            x=x,
            y=y,
            timestep=timestep,
        )
    )
    multilayer = MultiLayerActivationBuffer(layer_names=[name], d_model=d_model, capacity_per_layer=16)
    added_by_layer = multilayer.add_context_records(context)
    assert added_by_layer == {name: 6}
    assert context.records == {}
    assert multilayer.ready_layer_names(6) == [name]
    assert multilayer.stats()[name].size == 6

    print("transcoder buffer smoke test passed")


if __name__ == "__main__":
    main()

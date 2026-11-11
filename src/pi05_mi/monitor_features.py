"""Matched, frozen action-expert representations for monitoring (not replacement)."""
from __future__ import annotations

import hashlib
import math

import torch


class MatchedExpertTap:
    """Capture input, original output and sparse code at the same MLP calls.

    Average tokens and denoising calls identically. An optional fixed Gaussian
    projection controls head width without selecting coordinates using labels.
    A projected pilot is not evidence about every feature in the full dictionary.
    """

    def __init__(self, targets, *, layers, width=256):
        if width < 0:
            raise ValueError('comparison width must be >= 0; 0 retains native dimensions')
        self.layers = tuple(layers)
        self.width = width
        self.values, self.projections, self.schema, self.handles = {}, {}, {}, []
        selected = [t for t in targets if t.layer_index in self.layers]
        if {t.layer_index for t in selected} != set(self.layers):
            raise ValueError('Missing requested action-expert MLPs')
        for target in selected:
            self.handles.append(target.module.register_forward_hook(self._hook(target.layer_index)))

    def _record(self, family, layer, tensor):
        key = f'{family}_l{layer:02d}'
        pooled = tensor.detach().float().mean(dim=1)
        native_width = pooled.shape[-1]
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], 'little')
        if self.width and native_width > self.width:
            if key not in self.projections:
                generator = torch.Generator(device='cpu').manual_seed(seed)
                projection = torch.randn(native_width, self.width, generator=generator)
                self.projections[key] = (projection / math.sqrt(self.width)).to(pooled.device)
            pooled = pooled @ self.projections[key]
        self.schema[key] = dict(layer=layer, component='action_expert', native_width=native_width,
                                saved_width=pooled.shape[-1], projection_seed=seed,
                                projection='fixed_gaussian' if pooled.shape[-1] != native_width else 'none',
                                pooling='mean tokens, then mean denoising calls')
        self.values.setdefault(key, []).append(pooled)

    def _hook(self, layer):
        def hook(module, inputs, output):
            self._record('expert_input', layer, inputs[0])
            self._record('expert_output', layer, output)
        return hook

    def latent(self, name, layer, latent, timestep):
        if layer in self.layers:
            self._record('expert_transcoder', layer, latent)

    def reset(self):
        self.values.clear()

    def arrays(self):
        expected = {f'{family}_l{layer:02d}' for layer in self.layers
                    for family in ('expert_input', 'expert_output', 'expert_transcoder')}
        if set(self.values) != expected:
            raise RuntimeError(f'Missing matched representations: {sorted(expected - set(self.values))}')
        for layer in self.layers:
            counts = {len(self.values[f'{family}_l{layer:02d}'])
                      for family in ('expert_input', 'expert_output', 'expert_transcoder')}
            if len(counts) != 1:
                raise RuntimeError(f'Unmatched MLP and transcoder call counts at layer {layer}')
            for family in ('expert_input', 'expert_output', 'expert_transcoder'):
                self.schema[f'{family}_l{layer:02d}']['calls_per_forward'] = next(iter(counts))
        return {key: torch.stack(value).mean(0).cpu().numpy() for key, value in self.values.items()}

    def close(self):
        for handle in self.handles:
            handle.remove()

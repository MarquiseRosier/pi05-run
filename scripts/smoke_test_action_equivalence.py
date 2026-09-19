#!/usr/bin/env python
"""Smoke test for the paired transcoder action-equivalence measurement.

Covers the two things the real Colab run depends on and cannot cheaply re-check:

1. the error metrics reported in ``action_equivalence_summary.json`` are the
   quantities they claim to be, and
2. the baseline/replace pairing is sound -- baseline mode reproduces the
   original MLP bit-for-bit (so the determinism control floor is zero) while
   replace mode actually routes through the transcoder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pi05_mi.patch_pi05 import Pi05TranscoderContext, install_pi05_action_expert_wrappers
from pi05_mi.transcoders import TimeConditionedTranscoder, TimeConditionedTranscoderConfig

from eval_pi05_transcoder_action_equivalence import (
    _aggregate,
    _compare_one,
    _relative_l2,
    _shared_noise,
)


def _row(original: np.ndarray, replace: np.ndarray, *, n_action_steps: int = 2) -> dict:
    return _compare_one(
        original=original,
        replace=replace,
        batch_index=1,
        item_index=0,
        n_action_steps=n_action_steps,
        metadata={},
    )


def test_identical_chunks_report_zero_error() -> None:
    actions = np.array([[1.0, -2.0], [0.5, 0.25], [3.0, 1.0], [0.1, 0.2]], dtype=np.float64)
    row = _row(actions, actions.copy())
    assert row["rmse"] == 0.0
    assert row["mae"] == 0.0
    assert row["max_abs"] == 0.0
    assert row["rel_l2"] == 0.0
    assert row["executed_rel_l2"] == 0.0
    assert abs(row["cosine"] - 1.0) < 1e-12
    assert abs(row["norm_delta_mean"]) < 1e-12


def test_constant_offset_matches_hand_computed_metrics() -> None:
    original = np.ones((4, 2), dtype=np.float64)
    replace = original + 0.1
    row = _row(original, replace)
    assert abs(row["rmse"] - 0.1) < 1e-12
    assert abs(row["mae"] - 0.1) < 1e-12
    assert abs(row["max_abs"] - 0.1) < 1e-12
    # ||0.1 * 1|| / ||1|| == 0.1 regardless of how many elements there are.
    assert abs(row["rel_l2"] - 0.1) < 1e-12
    assert abs(row["executed_rel_l2"] - 0.1) < 1e-12
    assert abs(row["original_rms"] - 1.0) < 1e-12
    # Every per-dim RMSE is the same constant offset.
    assert abs(row["dim_0_rmse"] - 0.1) < 1e-12
    assert abs(row["dim_1_rmse"] - 0.1) < 1e-12


def test_executed_window_ignores_steps_beyond_n_action_steps() -> None:
    original = np.ones((4, 2), dtype=np.float64)
    replace = original.copy()
    replace[2:] += 5.0  # only the un-executed tail diverges
    row = _row(original, replace, n_action_steps=2)
    assert row["executed_rmse"] == 0.0
    assert row["executed_rel_l2"] == 0.0
    assert row["rmse"] > 0.0, "full-chunk RMSE must still see the tail"


def test_relative_l2_is_none_for_a_zero_reference() -> None:
    zeros = np.zeros((3, 2), dtype=np.float64)
    assert _relative_l2(zeros, np.ones((3, 2))) is None
    row = _row(zeros, np.ones((3, 2)))
    assert row["rel_l2"] is None
    assert row["cosine"] is None, "a zero reference has no defined direction"


def test_aggregate_tolerates_missing_and_none_metrics() -> None:
    rows = [
        _row(np.ones((4, 2)), np.ones((4, 2)) + 0.1),
        _row(np.zeros((4, 2)), np.ones((4, 2))),  # rel_l2/cosine are None here
    ]
    summary = _aggregate(rows)
    assert summary["pairs"] == 2
    assert summary["rel_l2"]["n"] == 1, "the undefined relative error must be dropped, not counted as 0"
    assert summary["rmse"]["n"] == 2
    # The one defined relative error is ~0.1, so it clears 0.25 and misses 0.05.
    assert summary["executed_rel_l2_le_0.25"] == 1.0
    assert summary["executed_rel_l2_le_0.05"] == 0.0
    assert summary["per_dim_rmse"]["dim_0_rmse"]["n"] == 2


class _FakeMLP(nn.Module):
    """Stand-in for one Gemma action-expert MLP."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.down_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(x)


class _FakeLayer(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.mlp = _FakeMLP(d_model)


class _FakeExpertModel(nn.Module):
    def __init__(self, d_model: int, n_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_FakeLayer(d_model) for _ in range(n_layers))


class _FakeExpert(nn.Module):
    def __init__(self, d_model: int, n_layers: int) -> None:
        super().__init__()
        self.model = _FakeExpertModel(d_model, n_layers)


class _FakePaligemmaWithExpert(nn.Module):
    def __init__(self, d_model: int, n_layers: int) -> None:
        super().__init__()
        self.gemma_expert = _FakeExpert(d_model, n_layers)


class _FakePi05Model(nn.Module):
    """Minimal module tree matching ``ACTION_EXPERT_MLP_RE`` and the patched methods."""

    def __init__(self, d_model: int = 8, n_layers: int = 2) -> None:
        super().__init__()
        self.paligemma_with_expert = _FakePaligemmaWithExpert(d_model, n_layers)

    def _mlps(self) -> list[nn.Module]:
        return [layer.mlp for layer in self.paligemma_with_expert.gemma_expert.model.layers]

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for mlp in self._mlps():
            x = mlp(x)
        return x

    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
        return self.forward(x_t, time=timestep)

    def sample_noise(self, shape, device):
        return torch.zeros(shape, device=device)


def _build_wrapped_model() -> tuple[_FakePi05Model, Pi05TranscoderContext, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    d_model = 8
    model = _FakePi05Model(d_model=d_model, n_layers=2).eval()

    transcoders = {}
    for name, _module in model.named_modules():
        if not name.endswith(".mlp"):
            continue
        config = TimeConditionedTranscoderConfig(d_model=d_model, expansion_factor=2)
        transcoder = TimeConditionedTranscoder(config).eval()
        for parameter in transcoder.parameters():
            parameter.requires_grad_(False)
        transcoders[name] = transcoder

    context = Pi05TranscoderContext(
        mode="train",
        capture_records=False,
        capture_latents=False,
        store_latent_summaries=False,
    )
    context, wrapped = install_pi05_action_expert_wrappers(
        model, context=context, transcoders=transcoders, mode="train"
    )
    assert len(wrapped) == 2, f"expected both action-expert MLPs to be wrapped, got {wrapped}"

    x = torch.randn(3, 4, d_model)
    timestep = torch.full((3,), 0.4)
    return model, context, x, timestep


def test_baseline_modes_preserve_the_original_output_and_replace_does_not() -> None:
    model, context, x, timestep = _build_wrapped_model()

    # Reference: bypass the wrappers entirely.
    with torch.no_grad():
        reference = x
        for layer in model.paligemma_with_expert.gemma_expert.model.layers:
            reference = layer.mlp.original_mlp(reference)

    outputs = {}
    for mode in ("train", "probe", "replace"):
        context.mode = mode
        with torch.no_grad():
            outputs[mode] = model.denoise_step(None, None, x, timestep)

    assert torch.equal(outputs["train"], reference), "train mode must return the original MLP output"
    assert torch.equal(outputs["probe"], reference), (
        "probe mode runs the transcoder but must discard it, so it must match the original exactly"
    )
    assert not torch.allclose(outputs["replace"], reference), (
        "replace mode must actually route through the transcoder"
    )


def test_baseline_repeats_are_bit_identical_so_the_control_floor_is_zero() -> None:
    model, context, x, timestep = _build_wrapped_model()
    context.mode = "train"
    with torch.no_grad():
        first = model.denoise_step(None, None, x, timestep)
        second = model.denoise_step(None, None, x, timestep)
    assert torch.equal(first, second)

    row = _row(first[0].numpy().astype(np.float64), second[0].numpy().astype(np.float64), n_action_steps=4)
    assert row["executed_rmse"] == 0.0
    assert row["rel_l2"] == 0.0


class _FakeConfig:
    chunk_size = 50
    max_action_dim = 32


class _FakePolicyForNoise:
    class model:  # noqa: N801 - mirrors ``policy.model.config`` access
        config = _FakeConfig()


def test_lerobot_still_lets_us_inject_the_noise() -> None:
    """``predict_action_chunk`` must forward ``noise``/``num_steps`` to ``sample_actions``.

    If a lerobot upgrade drops this, the paired comparison would silently fall
    back to independently sampled noise and the reported error would be
    dominated by sampling, not by the transcoder substitution.
    """
    import inspect

    from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch

    chunk_params = inspect.signature(PI05Policy.predict_action_chunk).parameters
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in chunk_params.values()), (
        "predict_action_chunk no longer forwards **kwargs"
    )
    sample_params = inspect.signature(PI05Pytorch.sample_actions).parameters
    assert "noise" in sample_params, "sample_actions no longer accepts an explicit noise tensor"
    assert "num_steps" in sample_params, "sample_actions no longer accepts num_steps"


def test_shared_noise_matches_the_flow_matching_contract() -> None:
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS

    batch = {OBS_LANGUAGE_TOKENS: torch.zeros(3, 7, dtype=torch.long)}
    noise = _shared_noise(_FakePolicyForNoise(), batch, torch.device("cpu"))
    # lerobot's sample_noise draws float32 (batch, chunk_size, max_action_dim).
    assert noise.shape == (3, 50, 32)
    assert noise.dtype == torch.float32


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok {test.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    main()

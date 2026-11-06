"""Validate same-call capture and noninterference without a downloaded VLA."""
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pi05_mi.monitor_features import MatchedExpertTap
from pi05_mi.patch_pi05 import WrappedActionExpertMLP, Pi05TranscoderContext
from pi05_mi.transcoders import TimeConditionedTranscoder, TimeConditionedTranscoderConfig


def test_wrapped_probe_captures_same_calls_without_changing_output():
    torch.manual_seed(12)
    mlp = torch.nn.Linear(8, 8)
    target = SimpleNamespace(module=mlp, layer_index=0)
    tap = MatchedExpertTap([target], layers=(0,), width=0)
    context = Pi05TranscoderContext(mode='probe', capture_records=False,
                                   capture_latents=True, store_latent_summaries=False,
                                   latent_callback=tap.latent)
    transcoder = TimeConditionedTranscoder(TimeConditionedTranscoderConfig(d_model=8, d_features=16))
    wrapped = WrappedActionExpertMLP(name='test', layer_index=0, original_mlp=mlp,
                                    context=context, transcoder=transcoder)
    x = torch.randn(2, 5, 8)
    context.current_timestep = torch.tensor([0.2, 0.8])
    expected = torch.nn.functional.linear(x, mlp.weight, mlp.bias)
    with torch.inference_mode():
        actual = wrapped(x)
    assert torch.equal(actual, expected)
    arrays = tap.arrays()
    torch.testing.assert_close(torch.from_numpy(arrays['expert_input_l00']), x.mean(1))
    torch.testing.assert_close(torch.from_numpy(arrays['expert_output_l00']), expected.mean(1))
    assert arrays['expert_transcoder_l00'].shape == (2, 16)
    tap.reset()
    with pytest.raises(RuntimeError, match='Missing matched'):
        tap.arrays()
    tap.close()


def test_projection_is_reproducible_and_call_mismatch_fails():
    x = torch.randn(2, 5, 8)
    taps = [MatchedExpertTap([SimpleNamespace(module=torch.nn.Identity(), layer_index=0)],
                            layers=(0,), width=4) for _ in range(2)]
    for tap in taps:
        for family in ('expert_input', 'expert_output', 'expert_transcoder'):
            tap._record(family, 0, x)
    a, b = [tap.arrays() for tap in taps]
    for key in a:
        assert a[key].shape == (2, 4)
        torch.testing.assert_close(torch.from_numpy(a[key]), torch.from_numpy(b[key]))
    taps[0]._record('expert_input', 0, x)
    with pytest.raises(RuntimeError, match='Unmatched'):
        taps[0].arrays()
    for tap in taps:
        tap.close()

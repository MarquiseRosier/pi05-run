#!/usr/bin/env python
"""Smoke test for causal feature patching.

The interchange intervention is only interpretable if the mechanism is exact:
donating every feature must reproduce the donor's output, an empty patch must
change nothing, and a partial patch must land strictly between the endpoints.
These check all three against a real wrapped model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from patch_pi05_transcoder_features import FeaturePatcher, random_like, recovery  # noqa: E402
from smoke_test_action_equivalence import _build_wrapped_model  # noqa: E402


def _fixture():
    model, context, x_a, timestep = _build_wrapped_model()
    torch.manual_seed(11)
    x_b = torch.randn_like(x_a)
    layers = [name for name, _ in model.named_modules() if name.endswith(".mlp")]
    latent_dim = model.paligemma_with_expert.gemma_expert.model.layers[0].mlp.transcoder.config.latent_dim
    return model, context, x_a, x_b, timestep, layers, latent_dim


def _run(model, context, patchers, mode, x, timestep):
    context.mode = "replace"
    for item in patchers:
        item.begin(mode)
    if patchers:
        def dispatch(name, index, latent, t):
            for item in patchers:
                result = item(name, index, latent, t)
                if result is not None:
                    return result
            return None

        context.latent_intervention = dispatch
    else:
        context.latent_intervention = None
    with torch.no_grad():
        return model.denoise_step(None, None, x, timestep)


def test_full_donation_reproduces_the_donor_exactly() -> None:
    model, context, x_a, x_b, timestep, layers, latent_dim = _fixture()
    reference_b = _run(model, context, [], "off", x_b, timestep)
    patcher = FeaturePatcher({layer: list(range(latent_dim)) for layer in layers})
    _run(model, context, [patcher], "record", x_b, timestep)
    patched = _run(model, context, [patcher], "apply", x_a, timestep)
    assert torch.allclose(patched, reference_b, atol=1e-5), (
        "donating every feature must reproduce the donor output; if it does not, "
        "the features do not mediate the layer and partial patches are uninterpretable"
    )


def test_empty_patch_is_an_exact_no_op() -> None:
    model, context, x_a, x_b, timestep, _layers, _dim = _fixture()
    reference_a = _run(model, context, [], "off", x_a, timestep)
    patcher = FeaturePatcher({})
    _run(model, context, [patcher], "record", x_b, timestep)
    assert torch.equal(_run(model, context, [patcher], "apply", x_a, timestep), reference_a)


def test_partial_patch_lands_between_the_endpoints() -> None:
    model, context, x_a, x_b, timestep, layers, latent_dim = _fixture()
    reference_a = _run(model, context, [], "off", x_a, timestep)
    reference_b = _run(model, context, [], "off", x_b, timestep)
    patcher = FeaturePatcher({layer: list(range(latent_dim // 2)) for layer in layers})
    _run(model, context, [patcher], "record", x_b, timestep)
    patched = _run(model, context, [patcher], "apply", x_a, timestep)
    fraction = recovery(patched.numpy().ravel(), reference_b.numpy().ravel(), reference_a.numpy().ravel())
    assert 0.0 < fraction < 1.0, f"a half patch should partially recover, got {fraction}"


def test_zero_ablation_changes_the_output_and_reports_itself() -> None:
    model, context, x_a, _x_b, timestep, layers, _dim = _fixture()
    reference_a = _run(model, context, [], "off", x_a, timestep)
    patcher = FeaturePatcher({layers[0]: [0, 1, 2, 3]})
    ablated = _run(model, context, [patcher], "zero", x_a, timestep)
    assert not torch.equal(ablated, reference_a)
    assert patcher.applied > 0, "the patcher must report how many times it fired"


def test_recovery_metric_endpoints() -> None:
    import numpy as np

    away = np.zeros(4)
    toward = np.ones(4)
    assert abs(recovery(toward, toward, away) - 1.0) < 1e-12, "reaching the donor is full recovery"
    assert abs(recovery(away, toward, away) - 0.0) < 1e-12, "not moving is zero recovery"
    midpoint = 0.5 * (toward + away)
    assert abs(recovery(midpoint, toward, away) - 0.5) < 1e-12


def test_random_control_matches_the_candidate_counts_per_layer() -> None:
    reference = {"a.mlp": [1, 2, 3], "b.mlp": [7]}
    generator = torch.Generator().manual_seed(0)
    control = random_like(reference, n_features_total=500, generator=generator)
    assert {k: len(v) for k, v in control.items()} == {k: len(v) for k, v in reference.items()}
    for ids in control.values():
        assert len(set(ids)) == len(ids), "random control must not repeat a feature"
        assert all(0 <= i < 500 for i in ids)


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok {test.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    main()

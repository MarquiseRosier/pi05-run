#!/usr/bin/env python
"""Smoke test for the transcoder latent intervention hook.

The hook is what turns a correlational finding ("these features differ between
two images") into a causal one ("substituting these features changes the
behavior"). These checks pin the properties that claim depends on:

* an identity patch and an opt-out must be exact no-ops, so a null intervention
  cannot masquerade as an effect,
* patching a single feature must actually move the output, and
* a wrong-shaped patch must raise rather than silently corrupt the pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from smoke_test_action_equivalence import _build_wrapped_model  # noqa: E402


def _reference():
    model, context, x, timestep = _build_wrapped_model()
    context.mode = "replace"
    context.latent_intervention = None
    with torch.no_grad():
        ref = model.denoise_step(None, None, x, timestep)
    return model, context, x, timestep, ref


def _run(model, x, timestep):
    with torch.no_grad():
        return model.denoise_step(None, None, x, timestep)


def test_identity_intervention_is_an_exact_no_op() -> None:
    model, context, x, timestep, ref = _reference()
    context.latent_intervention = lambda name, index, latent, t: latent
    assert torch.equal(_run(model, x, timestep), ref)


def test_returning_none_opts_out_per_layer() -> None:
    model, context, x, timestep, ref = _reference()
    context.latent_intervention = lambda name, index, latent, t: None
    assert torch.equal(_run(model, x, timestep), ref)


def test_intervention_only_fires_in_replace_mode() -> None:
    """Probe mode must stay a pure observation path even with a hook installed."""
    model, context, x, timestep, _ref = _reference()
    calls = []
    context.latent_intervention = lambda name, index, latent, t: calls.append(name) or None

    context.mode = "probe"
    probe_out = _run(model, x, timestep)
    assert calls == [], "the intervention must not run while merely probing"

    with torch.no_grad():
        expected = x
        for layer in model.paligemma_with_expert.gemma_expert.model.layers:
            expected = layer.mlp.original_mlp(expected)
    assert torch.equal(probe_out, expected)

    context.mode = "replace"
    _run(model, x, timestep)
    assert len(calls) == 2, f"expected one call per wrapped layer, got {calls}"


def test_zeroing_one_feature_changes_the_output() -> None:
    model, context, x, timestep, ref = _reference()

    captured: dict[str, torch.Tensor] = {}

    def capture(name, index, latent, t):
        captured[name] = latent.detach().clone()
        return None

    context.latent_intervention = capture
    _run(model, x, timestep)

    layer = sorted(captured)[0]
    feature = int(captured[layer].abs().sum(dim=(0, 1)).argmax())

    def patch(name, index, latent, t):
        if name != layer:
            return None
        out = latent.clone()
        out[..., feature] = 0.0
        return out

    context.latent_intervention = patch
    patched = _run(model, x, timestep)
    assert not torch.equal(patched, ref), "ablating an active feature must change the output"


def test_patching_an_inactive_feature_is_a_no_op() -> None:
    """Specificity: zeroing an already-zero ReLU feature must change nothing."""
    model, context, x, timestep, ref = _reference()

    captured: dict[str, torch.Tensor] = {}

    def capture(name, index, latent, t):
        captured[name] = latent.detach().clone()
        return None

    context.latent_intervention = capture
    _run(model, x, timestep)

    layer = sorted(captured)[0]
    totals = captured[layer].abs().sum(dim=(0, 1))
    inactive = (totals == 0).nonzero().flatten()
    if inactive.numel() == 0:
        return  # no dead feature in this tiny fixture; nothing to assert

    feature = int(inactive[0])

    def patch(name, index, latent, t):
        if name != layer:
            return None
        out = latent.clone()
        out[..., feature] = 0.0
        return out

    context.latent_intervention = patch
    assert torch.equal(_run(model, x, timestep), ref)


def test_wrong_shaped_patch_raises() -> None:
    model, context, x, timestep, _ref = _reference()
    context.latent_intervention = lambda name, index, latent, t: latent[..., :-1]
    try:
        _run(model, x, timestep)
    except ValueError as exc:
        assert "expected" in str(exc)
    else:
        raise AssertionError("a wrong-shaped patch must raise, not silently corrupt the pass")


def test_interchange_patch_reproduces_the_donor_output() -> None:
    """The core causal test: donating every latent must reproduce the donor's output.

    If substituting *all* transcoder features from input B into a pass on input
    A does not reproduce B's output, then the features do not fully mediate the
    layer and a partial patch cannot be interpreted cleanly either.
    """
    model, context, x_a, timestep, _ = _reference()
    torch.manual_seed(7)
    x_b = torch.randn_like(x_a)

    donor: dict[tuple[str, int], torch.Tensor] = {}
    step = {"i": 0}

    def capture(name, index, latent, t):
        donor[(name, step["i"])] = latent.detach().clone()
        step["i"] += 1
        return None

    context.latent_intervention = capture
    with torch.no_grad():
        out_b = model.denoise_step(None, None, x_b, timestep)

    step["i"] = 0

    def donate(name, index, latent, t):
        key = (name, step["i"])
        step["i"] += 1
        return donor[key]

    context.latent_intervention = donate
    with torch.no_grad():
        patched = model.denoise_step(None, None, x_a, timestep)

    assert torch.allclose(patched, out_b, atol=1e-5), (
        "donating all latents should reproduce the donor output; "
        f"max diff {(patched - out_b).abs().max().item():.3g}"
    )


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok {test.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    main()

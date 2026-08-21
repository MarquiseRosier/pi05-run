"""Runtime wrappers for Pi0.5 action-expert MLP transcoders."""

from __future__ import annotations

import re
import types
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Literal

import torch
from torch import Tensor, nn

from .transcoders import TimeConditionedTranscoder


TranscoderMode = Literal["train", "probe", "replace"]

ACTION_EXPERT_MLP_RE = re.compile(r"paligemma_with_expert\.gemma_expert\.model\.layers\.(\d+)\.mlp$")


@dataclass(frozen=True)
class MLPActivationRecord:
    """One captured MLP input-output tensor for one layer call."""

    name: str
    layer_index: int
    x: Tensor
    y: Tensor
    timestep: Tensor


@dataclass(frozen=True)
class ActionExpertMLPTarget:
    """One Pi0.5 action-expert MLP target."""

    name: str
    layer_index: int
    module: nn.Module
    d_model: int


class Pi05TranscoderContext:
    """Shared runtime state for Pi0.5 transcoder wrappers."""

    def __init__(self, mode: TranscoderMode = "train", *, detach_records: bool = True):
        self.mode = mode
        self.detach_records = detach_records
        self.current_timestep: Tensor | None = None
        self.records: dict[str, list[MLPActivationRecord]] = defaultdict(list)

    @contextmanager
    def use_timestep(self, timestep: Tensor) -> Iterator[None]:
        previous = self.current_timestep
        self.current_timestep = timestep
        try:
            yield
        finally:
            self.current_timestep = previous

    def clear_records(self) -> None:
        self.records.clear()

    def timestep_for(self, x: Tensor) -> Tensor:
        if self.current_timestep is None:
            raise RuntimeError(
                "No Pi0.5 timestep is active. Call the wrapped model through PI05Pytorch.forward "
                "or PI05Pytorch.denoise_step so the transcoder context can see t."
            )
        timestep = self.current_timestep.to(device=x.device)
        if timestep.ndim == 0:
            timestep = timestep.reshape(1)
        if x.ndim == 3 and timestep.shape[0] != x.shape[0]:
            if timestep.numel() == 1:
                timestep = timestep.expand(x.shape[0])
            else:
                raise ValueError(f"Expected timestep batch {x.shape[0]}, got shape {tuple(timestep.shape)}")
        if x.ndim == 2 and timestep.shape[0] != x.shape[0]:
            if timestep.numel() == 1:
                timestep = timestep.expand(x.shape[0])
            else:
                raise ValueError(f"Expected one timestep per record for x shape {tuple(x.shape)}, got {tuple(timestep.shape)}")
        return timestep

    def record(self, name: str, layer_index: int, x: Tensor, y: Tensor, timestep: Tensor) -> None:
        if self.detach_records:
            x = x.detach()
            y = y.detach()
            timestep = timestep.detach()
        self.records[name].append(
            MLPActivationRecord(
                name=name,
                layer_index=layer_index,
                x=x,
                y=y,
                timestep=timestep,
            )
        )


class WrappedActionExpertMLP(nn.Module):
    """Wrapper that preserves, observes, or replaces one Pi0.5 action-expert MLP."""

    def __init__(
        self,
        *,
        name: str,
        layer_index: int,
        original_mlp: nn.Module,
        context: Pi05TranscoderContext,
        transcoder: TimeConditionedTranscoder | None = None,
    ):
        super().__init__()
        self.name = name
        self.layer_index = layer_index
        self.original_mlp = original_mlp
        self.transcoder = transcoder
        self.context = context

    def forward(self, x: Tensor) -> Tensor:
        timestep = self.context.timestep_for(x)

        if self.context.mode == "replace":
            if self.transcoder is None:
                raise RuntimeError(f"Cannot run {self.name} in replace mode without a transcoder")
            y_hat, _ = self.transcoder(x, timestep)
            return y_hat.to(dtype=x.dtype)

        y = self.original_mlp(x)
        self.context.record(self.name, self.layer_index, x, y, timestep)

        if self.context.mode == "probe" and self.transcoder is not None:
            with torch.no_grad():
                self.transcoder(x, timestep)

        return y


def _core_pi05_model(policy_or_model: nn.Module) -> nn.Module:
    return getattr(policy_or_model, "model", policy_or_model)


def _module_parent(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _target_mlps(model: nn.Module) -> list[tuple[str, int, nn.Module]]:
    targets: list[tuple[str, int, nn.Module]] = []
    for name, module in model.named_modules():
        if isinstance(module, WrappedActionExpertMLP):
            continue
        match = ACTION_EXPERT_MLP_RE.fullmatch(name)
        if match is None:
            continue
        targets.append((name, int(match.group(1)), module))
    return targets


def infer_mlp_d_model(module: nn.Module) -> int:
    """Infer the hidden width of a Gemma-style MLP module."""
    for attr in ("down_proj", "up_proj", "gate_proj"):
        projection = getattr(module, attr, None)
        if isinstance(projection, nn.Linear):
            if attr == "down_proj":
                return projection.out_features
            return projection.in_features
    raise ValueError(f"Could not infer d_model for MLP module {module.__class__.__name__}")


def list_pi05_action_expert_mlp_targets(policy_or_model: nn.Module) -> list[ActionExpertMLPTarget]:
    """List action-expert MLPs targeted by DifFRACT-style transcoders."""
    model = _core_pi05_model(policy_or_model)
    return [
        ActionExpertMLPTarget(
            name=name,
            layer_index=layer_index,
            module=module,
            d_model=infer_mlp_d_model(module),
        )
        for name, layer_index, module in _target_mlps(model)
    ]


def _patch_timestep_methods(model: nn.Module, context: Pi05TranscoderContext) -> None:
    if not hasattr(model, "_pi05_mi_original_forward"):
        model._pi05_mi_original_forward = model.forward  # type: ignore[attr-defined]
    if not hasattr(model, "_pi05_mi_original_denoise_step"):
        model._pi05_mi_original_denoise_step = model.denoise_step  # type: ignore[attr-defined]

    original_forward = model._pi05_mi_original_forward  # type: ignore[attr-defined]
    original_denoise_step = model._pi05_mi_original_denoise_step  # type: ignore[attr-defined]

    def forward_with_timestep(_self: nn.Module, *args, **kwargs):
        timestep = kwargs.get("time")
        if timestep is None:
            if len(args) < 7:
                raise TypeError("PI05Pytorch.forward wrapper could not find positional argument `time`")
            timestep = args[6]
        with context.use_timestep(timestep):
            return original_forward(*args, **kwargs)

    def denoise_step_with_timestep(_self: nn.Module, *args, **kwargs):
        timestep = kwargs.get("timestep")
        if timestep is None:
            if len(args) < 4:
                raise TypeError("PI05Pytorch.denoise_step wrapper could not find positional argument `timestep`")
            timestep = args[3]
        with context.use_timestep(timestep):
            return original_denoise_step(*args, **kwargs)

    model.forward = types.MethodType(forward_with_timestep, model)  # type: ignore[method-assign]
    model.denoise_step = types.MethodType(denoise_step_with_timestep, model)  # type: ignore[method-assign]
    model._pi05_mi_timestep_context = context  # type: ignore[attr-defined]


def install_pi05_action_expert_wrappers(
    policy_or_model: nn.Module,
    *,
    context: Pi05TranscoderContext | None = None,
    transcoders: dict[str, TimeConditionedTranscoder] | None = None,
    mode: TranscoderMode = "train",
) -> tuple[Pi05TranscoderContext, list[str]]:
    """Replace Pi0.5 action-expert MLP modules with wrapper modules.

    Args:
        policy_or_model: Either a ``PI05Policy`` or its underlying ``PI05Pytorch`` model.
        context: Optional shared context. A new one is created when omitted.
        transcoders: Optional mapping from target MLP module name to transcoder.
        mode: Initial context mode.

    Returns:
        The shared context and the list of wrapped module names.
    """
    model = _core_pi05_model(policy_or_model)
    if context is None:
        context = Pi05TranscoderContext(mode=mode)
    else:
        context.mode = mode
    transcoders = transcoders or {}

    wrapped_names: list[str] = []
    for name, layer_index, module in _target_mlps(model):
        parent, attr = _module_parent(model, name)
        setattr(
            parent,
            attr,
            WrappedActionExpertMLP(
                name=name,
                layer_index=layer_index,
                original_mlp=module,
                context=context,
                transcoder=transcoders.get(name),
            ),
        )
        wrapped_names.append(name)

    _patch_timestep_methods(model, context)
    return context, wrapped_names

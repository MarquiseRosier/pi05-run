"""Activation buffers and variance stats for Pi0.5 transcoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor


@dataclass(frozen=True)
class ActivationBatch:
    """Sampled token-position records for one target MLP."""

    x: Tensor
    y: Tensor
    timestep: Tensor


@dataclass(frozen=True)
class ActivationBufferStats:
    """Lightweight buffer state for logging."""

    size: int
    capacity: int
    variance_count: int
    variance_denominator: float


class RunningVarianceDenominator:
    """Online estimate of ``sum_j Var[y_j]`` for one target MLP."""

    def __init__(self, d_model: int, *, device: torch.device | str = "cpu"):
        self.d_model = d_model
        self.device = torch.device(device)
        self.count = 0
        self.mean = torch.zeros(d_model, device=self.device, dtype=torch.float32)
        self.m2 = torch.zeros(d_model, device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def update(self, values: Tensor) -> None:
        """Update stats from records shaped ``[records, d_model]``."""
        if values.ndim != 2 or values.shape[-1] != self.d_model:
            raise ValueError(f"Expected values shape [records, {self.d_model}], got {tuple(values.shape)}")
        if values.shape[0] == 0:
            return

        batch = values.detach().to(device=self.device, dtype=torch.float32)
        batch_count = batch.shape[0]
        batch_mean = batch.mean(dim=0)
        batch_m2 = (batch - batch_mean).pow(2).sum(dim=0)

        if self.count == 0:
            self.count = batch_count
            self.mean.copy_(batch_mean)
            self.m2.copy_(batch_m2)
            return

        total_count = self.count + batch_count
        delta = batch_mean - self.mean
        self.m2.add_(batch_m2 + delta.pow(2) * self.count * batch_count / total_count)
        self.mean.add_(delta * batch_count / total_count)
        self.count = total_count

    def variance_denominator(self, *, device: torch.device | str | None = None) -> Tensor:
        """Return ``sum_j Var[y_j]`` as a scalar tensor."""
        output_device = self.device if device is None else torch.device(device)
        if self.count < 2:
            return torch.ones((), device=output_device, dtype=torch.float32)
        denominator = (self.m2 / (self.count - 1)).sum()
        return denominator.to(device=output_device)


def flatten_activation_record(x: Tensor, y: Tensor, timestep: Tensor) -> ActivationBatch:
    """Flatten one wrapped MLP call into token-position records."""
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {tuple(x.shape)} and {tuple(y.shape)}")
    if x.ndim not in (2, 3):
        raise ValueError(f"Expected x/y rank 2 or 3, got {tuple(x.shape)}")

    if x.ndim == 2:
        num_records = x.shape[0]
        flat_timestep = _expand_2d_timestep(timestep, num_records, device=x.device)
        return ActivationBatch(x=x.reshape(num_records, x.shape[-1]), y=y.reshape(num_records, y.shape[-1]), timestep=flat_timestep)

    batch, tokens, d_model = x.shape
    flat_timestep = _expand_3d_timestep(timestep, batch, tokens, device=x.device)
    return ActivationBatch(
        x=x.reshape(batch * tokens, d_model),
        y=y.reshape(batch * tokens, d_model),
        timestep=flat_timestep,
    )


def _expand_2d_timestep(timestep: Tensor, records: int, *, device: torch.device) -> Tensor:
    timestep = timestep.to(device=device).reshape(-1)
    if timestep.numel() == 1:
        return timestep.expand(records)
    if timestep.numel() != records:
        raise ValueError(f"Expected 1 or {records} timestep values, got {timestep.numel()}")
    return timestep


def _expand_3d_timestep(timestep: Tensor, batch: int, tokens: int, *, device: torch.device) -> Tensor:
    timestep = timestep.to(device=device)
    if timestep.ndim == 0:
        return timestep.reshape(1).expand(batch * tokens)
    if timestep.ndim == 1:
        if timestep.numel() == 1:
            return timestep.expand(batch * tokens)
        if timestep.shape[0] != batch:
            raise ValueError(f"Expected timestep shape [1] or [{batch}], got {tuple(timestep.shape)}")
        return timestep[:, None].expand(batch, tokens).reshape(batch * tokens)
    if timestep.ndim == 2:
        if timestep.shape == (batch, 1):
            return timestep.expand(batch, tokens).reshape(batch * tokens)
        if timestep.shape == (batch, tokens):
            return timestep.reshape(batch * tokens)
    raise ValueError(f"Cannot expand timestep shape {tuple(timestep.shape)} for x shape [{batch}, {tokens}, d_model]")


class ActivationBuffer:
    """Bounded ring buffer for one target MLP's token-position records."""

    def __init__(
        self,
        *,
        d_model: int,
        capacity: int,
        storage_device: torch.device | str = "cpu",
        storage_dtype: torch.dtype = torch.float16,
    ):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.d_model = d_model
        self.capacity = capacity
        self.storage_device = torch.device(storage_device)
        self.storage_dtype = storage_dtype
        self.x = torch.empty((capacity, d_model), device=self.storage_device, dtype=storage_dtype)
        self.y = torch.empty((capacity, d_model), device=self.storage_device, dtype=storage_dtype)
        self.timestep = torch.empty((capacity,), device=self.storage_device, dtype=torch.float32)
        self.next_index = 0
        self.size = 0
        self.variance = RunningVarianceDenominator(d_model, device=self.storage_device)

    @torch.no_grad()
    def add(self, x: Tensor, y: Tensor, timestep: Tensor) -> int:
        """Add one wrapped MLP call, returning the number of token records stored."""
        batch = flatten_activation_record(x, y, timestep)
        return self.add_batch(batch)

    @torch.no_grad()
    def add_batch(self, batch: ActivationBatch) -> int:
        if batch.x.ndim != 2 or batch.x.shape[-1] != self.d_model:
            raise ValueError(f"Expected batch.x shape [records, {self.d_model}], got {tuple(batch.x.shape)}")
        if batch.y.shape != batch.x.shape:
            raise ValueError(f"Expected batch.y shape {tuple(batch.x.shape)}, got {tuple(batch.y.shape)}")
        if batch.timestep.reshape(-1).shape[0] != batch.x.shape[0]:
            raise ValueError(
                f"Expected {batch.x.shape[0]} timesteps for activation batch, got {batch.timestep.reshape(-1).shape[0]}"
            )

        records = batch.x.shape[0]
        if records == 0:
            return 0

        self.variance.update(batch.y)
        source_x = batch.x.detach().to(device=self.storage_device, dtype=self.storage_dtype)
        source_y = batch.y.detach().to(device=self.storage_device, dtype=self.storage_dtype)
        source_t = batch.timestep.detach().reshape(-1).to(device=self.storage_device, dtype=torch.float32)

        remaining = records
        source_start = 0
        while remaining > 0:
            destination_space = self.capacity - self.next_index
            chunk = min(remaining, destination_space)
            source_slice = slice(source_start, source_start + chunk)
            destination_slice = slice(self.next_index, self.next_index + chunk)
            self.x[destination_slice].copy_(source_x[source_slice])
            self.y[destination_slice].copy_(source_y[source_slice])
            self.timestep[destination_slice].copy_(source_t[source_slice])
            self.next_index = (self.next_index + chunk) % self.capacity
            self.size = min(self.capacity, self.size + chunk)
            source_start += chunk
            remaining -= chunk

        return records

    def sample(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
        generator: torch.Generator | None = None,
    ) -> ActivationBatch:
        """Sample records for one transcoder optimizer step."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty activation buffer")

        indices = torch.randint(self.size, (batch_size,), device=self.storage_device, generator=generator)
        output_device = torch.device(device)
        output_dtype = self.storage_dtype if dtype is None else dtype
        return ActivationBatch(
            x=self.x[indices].to(device=output_device, dtype=output_dtype),
            y=self.y[indices].to(device=output_device, dtype=output_dtype),
            timestep=self.timestep[indices].to(device=output_device),
        )

    def epoch_batches(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
        shuffle: bool = True,
        drop_last: bool = False,
        generator: torch.Generator | None = None,
    ) -> Iterable[ActivationBatch]:
        """Yield one epoch of minibatches without replacement."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if self.size == 0:
            raise RuntimeError("Cannot iterate over an empty activation buffer")

        if shuffle:
            indices = torch.randperm(self.size, device=self.storage_device, generator=generator)
        else:
            indices = torch.arange(self.size, device=self.storage_device)

        output_device = torch.device(device)
        output_dtype = self.storage_dtype if dtype is None else dtype
        for start in range(0, self.size, batch_size):
            batch_indices = indices[start : start + batch_size]
            if drop_last and batch_indices.numel() < batch_size:
                continue
            yield ActivationBatch(
                x=self.x[batch_indices].to(device=output_device, dtype=output_dtype),
                y=self.y[batch_indices].to(device=output_device, dtype=output_dtype),
                timestep=self.timestep[batch_indices].to(device=output_device),
            )

    def variance_denominator(self, *, device: torch.device | str | None = None) -> Tensor:
        return self.variance.variance_denominator(device=device)

    def stats(self) -> ActivationBufferStats:
        return ActivationBufferStats(
            size=self.size,
            capacity=self.capacity,
            variance_count=self.variance.count,
            variance_denominator=float(self.variance_denominator(device="cpu").item()),
        )


class MultiLayerActivationBuffer:
    """Per-target collection of activation buffers."""

    def __init__(
        self,
        *,
        layer_names: Iterable[str],
        d_model: int,
        capacity_per_layer: int,
        storage_device: torch.device | str = "cpu",
        storage_dtype: torch.dtype = torch.float16,
    ):
        self.buffers = {
            name: ActivationBuffer(
                d_model=d_model,
                capacity=capacity_per_layer,
                storage_device=storage_device,
                storage_dtype=storage_dtype,
            )
            for name in layer_names
        }

    @torch.no_grad()
    def add_context_records(self, context, *, clear_context: bool = True) -> dict[str, int]:
        """Drain records captured by ``Pi05TranscoderContext`` into buffers."""
        added: dict[str, int] = {}
        for name, records in context.records.items():
            if name not in self.buffers:
                continue
            total = 0
            for record in records:
                total += self.buffers[name].add(record.x, record.y, record.timestep)
            added[name] = total
        if clear_context:
            context.clear_records()
        return added

    def sample(
        self,
        name: str,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
        generator: torch.Generator | None = None,
    ) -> ActivationBatch:
        return self.buffers[name].sample(batch_size, device=device, dtype=dtype, generator=generator)

    def variance_denominator(self, name: str, *, device: torch.device | str | None = None) -> Tensor:
        return self.buffers[name].variance_denominator(device=device)

    def epoch_batches(
        self,
        name: str,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
        shuffle: bool = True,
        drop_last: bool = False,
        generator: torch.Generator | None = None,
    ) -> Iterable[ActivationBatch]:
        return self.buffers[name].epoch_batches(
            batch_size,
            device=device,
            dtype=dtype,
            shuffle=shuffle,
            drop_last=drop_last,
            generator=generator,
        )

    def ready_layer_names(self, minimum_records: int) -> list[str]:
        return [name for name, buffer in self.buffers.items() if buffer.size >= minimum_records]

    def stats(self) -> dict[str, ActivationBufferStats]:
        return {name: buffer.stats() for name, buffer in self.buffers.items()}

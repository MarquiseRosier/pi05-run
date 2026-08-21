"""DifFRACT-style timestep-conditioned transcoders.

The transcoder approximates one transformer MLP:

    TC_l(x, t) ~= MLP_l(x)

where x is one token-position hidden vector and t is the flow/diffusion
timestep. The module also supports batched token tensors shaped
``[batch, tokens, d_model]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class TimeConditionedTranscoderConfig:
    """Configuration for one timestep-conditioned transcoder."""

    d_model: int
    d_features: int | None = None
    expansion_factor: int = 16
    time_embedding_dim: int | None = None
    time_hidden_dim: int | None = None
    min_period: float = 4e-3
    max_period: float = 4.0
    eps: float = 1e-8

    @property
    def latent_dim(self) -> int:
        return self.d_features if self.d_features is not None else self.d_model * self.expansion_factor

    @property
    def time_dim(self) -> int:
        return self.time_embedding_dim if self.time_embedding_dim is not None else self.d_model

    @property
    def time_hidden(self) -> int:
        return self.time_hidden_dim if self.time_hidden_dim is not None else self.d_model


@dataclass(frozen=True)
class TranscoderLoss:
    """Scalar loss terms for logging and optimization."""

    total: Tensor
    normalized_mse: Tensor
    l1: Tensor


def sinusoidal_time_embedding(
    timesteps: Tensor,
    dim: int,
    *,
    min_period: float = 4e-3,
    max_period: float = 4.0,
) -> Tensor:
    """Create sinusoidal timestep embeddings for scalar flow/diffusion times.

    Args:
        timesteps: Tensor shaped ``[batch]`` or ``[batch, 1]``.
        dim: Output embedding width.
        min_period: Shortest sinusoid period.
        max_period: Longest sinusoid period.
    """
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")

    t = timesteps.float().reshape(-1, 1)
    half_dim = dim // 2
    if half_dim == 0:
        return t.new_zeros((t.shape[0], dim))

    periods = torch.logspace(
        start=torch.log10(t.new_tensor(min_period)),
        end=torch.log10(t.new_tensor(max_period)),
        steps=half_dim,
        device=t.device,
        dtype=t.dtype,
    )
    phases = 2.0 * torch.pi * t / periods.unsqueeze(0)
    emb = torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb


def normalized_mse_loss(prediction: Tensor, target: Tensor, variance_denominator: Tensor, eps: float) -> Tensor:
    """Return variance-normalized squared reconstruction error.

    ``variance_denominator`` is the per-transcoder denominator:
    ``sum_j Var[MLP_l(x)_j]`` estimated over that MLP's activation records.
    """
    squared_error = (prediction.float() - target.float()).pow(2).sum(dim=-1).mean()
    denom = variance_denominator.float().clamp_min(eps)
    return squared_error / denom


class TimeConditionedTranscoder(nn.Module):
    """Sparse timestep-conditioned transcoder for one MLP."""

    def __init__(self, config: TimeConditionedTranscoderConfig):
        super().__init__()
        self.config = config
        d_model = config.d_model
        d_features = config.latent_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(config.time_dim, config.time_hidden),
            nn.SiLU(),
            nn.Linear(config.time_hidden, 2 * d_model),
        )
        self.encoder = nn.Linear(d_model, d_features)
        self.decoder = nn.Linear(d_features, d_model)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.encoder.weight, a=5**0.5)
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.decoder.weight, a=5**0.5)
        nn.init.zeros_(self.decoder.bias)

        last_time = self.time_mlp[-1]
        if not isinstance(last_time, nn.Linear):
            raise TypeError("Expected final time_mlp module to be nn.Linear")
        nn.init.zeros_(last_time.weight)
        nn.init.zeros_(last_time.bias)

    def _time_scale_shift(self, timesteps: Tensor, *, dtype: torch.dtype, device: torch.device) -> tuple[Tensor, Tensor]:
        time_emb = sinusoidal_time_embedding(
            timesteps.to(device=device),
            self.config.time_dim,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
        ).to(dtype=dtype)
        scale_shift = self.time_mlp(time_emb)
        scale, shift = scale_shift.chunk(2, dim=-1)
        return scale, shift

    def forward(self, x: Tensor, timesteps: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(predicted_mlp_output, sparse_latent)``.

        Args:
            x: MLP input, shaped ``[records, d_model]`` or
                ``[batch, tokens, d_model]``.
            timesteps: Scalar timesteps shaped ``[records]`` for 2D x or
                ``[batch]`` for 3D x.
        """
        if x.shape[-1] != self.config.d_model:
            raise ValueError(f"Expected x last dim {self.config.d_model}, got {x.shape[-1]}")

        param_dtype = self.encoder.weight.dtype
        x_for_transcoder = x.to(dtype=param_dtype)
        scale, shift = self._time_scale_shift(timesteps, dtype=param_dtype, device=x.device)
        if x.ndim == 3:
            if scale.shape[0] != x.shape[0]:
                raise ValueError(f"Expected {x.shape[0]} timesteps for 3D x, got {scale.shape[0]}")
            scale = scale[:, None, :]
            shift = shift[:, None, :]
        elif x.ndim == 2:
            if scale.shape[0] != x.shape[0]:
                raise ValueError(f"Expected {x.shape[0]} timesteps for 2D x, got {scale.shape[0]}")
        else:
            raise ValueError(f"Expected x with rank 2 or 3, got shape {tuple(x.shape)}")

        x_mod = x_for_transcoder * (1.0 + scale) + shift
        z = F.relu(self.encoder(x_mod))
        y_hat = self.decoder(z)
        return y_hat, z

    def loss(
        self,
        x: Tensor,
        timesteps: Tensor,
        target: Tensor,
        variance_denominator: Tensor,
        *,
        lambda_l1: float,
    ) -> TranscoderLoss:
        """Compute DifFRACT-style normalized reconstruction + sparsity loss."""
        prediction, latent = self(x, timesteps)
        normalized_mse = normalized_mse_loss(
            prediction,
            target,
            variance_denominator,
            eps=self.config.eps,
        )
        l1 = latent.float().abs().sum(dim=-1).mean()
        total = normalized_mse + lambda_l1 * l1
        return TranscoderLoss(total=total, normalized_mse=normalized_mse, l1=l1)

    @torch.no_grad()
    def renormalize_decoder_columns(self, eps: float | None = None) -> None:
        """Normalize decoder feature vectors after an optimizer step.

        ``nn.Linear`` stores decoder weights as ``[d_model, d_features]``.
        Each column is the output vector for one sparse feature.
        """
        epsilon = self.config.eps if eps is None else eps
        weight = self.decoder.weight
        column_norms = weight.norm(dim=0, keepdim=True).clamp_min(epsilon)
        weight.div_(column_norms)

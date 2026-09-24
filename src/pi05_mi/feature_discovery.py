"""Streaming feature-discovery summaries for Pi0.5 transcoders."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class FeatureDiscoveryConfig:
    """Configuration for Top-K and global sparse-feature statistics."""

    top_k: int = 20
    firing_threshold: float = 1e-6
    top_m_active: int = 0


def _jsonable(value: Any) -> Any:
    if isinstance(value, Tensor):
        value = value.detach().cpu()
        if value.ndim == 0:
            return _jsonable(value.item())
        if value.numel() <= 16:
            return [_jsonable(item) for item in value.tolist()]
        return {"shape": list(value.shape), "dtype": str(value.dtype).replace("torch.", "")}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value[:16]]
    return str(value)


def batch_size_from_raw_batch(raw_batch: dict[str, Any]) -> int:
    """Infer dataloader batch size from the first batched value."""
    for value in raw_batch.values():
        if isinstance(value, Tensor) and value.ndim > 0:
            return int(value.shape[0])
        if isinstance(value, (list, tuple)):
            return len(value)
    raise ValueError("Could not infer batch size from raw batch")


def observation_metadata(raw_batch: dict[str, Any], row: int, *, camera_keys: list[str] | tuple[str, ...]) -> dict[str, Any]:
    """Return compact JSON metadata for one robot observation row."""
    metadata: dict[str, Any] = {}
    preferred_keys = (
        "index",
        "episode_index",
        "frame_index",
        "timestamp",
        "task",
        "task_index",
        "action",
        "observation.state",
    )
    for key in preferred_keys:
        if key not in raw_batch or key in camera_keys:
            continue
        value = raw_batch[key]
        if isinstance(value, Tensor) and value.ndim > 0 and value.shape[0] > row:
            value = value[row]
        elif isinstance(value, (list, tuple)) and len(value) > row:
            value = value[row]
        metadata[key] = _jsonable(value)

    metadata["camera_keys"] = list(camera_keys)
    for key in camera_keys:
        value = raw_batch.get(key)
        if isinstance(value, Tensor):
            metadata[f"{key}.shape"] = list(value.shape[1:])
            metadata[f"{key}.dtype"] = str(value.dtype).replace("torch.", "")
    return metadata


class RunningFeatureStats:
    """Online mean/std/frequency over collapsed feature scores."""

    def __init__(self, d_features: int, *, firing_threshold: float, top_m_active: int = 0):
        self.d_features = d_features
        self.firing_threshold = firing_threshold
        self.top_m_active = max(0, int(top_m_active))
        self.count = 0
        self.mean = torch.zeros(d_features, dtype=torch.float32)
        self.m2 = torch.zeros(d_features, dtype=torch.float32)
        self.active_count = torch.zeros(d_features, dtype=torch.int64)
        self.top_m_count = torch.zeros(d_features, dtype=torch.int64)

    @torch.no_grad()
    def update(self, scores: Tensor) -> None:
        if scores.ndim != 2 or scores.shape[-1] != self.d_features:
            raise ValueError(f"Expected scores shape [batch, {self.d_features}], got {tuple(scores.shape)}")
        scores = scores.detach().to(device="cpu", dtype=torch.float32)
        batch_count = scores.shape[0]
        if batch_count == 0:
            return

        batch_mean = scores.mean(dim=0)
        batch_m2 = (scores - batch_mean).pow(2).sum(dim=0)
        if self.count == 0:
            self.count = batch_count
            self.mean.copy_(batch_mean)
            self.m2.copy_(batch_m2)
        else:
            total_count = self.count + batch_count
            delta = batch_mean - self.mean
            self.m2.add_(batch_m2 + delta.pow(2) * self.count * batch_count / total_count)
            self.mean.add_(delta * batch_count / total_count)
            self.count = total_count

        self.active_count.add_((scores > self.firing_threshold).sum(dim=0, dtype=torch.int64))
        if self.top_m_active > 0:
            k = min(self.top_m_active, self.d_features)
            indices = torch.topk(scores, k=k, dim=-1).indices.reshape(-1)
            self.top_m_count.scatter_add_(0, indices, torch.ones_like(indices, dtype=torch.int64))

    def state_dict(self) -> dict[str, Any]:
        denominator = max(1, self.count)
        if self.count > 1:
            variance = self.m2 / (self.count - 1)
        else:
            variance = torch.zeros_like(self.mean)
        return {
            "count": self.count,
            "mean": self.mean,
            "std": variance.clamp_min(0).sqrt(),
            "firing_frequency": self.active_count.float() / denominator,
            "active_count": self.active_count,
            "top_m_active": self.top_m_active,
            "top_m_frequency": self.top_m_count.float() / denominator,
            "top_m_count": self.top_m_count,
        }


class FeatureTopK:
    """Streaming Top-K observation records for every sparse feature."""

    def __init__(self, d_features: int, top_k: int):
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        self.d_features = d_features
        self.top_k = top_k
        self.scores = torch.full((d_features, top_k), -torch.inf, dtype=torch.float32)
        self.observation_ids = torch.full((d_features, top_k), -1, dtype=torch.int64)
        self.action_positions = torch.full((d_features, top_k), -1, dtype=torch.int16)
        self.flow_timesteps = torch.full((d_features, top_k), torch.nan, dtype=torch.float32)

    @torch.no_grad()
    def update(self, scores: Tensor, observation_ids: Tensor, action_positions: Tensor, flow_timesteps: Tensor) -> None:
        if scores.ndim != 2 or scores.shape[-1] != self.d_features:
            raise ValueError(f"Expected scores shape [batch, {self.d_features}], got {tuple(scores.shape)}")
        batch_size = scores.shape[0]
        scores_t = scores.detach().to(device="cpu", dtype=torch.float32).transpose(0, 1)
        obs_t = observation_ids.detach().to(device="cpu", dtype=torch.int64).reshape(1, batch_size).expand(self.d_features, -1)
        pos_t = action_positions.detach().to(device="cpu", dtype=torch.int16).transpose(0, 1)
        tau_t = flow_timesteps.detach().to(device="cpu", dtype=torch.float32).transpose(0, 1)

        combined_scores = torch.cat([self.scores, scores_t], dim=1)
        combined_obs = torch.cat([self.observation_ids, obs_t], dim=1)
        combined_pos = torch.cat([self.action_positions, pos_t], dim=1)
        combined_tau = torch.cat([self.flow_timesteps, tau_t], dim=1)

        top_scores, top_indices = torch.topk(combined_scores, k=self.top_k, dim=1)
        self.scores = top_scores
        self.observation_ids = torch.gather(combined_obs, 1, top_indices)
        self.action_positions = torch.gather(combined_pos, 1, top_indices)
        self.flow_timesteps = torch.gather(combined_tau, 1, top_indices)

    def state_dict(self) -> dict[str, Tensor]:
        return {
            "scores": self.scores,
            "observation_ids": self.observation_ids,
            "action_positions": self.action_positions,
            "flow_timesteps": self.flow_timesteps,
        }


class TokenSparsity:
    """Per-token L0 of a sparse code, accumulated without keeping the tokens.

    L0 is the number of features active at one action token. The feature
    statistics elsewhere in this module collapse the code by a maximum over
    tokens, which answers "did this feature fire anywhere in the chunk" and
    therefore bounds L0 from above by up to the token count. This records the
    quantity the word "sparse" actually refers to.

    A histogram over the 0..d_features range gives exact quantiles at any sample
    size for the cost of one integer array per (layer, flow time).
    """

    def __init__(self, d_features: int):
        self.d_features = int(d_features)
        self.histogram = torch.zeros(self.d_features + 1, dtype=torch.int64)
        self.tokens = 0
        self.total = 0

    def update(self, l0_values: Tensor) -> None:
        flat = l0_values.detach().reshape(-1).to(dtype=torch.int64).clamp_(0, self.d_features)
        self.histogram += torch.bincount(flat.cpu(), minlength=self.d_features + 1)
        self.tokens += int(flat.numel())
        self.total += int(flat.sum())

    def quantile(self, q: float) -> float:
        """Smallest L0 whose cumulative share reaches ``q``.

        The search runs in floating point: casting a fractional target to the
        histogram's integer dtype would truncate it and return the quantile
        below the one asked for.
        """
        if self.tokens == 0:
            return float("nan")
        cumulative = torch.cumsum(self.histogram, dim=0).to(torch.float64)
        target = torch.tensor(q * self.tokens, dtype=torch.float64)
        index = int(torch.searchsorted(cumulative, target))
        return float(min(index, self.d_features))

    def state_dict(self) -> dict[str, Any]:
        nonzero = torch.nonzero(self.histogram, as_tuple=False).reshape(-1)
        return {
            "tokens": self.tokens,
            "d_features": self.d_features,
            "mean": (self.total / self.tokens) if self.tokens else float("nan"),
            "median": self.quantile(0.5),
            "p90": self.quantile(0.9),
            "min": float(nonzero[0]) if nonzero.numel() else float("nan"),
            "max": float(nonzero[-1]) if nonzero.numel() else float("nan"),
            "histogram": self.histogram,
        }


class FeatureDiscoveryCollector:
    """Collapse latent activations over action position while keeping flow time."""

    def __init__(
        self,
        *,
        layer_names: list[str],
        layer_indices: dict[str, int],
        d_features: int,
        config: FeatureDiscoveryConfig,
        observations_path: Path,
        camera_keys: list[str] | tuple[str, ...],
    ):
        self.layer_names = layer_names
        self.layer_indices = layer_indices
        self.d_features = d_features
        self.config = config
        self.observations_path = observations_path
        self.camera_keys = list(camera_keys)
        self.topk: dict[str, dict[str, FeatureTopK]] = {name: {} for name in layer_names}
        self.stats: dict[str, dict[str, RunningFeatureStats]] = {name: {} for name in layer_names}
        self.token_sparsity: dict[str, dict[str, TokenSparsity]] = {name: {} for name in layer_names}
        self.timestep_values: dict[str, dict[str, float]] = {name: {} for name in layer_names}
        self.current_observation_ids: Tensor | None = None
        self.pending: dict[str, dict[str, Tensor]] = {}
        self.next_observation_id = 0
        self.observation_count = 0
        self.observations_path.parent.mkdir(parents=True, exist_ok=True)
        self._observations_file = self.observations_path.open("w", buffering=1)

    @staticmethod
    def _timestep_key(value: float) -> str:
        return f"{value:.8f}"

    def _ensure_timestep(self, name: str, timestep_value: float) -> str:
        key = self._timestep_key(timestep_value)
        if key not in self.topk[name]:
            self.topk[name][key] = FeatureTopK(self.d_features, self.config.top_k)
            self.stats[name][key] = RunningFeatureStats(
                self.d_features,
                firing_threshold=self.config.firing_threshold,
                top_m_active=self.config.top_m_active,
            )
            self.timestep_values[name][key] = timestep_value
        return key

    def close(self) -> None:
        if not self._observations_file.closed:
            self._observations_file.close()

    def begin_batch(self, raw_batch: dict[str, Any]) -> int:
        if self.current_observation_ids is not None:
            raise RuntimeError("begin_batch called before end_batch")
        batch_size = batch_size_from_raw_batch(raw_batch)
        observation_ids = torch.arange(
            self.next_observation_id,
            self.next_observation_id + batch_size,
            dtype=torch.int64,
        )
        self.current_observation_ids = observation_ids
        self.pending = {}
        for row, observation_id in enumerate(observation_ids.tolist()):
            metadata = observation_metadata(raw_batch, row, camera_keys=self.camera_keys)
            metadata["observation_id"] = observation_id
            self._observations_file.write(json.dumps(metadata, sort_keys=True, allow_nan=False) + "\n")
        self.next_observation_id += batch_size
        self.observation_count += batch_size
        return batch_size

    @torch.no_grad()
    def observe_latent(self, name: str, layer_index: int, latent: Tensor, timestep: Tensor) -> None:
        if self.current_observation_ids is None:
            raise RuntimeError("observe_latent called outside an active batch")
        if name not in self.topk:
            return
        z = latent.detach().float()
        # Per-token L0, taken before the max over action positions discards the
        # token axis. Rank-2 latents have no token axis and are skipped.
        token_l0 = (z > 0).sum(dim=-1) if z.ndim == 3 else None
        if z.ndim == 3:
            values, positions = z.max(dim=1)
        elif z.ndim == 2:
            values = z
            positions = torch.zeros_like(values, dtype=torch.long)
        else:
            raise ValueError(f"Expected latent rank 2 or 3, got {tuple(z.shape)}")

        batch_size = self.current_observation_ids.numel()
        if values.shape != (batch_size, self.d_features):
            raise ValueError(
                f"Expected collapsed latent shape [{batch_size}, {self.d_features}], got {tuple(values.shape)}"
            )
        tau = timestep.detach().float().to(device=values.device).reshape(-1)
        if tau.numel() == 1:
            tau = tau.expand(batch_size)
        if tau.numel() != batch_size:
            raise ValueError(f"Expected one timestep per observation, got {tuple(timestep.shape)} for batch {batch_size}")
        self.layer_indices[name] = layer_index

        tau_cpu = tau.detach().to(device="cpu", dtype=torch.float32)
        for timestep_value in sorted({float(value) for value in tau_cpu.tolist()}):
            timestep_key = self._ensure_timestep(name, timestep_value)
            row_mask = tau_cpu == timestep_value
            row_indices = torch.nonzero(row_mask, as_tuple=False).reshape(-1)
            selected_scores = values.detach().cpu()[row_indices]
            selected_positions = positions.detach().to(device="cpu", dtype=torch.int16)[row_indices]
            selected_observations = self.current_observation_ids[row_indices]
            if token_l0 is not None:
                store = self.token_sparsity[name].get(timestep_key)
                if store is None:
                    store = TokenSparsity(self.d_features)
                    self.token_sparsity[name][timestep_key] = store
                store.update(token_l0[row_indices])
            selected_timesteps = torch.full_like(selected_scores, timestep_value, dtype=torch.float32)
            pending_key = f"{name}|{timestep_key}"

            pending = self.pending.get(pending_key)
            if pending is None:
                self.pending[pending_key] = {
                    "scores": selected_scores,
                    "positions": selected_positions,
                    "timesteps": selected_timesteps,
                    "observation_ids": selected_observations,
                }
                continue

            if torch.equal(pending["observation_ids"], selected_observations):
                old_scores = pending["scores"]
                mask = selected_scores > old_scores
                pending["scores"] = torch.where(mask, selected_scores, old_scores)
                pending["positions"] = torch.where(mask, selected_positions, pending["positions"])
                pending["timesteps"] = torch.where(mask, selected_timesteps, pending["timesteps"])
            else:
                pending["scores"] = torch.cat([pending["scores"], selected_scores], dim=0)
                pending["positions"] = torch.cat([pending["positions"], selected_positions], dim=0)
                pending["timesteps"] = torch.cat([pending["timesteps"], selected_timesteps], dim=0)
                pending["observation_ids"] = torch.cat([pending["observation_ids"], selected_observations], dim=0)

    def end_batch(self) -> dict[str, int]:
        if self.current_observation_ids is None:
            raise RuntimeError("end_batch called without begin_batch")
        updated: dict[str, int] = {}
        for pending_key, pending in self.pending.items():
            name, timestep_key = pending_key.rsplit("|", 1)
            scores = pending["scores"]
            self.stats[name][timestep_key].update(scores)
            self.topk[name][timestep_key].update(
                scores,
                pending["observation_ids"],
                pending["positions"],
                pending["timesteps"],
            )
            updated[name] = updated.get(name, 0) + int(scores.shape[0])
        self.current_observation_ids = None
        self.pending = {}
        return updated

    def save(self, output_dir: Path, *, extra_config: dict[str, Any] | None = None) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        topk_payload = {
            "format_version": 2,
            "layer_names": self.layer_names,
            "layer_indices": self.layer_indices,
            "timestep_values": self.timestep_values,
            "d_features": self.d_features,
            "top_k": self.config.top_k,
            "observation_count": self.observation_count,
            "topk": {
                name: {timestep: store.state_dict() for timestep, store in sorted(stores.items())}
                for name, stores in self.topk.items()
            },
        }
        stats_payload = {
            "format_version": 2,
            "layer_names": self.layer_names,
            "layer_indices": self.layer_indices,
            "timestep_values": self.timestep_values,
            "d_features": self.d_features,
            "observation_count": self.observation_count,
            "stats": {
                name: {timestep: store.state_dict() for timestep, store in sorted(stores.items())}
                for name, stores in self.stats.items()
            },
        }
        sparsity_payload = {
            "format_version": 1,
            "layer_names": self.layer_names,
            "layer_indices": self.layer_indices,
            "timestep_values": self.timestep_values,
            "d_features": self.d_features,
            "observation_count": self.observation_count,
            "token_sparsity": {
                name: {timestep: store.state_dict() for timestep, store in sorted(stores.items())}
                for name, stores in self.token_sparsity.items()
            },
        }
        torch.save(topk_payload, output_dir / "feature_topk.pt")
        torch.save(stats_payload, output_dir / "feature_stats.pt")
        torch.save(sparsity_payload, output_dir / "token_sparsity.pt")
        config = {
            "top_k": self.config.top_k,
            "firing_threshold": self.config.firing_threshold,
            "top_m_active": self.config.top_m_active,
            "d_features": self.d_features,
            "layer_count": len(self.layer_names),
            "observation_count": self.observation_count,
            "format_version": 2,
            "aggregation": "max_over_action_position_only",
        }
        if extra_config:
            config.update(extra_config)
        with (output_dir / "config.json").open("w") as f:
            json.dump(config, f, indent=2, sort_keys=True)

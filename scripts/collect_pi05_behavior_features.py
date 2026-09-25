#!/usr/bin/env python
"""Collect behavior-linked Pi0.5 transcoder feature statistics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from tqdm.auto import tqdm

from lerobot.datasets.factory import make_dataset
from lerobot.policies import make_policy
from lerobot.utils.constants import ACTION

from pi05_mi.feature_discovery import FeatureDiscoveryConfig, FeatureTopK, RunningFeatureStats, batch_size_from_raw_batch, observation_metadata
from pi05_mi.patch_pi05 import Pi05TranscoderContext, install_pi05_action_expert_wrappers
from pi05_mi.transcoders import TimeConditionedTranscoder, TimeConditionedTranscoderConfig
from train_pi05_transcoders import (
    DEFAULT_POLICY_PATH,
    _collect_random_timestep_records,
    _config_with_episodes,
    _configure_train_config,
    _episode_summary,
    _make_dataloader,
    _make_preprocessor,
    _parse_episode_ids,
    _prepare_raw_batch,
    patch_pi05_checkpoint_key_compat,
    patch_transformers_causal_mask_compat,
    resolve_device,
    resolve_policy_dtype,
)


DEFAULT_OUTPUT_DIR = Path("outputs/features/pi05_libero/behavior_features")


class RunningFeatureBehaviorStats:
    """Online feature-activation association with a scalar behavior score."""

    def __init__(self, d_features: int):
        self.d_features = int(d_features)
        self.count = 0
        self.sum_x = torch.zeros(self.d_features, dtype=torch.float64)
        self.sum_x2 = torch.zeros(self.d_features, dtype=torch.float64)
        self.sum_xy = torch.zeros(self.d_features, dtype=torch.float64)
        self.sum_y = 0.0
        self.sum_y2 = 0.0

    @torch.no_grad()
    def update(self, scores: Tensor, behavior: Tensor) -> None:
        if scores.ndim != 2 or scores.shape[-1] != self.d_features:
            raise ValueError(f"Expected scores shape [batch, {self.d_features}], got {tuple(scores.shape)}")
        behavior = behavior.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        if behavior.numel() != scores.shape[0]:
            raise ValueError(f"Expected {scores.shape[0]} behavior scores, got {behavior.numel()}")
        scores = scores.detach().to(device="cpu", dtype=torch.float64)
        self.count += int(scores.shape[0])
        self.sum_x.add_(scores.sum(dim=0))
        self.sum_x2.add_(scores.pow(2).sum(dim=0))
        self.sum_xy.add_((scores * behavior[:, None]).sum(dim=0))
        self.sum_y += float(behavior.sum())
        self.sum_y2 += float(behavior.pow(2).sum())

    def state_dict(self) -> dict[str, Any]:
        n = max(1, int(self.count))
        mean_x = self.sum_x / n
        mean_y = self.sum_y / n
        var_x = (self.sum_x2 / n - mean_x.pow(2)).clamp_min(0.0)
        var_y = max(0.0, self.sum_y2 / n - mean_y * mean_y)
        cov = self.sum_xy / n - mean_x * mean_y
        denom = (var_x.sqrt() * math.sqrt(max(var_y, 1e-24))).clamp_min(1e-12)
        corr = cov / denom
        corr = torch.where(torch.isfinite(corr), corr, torch.zeros_like(corr)).float()
        return {
            "count": self.count,
            "mean_behavior": float(mean_y),
            "std_behavior": float(math.sqrt(max(var_y, 0.0))),
            "corr_behavior": corr,
            "cov_behavior": cov.float(),
        }


def _parse_phase_bins(raw: str) -> list[tuple[str, float, float]]:
    """Parse named normalized-progress bins like early:0:0.25,middle:0.25:0.6."""
    bins: list[tuple[str, float, float]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid phase bin {item!r}; expected name:start:end")
        name, start_raw, end_raw = parts
        if not name:
            raise ValueError(f"Invalid empty phase name in {item!r}")
        start = float(start_raw)
        end = float(end_raw)
        if not (0.0 <= start < end <= 1.0):
            raise ValueError(f"Invalid phase range {item!r}; require 0 <= start < end <= 1")
        bins.append((name, start, end))
    if not bins:
        raise ValueError("At least one phase bin is required")
    return bins


def _parse_manual_event_windows(raw: str) -> dict[int, tuple[int, int]]:
    """Parse episode:start:end windows separated by commas."""
    windows: dict[int, tuple[int, int]] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid manual event window {item!r}; expected episode:start:end")
        episode, start, end = (int(value) for value in parts)
        if start < 0 or end < start:
            raise ValueError(f"Invalid manual event window {item!r}; require 0 <= start <= end")
        if episode in windows:
            raise ValueError(f"Duplicate manual event window for episode {episode}")
        windows[episode] = (start, end)
    return windows


def _manual_event_for_metadata(
    metadata: dict[str, Any],
    *,
    event_name: str,
    event_windows: dict[int, tuple[int, int]],
    event_bins: int,
) -> tuple[list[str], float, str | None]:
    episode = _extract_int(metadata.get("episode_index"))
    frame = _extract_int(metadata.get("frame_index"))
    if episode is None or frame is None or episode not in event_windows:
        return [], float("nan"), None
    start, end = event_windows[episode]
    if frame < start or frame > end:
        return [], float("nan"), None
    progress = 0.0 if end == start else (frame - start) / (end - start)
    bin_index = min(event_bins - 1, int(progress * event_bins))
    bin_label = f"{event_name}:q{bin_index + 1}"
    return [event_name, bin_label], progress, bin_label


def _subset_raw_batch(raw_batch: dict[str, Any], row_indices: Tensor) -> dict[str, Any]:
    """Select rows from a collated LeRobot batch while preserving scalar metadata."""
    batch_size = batch_size_from_raw_batch(raw_batch)
    selected = row_indices.detach().cpu().tolist()
    result: dict[str, Any] = {}
    for key, value in raw_batch.items():
        if isinstance(value, Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
            result[key] = value.index_select(0, row_indices.to(device=value.device))
        elif isinstance(value, list) and len(value) == batch_size:
            result[key] = [value[index] for index in selected]
        elif isinstance(value, tuple) and len(value) == batch_size:
            result[key] = tuple(value[index] for index in selected)
        else:
            result[key] = value
    return result


def _select_manual_event_rows(
    raw_batch: dict[str, Any], event_windows: dict[int, tuple[int, int]]
) -> Tensor:
    episodes = raw_batch.get("episode_index")
    frames = raw_batch.get("frame_index")
    if not isinstance(episodes, Tensor) or not isinstance(frames, Tensor):
        raise KeyError("Manual event filtering requires tensor episode_index and frame_index fields")
    episodes = episodes.detach().cpu().reshape(-1)
    frames = frames.detach().cpu().reshape(-1)
    keep = torch.zeros_like(episodes, dtype=torch.bool)
    for episode, (start, end) in event_windows.items():
        keep |= (episodes == episode) & (frames >= start) & (frames <= end)
    return torch.nonzero(keep, as_tuple=False).reshape(-1)


def _extract_int(value: Any) -> int | None:
    if isinstance(value, Tensor):
        value = value.detach().cpu()
        if value.numel() != 1:
            return None
        value = value.item()
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return None
    return None


def _episode_frame_max_from_dataset(dataset: Any) -> dict[int, int]:
    """Return max frame_index per episode without forcing policy inference."""
    frame_max: dict[int, int] = {}
    hf_dataset = getattr(dataset, "hf_dataset", None)
    if hf_dataset is not None and "episode_index" in hf_dataset.column_names and "frame_index" in hf_dataset.column_names:
        episodes = hf_dataset["episode_index"]
        frames = hf_dataset["frame_index"]
        for episode, frame in zip(episodes, frames, strict=False):
            ep_int = _extract_int(episode)
            frame_int = _extract_int(frame)
            if ep_int is None or frame_int is None:
                continue
            frame_max[ep_int] = max(frame_max.get(ep_int, -1), frame_int)
        if frame_max:
            return frame_max

    for row in range(len(dataset)):
        item = dataset[row]
        ep_int = _extract_int(item.get("episode_index"))
        frame_int = _extract_int(item.get("frame_index"))
        if ep_int is None or frame_int is None:
            continue
        frame_max[ep_int] = max(frame_max.get(ep_int, -1), frame_int)
    return frame_max


def _phase_for_metadata(
    metadata: dict[str, Any],
    *,
    episode_frame_max: dict[int, int],
    phase_bins: list[tuple[str, float, float]],
) -> tuple[str, float]:
    episode = _extract_int(metadata.get("episode_index"))
    frame = _extract_int(metadata.get("frame_index"))
    if episode is None or frame is None:
        return "unknown", float("nan")
    max_frame = max(1, int(episode_frame_max.get(episode, frame)))
    progress = max(0.0, min(1.0, float(frame) / float(max_frame)))
    for name, start, end in phase_bins:
        if start <= progress < end or (progress == 1.0 and end == 1.0):
            return name, progress
    return "unknown", progress


class BehaviorFeatureCollector:
    """Collect Top-K, activation stats, and feature-behavior correlations."""

    def __init__(
        self,
        *,
        layer_names: list[str],
        layer_indices: dict[str, int],
        d_features: int,
        top_k: int,
        firing_threshold: float,
        top_m_active: int,
        observations_path: Path,
        camera_keys: list[str] | tuple[str, ...],
        episode_frame_max: dict[int, int] | None = None,
        phase_bins: list[tuple[str, float, float]] | None = None,
        manual_event_name: str | None = None,
        manual_event_windows: dict[int, tuple[int, int]] | None = None,
        manual_event_bins: int = 4,
    ):
        self.layer_names = layer_names
        self.layer_indices = layer_indices
        self.d_features = int(d_features)
        self.top_k = int(top_k)
        self.firing_threshold = float(firing_threshold)
        self.top_m_active = int(top_m_active)
        self.observations_path = observations_path
        self.camera_keys = list(camera_keys)
        self.episode_frame_max = dict(episode_frame_max or {})
        self.phase_bins = list(phase_bins or [])
        self.manual_event_name = manual_event_name
        self.manual_event_windows = dict(manual_event_windows or {})
        self.manual_event_bins = int(manual_event_bins)
        self.topk: dict[str, dict[str, FeatureTopK]] = {name: {} for name in layer_names}
        self.stats: dict[str, dict[str, RunningFeatureStats]] = {name: {} for name in layer_names}
        self.behavior_stats: dict[str, dict[str, RunningFeatureBehaviorStats]] = {name: {} for name in layer_names}
        self.phase_stats: dict[str, dict[str, dict[str, RunningFeatureStats]]] = {name: {} for name in layer_names}
        self.phase_behavior_stats: dict[str, dict[str, dict[str, RunningFeatureBehaviorStats]]] = {
            name: {} for name in layer_names
        }
        self.event_stats: dict[str, dict[str, dict[str, RunningFeatureStats]]] = {name: {} for name in layer_names}
        self.event_behavior_stats: dict[str, dict[str, dict[str, RunningFeatureBehaviorStats]]] = {
            name: {} for name in layer_names
        }
        self.timestep_values: dict[str, dict[str, float]] = {name: {} for name in layer_names}
        self.phase_counts: dict[str, int] = {}
        self.event_counts: dict[str, int] = {}
        self.current_observation_ids: Tensor | None = None
        self.current_phase_labels: list[str] = []
        self.current_event_labels: list[list[str]] = []
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
            self.topk[name][key] = FeatureTopK(self.d_features, self.top_k)
            self.stats[name][key] = RunningFeatureStats(
                self.d_features,
                firing_threshold=self.firing_threshold,
                top_m_active=self.top_m_active,
            )
            self.behavior_stats[name][key] = RunningFeatureBehaviorStats(self.d_features)
            self.phase_stats[name][key] = {}
            self.phase_behavior_stats[name][key] = {}
            self.event_stats[name][key] = {}
            self.event_behavior_stats[name][key] = {}
            self.timestep_values[name][key] = timestep_value
        return key

    def _ensure_phase(self, name: str, timestep_key: str, phase: str) -> None:
        if phase not in self.phase_stats[name][timestep_key]:
            self.phase_stats[name][timestep_key][phase] = RunningFeatureStats(
                self.d_features,
                firing_threshold=self.firing_threshold,
                top_m_active=self.top_m_active,
            )
            self.phase_behavior_stats[name][timestep_key][phase] = RunningFeatureBehaviorStats(self.d_features)

    def _ensure_event(self, name: str, timestep_key: str, event_label: str) -> None:
        if event_label not in self.event_stats[name][timestep_key]:
            self.event_stats[name][timestep_key][event_label] = RunningFeatureStats(
                self.d_features,
                firing_threshold=self.firing_threshold,
                top_m_active=self.top_m_active,
            )
            self.event_behavior_stats[name][timestep_key][event_label] = RunningFeatureBehaviorStats(self.d_features)

    def close(self) -> None:
        if not self._observations_file.closed:
            self._observations_file.close()

    def begin_batch(self, raw_batch: dict[str, Any]) -> int:
        if self.current_observation_ids is not None:
            raise RuntimeError("begin_batch called before end_batch")
        batch_size = batch_size_from_raw_batch(raw_batch)
        observation_ids = torch.arange(self.next_observation_id, self.next_observation_id + batch_size, dtype=torch.int64)
        self.current_observation_ids = observation_ids
        self.pending = {}
        self._pending_metadata: list[dict[str, Any]] = []
        self.current_phase_labels = []
        self.current_event_labels = []
        for row, observation_id in enumerate(observation_ids.tolist()):
            metadata = observation_metadata(raw_batch, row, camera_keys=self.camera_keys)
            if self.phase_bins:
                phase, progress = _phase_for_metadata(
                    metadata,
                    episode_frame_max=self.episode_frame_max,
                    phase_bins=self.phase_bins,
                )
                metadata["event_phase"] = phase
                metadata["episode_progress"] = progress
                self.phase_counts[phase] = self.phase_counts.get(phase, 0) + 1
            else:
                phase = "all"
            metadata["observation_id"] = observation_id
            event_labels: list[str] = []
            if self.manual_event_name is not None and self.manual_event_windows:
                event_labels, event_progress, event_bin = _manual_event_for_metadata(
                    metadata,
                    event_name=self.manual_event_name,
                    event_windows=self.manual_event_windows,
                    event_bins=self.manual_event_bins,
                )
                if event_labels:
                    metadata["manual_event"] = self.manual_event_name
                    metadata["manual_event_progress"] = event_progress
                    metadata["manual_event_bin"] = event_bin
                    for event_label in event_labels:
                        self.event_counts[event_label] = self.event_counts.get(event_label, 0) + 1
            self._pending_metadata.append(metadata)
            self.current_phase_labels.append(phase)
            self.current_event_labels.append(event_labels)
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
        if z.ndim == 3:
            values, positions = z.max(dim=1)
        elif z.ndim == 2:
            values = z
            positions = torch.zeros_like(values, dtype=torch.long)
        else:
            raise ValueError(f"Expected latent rank 2 or 3, got {tuple(z.shape)}")

        batch_size = self.current_observation_ids.numel()
        if values.shape != (batch_size, self.d_features):
            raise ValueError(f"Expected collapsed latent shape [{batch_size}, {self.d_features}], got {tuple(values.shape)}")
        tau = timestep.detach().float().to(device=values.device).reshape(-1)
        if tau.numel() == 1:
            tau = tau.expand(batch_size)
        if tau.numel() != batch_size:
            raise ValueError(f"Expected one timestep per observation, got {tuple(timestep.shape)}")
        self.layer_indices[name] = layer_index

        tau_cpu = tau.detach().to(device="cpu", dtype=torch.float32)
        for timestep_value in sorted({float(value) for value in tau_cpu.tolist()}):
            timestep_key = self._ensure_timestep(name, timestep_value)
            row_mask = tau_cpu == timestep_value
            row_indices = torch.nonzero(row_mask, as_tuple=False).reshape(-1)
            selected_scores = values.detach().cpu()[row_indices]
            selected_positions = positions.detach().to(device="cpu", dtype=torch.int16)[row_indices]
            selected_observations = self.current_observation_ids[row_indices]
            selected_timesteps = torch.full_like(selected_scores, timestep_value, dtype=torch.float32)
            pending_key = f"{name}|{timestep_key}"
            pending = self.pending.get(pending_key)
            if pending is None:
                self.pending[pending_key] = {
                    "scores": selected_scores,
                    "positions": selected_positions,
                    "timesteps": selected_timesteps,
                    "observation_ids": selected_observations,
                    "row_indices": row_indices.detach().cpu(),
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
                pending["row_indices"] = torch.cat([pending["row_indices"], row_indices.detach().cpu()], dim=0)

    def end_batch(self, behavior_scores: Tensor, behavior_rows: list[dict[str, Any]]) -> dict[str, int]:
        if self.current_observation_ids is None:
            raise RuntimeError("end_batch called without begin_batch")
        behavior_scores = behavior_scores.detach().cpu().float().reshape(-1)
        if behavior_scores.numel() != self.current_observation_ids.numel():
            raise ValueError(f"Expected {self.current_observation_ids.numel()} behavior scores, got {behavior_scores.numel()}")
        for metadata, behavior in zip(self._pending_metadata, behavior_rows, strict=True):
            row = dict(metadata)
            row.update(behavior)
            self._observations_file.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")

        updated: dict[str, int] = {}
        for pending_key, pending in self.pending.items():
            name, timestep_key = pending_key.rsplit("|", 1)
            scores = pending["scores"]
            row_indices = pending["row_indices"].long()
            selected_behavior = behavior_scores[row_indices]
            self.stats[name][timestep_key].update(scores)
            self.behavior_stats[name][timestep_key].update(scores, selected_behavior)
            if self.phase_bins:
                selected_phases = [self.current_phase_labels[int(row)] for row in row_indices.tolist()]
                for phase in sorted(set(selected_phases)):
                    phase_mask = torch.tensor([label == phase for label in selected_phases], dtype=torch.bool)
                    if not bool(phase_mask.any()):
                        continue
                    self._ensure_phase(name, timestep_key, phase)
                    self.phase_stats[name][timestep_key][phase].update(scores[phase_mask])
                    self.phase_behavior_stats[name][timestep_key][phase].update(
                        scores[phase_mask],
                        selected_behavior[phase_mask],
                    )
            if self.manual_event_name is not None and self.manual_event_windows:
                selected_event_labels = [self.current_event_labels[int(row)] for row in row_indices.tolist()]
                event_labels = sorted({label for labels in selected_event_labels for label in labels})
                for event_label in event_labels:
                    event_mask = torch.tensor(
                        [event_label in labels for labels in selected_event_labels],
                        dtype=torch.bool,
                    )
                    if not bool(event_mask.any()):
                        continue
                    self._ensure_event(name, timestep_key, event_label)
                    self.event_stats[name][timestep_key][event_label].update(scores[event_mask])
                    self.event_behavior_stats[name][timestep_key][event_label].update(
                        scores[event_mask],
                        selected_behavior[event_mask],
                    )
            self.topk[name][timestep_key].update(
                scores,
                pending["observation_ids"],
                pending["positions"],
                pending["timesteps"],
            )
            updated[name] = updated.get(name, 0) + int(scores.shape[0])
        self.current_observation_ids = None
        self.current_phase_labels = []
        self.current_event_labels = []
        self.pending = {}
        self._pending_metadata = []
        return updated

    def save(self, output_dir: Path, *, extra_config: dict[str, Any] | None = None) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        topk_payload = {
            "format_version": 3,
            "layer_names": self.layer_names,
            "layer_indices": self.layer_indices,
            "timestep_values": self.timestep_values,
            "d_features": self.d_features,
            "top_k": self.top_k,
            "observation_count": self.observation_count,
            "topk": {
                name: {timestep: store.state_dict() for timestep, store in sorted(stores.items())}
                for name, stores in self.topk.items()
            },
        }
        stats_payload = {
            "format_version": 3,
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
        behavior_payload = {
            "format_version": 1,
            "layer_names": self.layer_names,
            "layer_indices": self.layer_indices,
            "timestep_values": self.timestep_values,
            "d_features": self.d_features,
            "observation_count": self.observation_count,
            "behavior": "speed_mean",
            "stats": {
                name: {timestep: store.state_dict() for timestep, store in sorted(stores.items())}
                for name, stores in self.behavior_stats.items()
            },
        }
        phase_stats_payload = {
            "format_version": 1,
            "layer_names": self.layer_names,
            "layer_indices": self.layer_indices,
            "timestep_values": self.timestep_values,
            "d_features": self.d_features,
            "observation_count": self.observation_count,
            "phase_bins": self.phase_bins,
            "phase_counts": self.phase_counts,
            "stats": {
                name: {
                    timestep: {phase: phase_store.state_dict() for phase, phase_store in sorted(phase_stores.items())}
                    for timestep, phase_stores in sorted(stores.items())
                }
                for name, stores in self.phase_stats.items()
            },
        }
        phase_behavior_payload = {
            "format_version": 1,
            "layer_names": self.layer_names,
            "layer_indices": self.layer_indices,
            "timestep_values": self.timestep_values,
            "d_features": self.d_features,
            "observation_count": self.observation_count,
            "phase_bins": self.phase_bins,
            "phase_counts": self.phase_counts,
            "behavior": "speed_mean",
            "stats": {
                name: {
                    timestep: {phase: phase_store.state_dict() for phase, phase_store in sorted(phase_stores.items())}
                    for timestep, phase_stores in sorted(stores.items())
                }
                for name, stores in self.phase_behavior_stats.items()
            },
        }
        event_stats_payload = {
            "format_version": 1,
            "layer_names": self.layer_names,
            "layer_indices": self.layer_indices,
            "timestep_values": self.timestep_values,
            "d_features": self.d_features,
            "observation_count": self.observation_count,
            "manual_event_name": self.manual_event_name,
            "manual_event_windows": self.manual_event_windows,
            "manual_event_bins": self.manual_event_bins,
            "event_counts": self.event_counts,
            "stats": {
                name: {
                    timestep: {
                        event_label: event_store.state_dict()
                        for event_label, event_store in sorted(event_stores.items())
                    }
                    for timestep, event_stores in sorted(stores.items())
                }
                for name, stores in self.event_stats.items()
            },
        }
        event_behavior_payload = {
            "format_version": 1,
            "layer_names": self.layer_names,
            "layer_indices": self.layer_indices,
            "timestep_values": self.timestep_values,
            "d_features": self.d_features,
            "observation_count": self.observation_count,
            "manual_event_name": self.manual_event_name,
            "manual_event_windows": self.manual_event_windows,
            "manual_event_bins": self.manual_event_bins,
            "event_counts": self.event_counts,
            "behavior": "speed_mean",
            "stats": {
                name: {
                    timestep: {
                        event_label: event_store.state_dict()
                        for event_label, event_store in sorted(event_stores.items())
                    }
                    for timestep, event_stores in sorted(stores.items())
                }
                for name, stores in self.event_behavior_stats.items()
            },
        }
        torch.save(topk_payload, output_dir / "feature_topk.pt")
        torch.save(stats_payload, output_dir / "feature_stats.pt")
        torch.save(behavior_payload, output_dir / "feature_behavior_stats.pt")
        if self.phase_bins:
            torch.save(phase_stats_payload, output_dir / "feature_phase_stats.pt")
            torch.save(phase_behavior_payload, output_dir / "feature_behavior_phase_stats.pt")
        if self.manual_event_name is not None and self.manual_event_windows:
            torch.save(event_stats_payload, output_dir / "feature_event_stats.pt")
            torch.save(event_behavior_payload, output_dir / "feature_behavior_event_stats.pt")
        config = {
            "top_k": self.top_k,
            "firing_threshold": self.firing_threshold,
            "top_m_active": self.top_m_active,
            "d_features": self.d_features,
            "layer_count": len(self.layer_names),
            "observation_count": self.observation_count,
            "format_version": 3,
            "aggregation": "max_over_action_position_only",
            "behavior": "speed_mean",
            "phase_bins": self.phase_bins,
            "phase_counts": self.phase_counts,
            "manual_event_name": self.manual_event_name,
            "manual_event_windows": self.manual_event_windows,
            "manual_event_bins": self.manual_event_bins,
            "event_counts": self.event_counts,
        }
        if extra_config:
            config.update(extra_config)
        with (output_dir / "config.json").open("w") as f:
            json.dump(config, f, indent=2, sort_keys=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", default=DEFAULT_POLICY_PATH)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--episodes", default=None, help="Comma-separated episode ids. Omit for all episodes.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--firing-threshold", type=float, default=1e-6)
    parser.add_argument("--top-m-active", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--top-candidates", type=int, default=100)
    parser.add_argument(
        "--phase-bins",
        default="early:0:0.25,middle:0.25:0.6,late:0.6:1",
        help="Comma-separated normalized-progress bins name:start:end. Empty string disables phase-conditioned stats.",
    )
    parser.add_argument("--manual-event-name", default=None, help="Name for a manually labeled task-local event.")
    parser.add_argument(
        "--manual-event-windows",
        default="",
        help="Comma-separated episode:start:end windows for the manual event.",
    )
    parser.add_argument(
        "--manual-event-bins",
        type=int,
        default=4,
        help="Number of normalized progress bins within each manual event window.",
    )
    parser.add_argument(
        "--manual-event-only",
        action="store_true",
        help="Feed forward only observations inside the supplied manual event windows.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--policy-dtype", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _freeze_policy(policy: torch.nn.Module) -> None:
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)


def _load_transcoders(checkpoint_path: Path, *, device: torch.device) -> dict[str, TimeConditionedTranscoder]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    transcoders: dict[str, TimeConditionedTranscoder] = {}
    for name, raw_config in checkpoint["configs"].items():
        config = TimeConditionedTranscoderConfig(**raw_config)
        transcoder = TimeConditionedTranscoder(config)
        transcoder.load_state_dict(checkpoint["state_dicts"][name])
        transcoder.to(device=device, dtype=torch.float32)
        transcoder.eval()
        for parameter in transcoder.parameters():
            parameter.requires_grad_(False)
        transcoders[name] = transcoder
    return transcoders


def _action_speed_metrics(actions: Tensor, *, action_dim: int) -> tuple[Tensor, Tensor, Tensor]:
    xyz_dim = min(3, int(action_dim), int(actions.shape[-1]))
    xyz = actions[..., :xyz_dim].float()
    per_step = xyz.pow(2).sum(dim=-1).sqrt()
    return per_step.mean(dim=-1), per_step.sum(dim=-1), per_step.max(dim=-1).values


def _rank_behavior_features(output_dir: Path, *, top_candidates: int) -> None:
    payload = torch.load(output_dir / "feature_behavior_stats.pt", map_location="cpu", weights_only=False)
    activation_stats = torch.load(output_dir / "feature_stats.pt", map_location="cpu", weights_only=False)
    rows: list[dict[str, Any]] = []
    for name in payload["layer_names"]:
        layer = int(payload["layer_indices"][name])
        for timestep, store in sorted(payload["stats"][name].items()):
            corr = store["corr_behavior"].float()
            behavior_std = float(store["std_behavior"])
            act_store = activation_stats["stats"][name][timestep]
            mean = act_store["mean"].float()
            std = act_store["std"].float()
            frequency = act_store["firing_frequency"].float()
            for feature in range(int(payload["d_features"])):
                rows.append(
                    {
                        "feature_key": f"L{layer:02d}:tau{float(timestep):.4g}:F{feature}",
                        "layer": layer,
                        "timestep": float(timestep),
                        "feature": feature,
                        "corr_speed": float(corr[feature]),
                        "abs_corr_speed": abs(float(corr[feature])),
                        "behavior_std": behavior_std,
                        "mean_activation": float(mean[feature]),
                        "std_activation": float(std[feature]),
                        "frequency": float(frequency[feature]),
                        "layer_name": name,
                        "timestep_key": timestep,
                    }
                )

    fieldnames = [
        "rank",
        "direction",
        "feature_key",
        "corr_speed",
        "abs_corr_speed",
        "behavior_std",
        "mean_activation",
        "std_activation",
        "frequency",
        "layer",
        "timestep",
        "feature",
        "layer_name",
        "timestep_key",
    ]
    for direction, selected in (
        ("fast", sorted(rows, key=lambda row: float(row["corr_speed"]), reverse=True)[:top_candidates]),
        ("slow", sorted(rows, key=lambda row: float(row["corr_speed"]))[:top_candidates]),
    ):
        for rank, row in enumerate(selected, start=1):
            row["rank"] = rank
            row["direction"] = direction
        with (output_dir / f"{direction}_feature_candidates.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(selected)


def _rank_phase_behavior_features(output_dir: Path, *, top_candidates: int) -> None:
    phase_behavior_path = output_dir / "feature_behavior_phase_stats.pt"
    phase_stats_path = output_dir / "feature_phase_stats.pt"
    if not phase_behavior_path.exists() or not phase_stats_path.exists():
        return
    payload = torch.load(phase_behavior_path, map_location="cpu", weights_only=False)
    activation_stats = torch.load(phase_stats_path, map_location="cpu", weights_only=False)
    rows: list[dict[str, Any]] = []
    for name in payload["layer_names"]:
        layer = int(payload["layer_indices"][name])
        for timestep, phase_stores in sorted(payload["stats"][name].items()):
            for phase, store in sorted(phase_stores.items()):
                corr = store["corr_behavior"].float()
                behavior_std = float(store["std_behavior"])
                behavior_count = int(store["count"])
                act_store = activation_stats["stats"][name][timestep][phase]
                mean = act_store["mean"].float()
                std = act_store["std"].float()
                frequency = act_store["firing_frequency"].float()
                for feature in range(int(payload["d_features"])):
                    rows.append(
                        {
                            "feature_key": f"L{layer:02d}:tau{float(timestep):.4g}:F{feature}",
                            "phase_feature_key": f"{phase}:L{layer:02d}:tau{float(timestep):.4g}:F{feature}",
                            "phase": phase,
                            "phase_observation_count": behavior_count,
                            "layer": layer,
                            "timestep": float(timestep),
                            "feature": feature,
                            "corr_speed": float(corr[feature]),
                            "abs_corr_speed": abs(float(corr[feature])),
                            "behavior_std": behavior_std,
                            "mean_activation": float(mean[feature]),
                            "std_activation": float(std[feature]),
                            "frequency": float(frequency[feature]),
                            "layer_name": name,
                            "timestep_key": timestep,
                        }
                    )

    fieldnames = [
        "rank",
        "direction",
        "phase",
        "phase_feature_key",
        "feature_key",
        "corr_speed",
        "abs_corr_speed",
        "behavior_std",
        "phase_observation_count",
        "mean_activation",
        "std_activation",
        "frequency",
        "layer",
        "timestep",
        "feature",
        "layer_name",
        "timestep_key",
    ]
    for direction, selected in (
        ("fast", sorted(rows, key=lambda row: float(row["corr_speed"]), reverse=True)[:top_candidates]),
        ("slow", sorted(rows, key=lambda row: float(row["corr_speed"]))[:top_candidates]),
    ):
        for rank, row in enumerate(selected, start=1):
            row["rank"] = rank
            row["direction"] = direction
        with (output_dir / f"{direction}_feature_candidates_by_phase.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(selected)


def _rank_event_behavior_features(output_dir: Path, *, top_candidates: int) -> None:
    event_behavior_path = output_dir / "feature_behavior_event_stats.pt"
    event_stats_path = output_dir / "feature_event_stats.pt"
    if not event_behavior_path.exists() or not event_stats_path.exists():
        return
    payload = torch.load(event_behavior_path, map_location="cpu", weights_only=False)
    activation_stats = torch.load(event_stats_path, map_location="cpu", weights_only=False)
    rows: list[dict[str, Any]] = []
    for name in payload["layer_names"]:
        layer = int(payload["layer_indices"][name])
        for timestep, event_stores in sorted(payload["stats"][name].items()):
            for event_label, store in sorted(event_stores.items()):
                corr = store["corr_behavior"].float()
                behavior_std = float(store["std_behavior"])
                behavior_count = int(store["count"])
                act_store = activation_stats["stats"][name][timestep][event_label]
                mean = act_store["mean"].float()
                std = act_store["std"].float()
                frequency = act_store["firing_frequency"].float()
                for feature in range(int(payload["d_features"])):
                    rows.append(
                        {
                            "event_feature_key": f"{event_label}:L{layer:02d}:tau{float(timestep):.4g}:F{feature}",
                            "feature_key": f"L{layer:02d}:tau{float(timestep):.4g}:F{feature}",
                            "event_label": event_label,
                            "event_observation_count": behavior_count,
                            "layer": layer,
                            "timestep": float(timestep),
                            "feature": feature,
                            "corr_speed": float(corr[feature]),
                            "abs_corr_speed": abs(float(corr[feature])),
                            "behavior_std": behavior_std,
                            "mean_activation": float(mean[feature]),
                            "std_activation": float(std[feature]),
                            "frequency": float(frequency[feature]),
                            "layer_name": name,
                            "timestep_key": timestep,
                        }
                    )

    fieldnames = [
        "rank",
        "direction",
        "event_label",
        "event_feature_key",
        "feature_key",
        "corr_speed",
        "abs_corr_speed",
        "behavior_std",
        "event_observation_count",
        "mean_activation",
        "std_activation",
        "frequency",
        "layer",
        "timestep",
        "feature",
        "layer_name",
        "timestep_key",
    ]
    for direction, selected in (
        ("fast", sorted(rows, key=lambda row: float(row["corr_speed"]), reverse=True)[:top_candidates]),
        ("slow", sorted(rows, key=lambda row: float(row["corr_speed"]))[:top_candidates]),
    ):
        for rank, row in enumerate(selected, start=1):
            row["rank"] = rank
            row["direction"] = direction
        with (output_dir / f"{direction}_feature_candidates_by_event.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(selected)


def main() -> None:
    patch_transformers_causal_mask_compat()
    patch_pi05_checkpoint_key_compat()
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.manual_event_bins <= 0:
        raise ValueError("--manual-event-bins must be positive")
    manual_event_windows = _parse_manual_event_windows(args.manual_event_windows)
    if bool(args.manual_event_name) != bool(manual_event_windows):
        raise ValueError("--manual-event-name and --manual-event-windows must be supplied together")
    if args.manual_event_only and not manual_event_windows:
        raise ValueError("--manual-event-only requires --manual-event-windows")

    device = resolve_device(args.device)
    args.resolved_device = device
    args.resolved_policy_dtype = resolve_policy_dtype(args.policy_dtype, device)

    cfg = _configure_train_config(args, episodes=None)
    cfg = _config_with_episodes(cfg, args.episodes)
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.policy.device = str(device)
    cfg.policy.dtype = args.resolved_policy_dtype
    cfg.policy.pretrained_path = Path(args.policy_path)
    cfg.policy.compile_model = False
    cfg.policy.gradient_checkpointing = False

    print(f"loading dataset {cfg.dataset.repo_id} episodes={_episode_summary(_parse_episode_ids(args.episodes))}", flush=True)
    dataset = make_dataset(cfg)
    phase_bins = _parse_phase_bins(args.phase_bins) if args.phase_bins.strip() else []
    episode_frame_max: dict[int, int] = {}
    if phase_bins:
        print(f"building episode frame ranges for phase bins={phase_bins}", flush=True)
        episode_frame_max = _episode_frame_max_from_dataset(dataset)
        if not episode_frame_max:
            raise RuntimeError("Could not infer episode frame ranges for phase-conditioned behavior stats")
        print(f"episode frame max: {dict(sorted(episode_frame_max.items()))}", flush=True)
    dataloader = _make_dataloader(cfg, dataset)
    planned_batches = len(dataloader) if args.max_batches is None else min(args.max_batches, len(dataloader))
    print(
        f"behavior feature collection frames={len(dataset)} batches={len(dataloader)} planned_batches={planned_batches} "
        f"batch_size={args.batch_size}",
        flush=True,
    )
    if args.plan_only:
        return

    print("loading preprocessor/tokenizer", flush=True)
    preprocessor = _make_preprocessor(cfg, dataset, args.policy_path)
    print("loading frozen Pi0.5 policy weights", flush=True)
    policy = make_policy(cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    _freeze_policy(policy)
    action_dim = int(policy.config.output_features[ACTION].shape[0])

    print(f"loading transcoders from {args.checkpoint}", flush=True)
    transcoders = _load_transcoders(args.checkpoint, device=device)
    d_features = {transcoder.config.latent_dim for transcoder in transcoders.values()}
    if len(d_features) != 1:
        raise ValueError(f"Expected all transcoders to share latent dim, got {sorted(d_features)}")
    layer_indices = {name: int(name.split(".layers.", 1)[1].split(".", 1)[0]) for name in transcoders}
    layer_names = sorted(transcoders, key=lambda name: layer_indices[name])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    collector = BehaviorFeatureCollector(
        layer_names=layer_names,
        layer_indices=layer_indices,
        d_features=d_features.pop(),
        top_k=args.top_k,
        firing_threshold=args.firing_threshold,
        top_m_active=args.top_m_active,
        observations_path=args.output_dir / "observations.jsonl",
        camera_keys=dataset.meta.camera_keys,
        episode_frame_max=episode_frame_max,
        phase_bins=phase_bins,
        manual_event_name=args.manual_event_name,
        manual_event_windows=manual_event_windows,
        manual_event_bins=args.manual_event_bins,
    )
    context = Pi05TranscoderContext(
        mode="probe",
        capture_records=False,
        capture_latents=True,
        latent_top_k=0,
        save_full_latents=False,
        latent_callback=collector.observe_latent,
        store_latent_summaries=False,
    )
    install_pi05_action_expert_wrappers(policy, context=context, transcoders=transcoders, mode="probe")

    metrics_path = args.output_dir / "collection_metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    started_at = time.time()
    iterator = tqdm(
        enumerate(dataloader, start=1),
        total=planned_batches,
        desc="behavior feature feed-forwards",
        disable=args.no_progress,
    )
    try:
        for batch_index, raw_batch in iterator:
            if batch_index > planned_batches:
                break
            if args.manual_event_only:
                event_rows = _select_manual_event_rows(raw_batch, manual_event_windows)
                if event_rows.numel() == 0:
                    continue
                raw_batch = _subset_raw_batch(raw_batch, event_rows)
            raw_batch = _prepare_raw_batch(raw_batch, dataset.meta.camera_keys)
            batch_size = collector.begin_batch(raw_batch)
            batch = preprocessor(raw_batch)
            context.clear_records()
            torch.manual_seed(args.seed + batch_index)
            with torch.no_grad():
                actions = policy.predict_action_chunk(batch, num_steps=args.num_inference_steps)
            speed_mean, speed_sum, speed_max = _action_speed_metrics(actions.detach(), action_dim=action_dim)
            behavior_rows = [
                {
                    "speed_mean": float(speed_mean[row].detach().cpu()),
                    "speed_sum": float(speed_sum[row].detach().cpu()),
                    "speed_max": float(speed_max[row].detach().cpu()),
                    "action_dim": action_dim,
                }
                for row in range(int(actions.shape[0]))
            ]
            updated = collector.end_batch(speed_mean.detach().cpu(), behavior_rows)
            context.clear_records()
            row = {
                "batch_index": batch_index,
                "elapsed_s": time.time() - started_at,
                "observations": collector.observation_count,
                "batch_size": batch_size,
                "layers_updated": len(updated),
            }
            with metrics_path.open("a") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
            if args.no_progress:
                print(f"batch={batch_index} observations={collector.observation_count} layers_updated={len(updated)}", flush=True)
            else:
                iterator.set_postfix(observations=collector.observation_count, layers=len(updated))
    finally:
        collector.close()

    collector.save(
        args.output_dir,
        extra_config={
            "policy_path": args.policy_path,
            "checkpoint": str(args.checkpoint),
            "dataset_repo_id": cfg.dataset.repo_id,
            "episodes": _parse_episode_ids(args.episodes),
            "batch_size": args.batch_size,
            "planned_batches": planned_batches,
            "num_inference_steps": args.num_inference_steps,
            "seed": args.seed,
            "speed_metric": "mean_t_l2_first_3_action_dims",
            "action_dim": action_dim,
            "device": str(device),
            "policy_dtype": args.resolved_policy_dtype,
            "phase_bins": phase_bins,
            "episode_frame_max": episode_frame_max,
            "manual_event_name": args.manual_event_name,
            "manual_event_windows": manual_event_windows,
            "manual_event_bins": args.manual_event_bins,
            "manual_event_only": args.manual_event_only,
        },
    )
    _rank_behavior_features(args.output_dir, top_candidates=args.top_candidates)
    _rank_phase_behavior_features(args.output_dir, top_candidates=args.top_candidates)
    _rank_event_behavior_features(args.output_dir, top_candidates=args.top_candidates)
    print(f"saved behavior-linked feature artifacts to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

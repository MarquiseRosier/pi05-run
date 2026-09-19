"""Action Atlas slots filled by Pi0.5 transcoders instead of SAEs.

Atlas pipeline:
    activations -> SAE.encode -> z -> concept_id / ablate / steer -> SAE.decode

Ours:
    MLP x, t -> transcoder -> z -> same scoring / ablate / steer -> decoder
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor

from .atlas_concepts import get_concept_task_mapping, normalize_suite
from .transcoders import TimeConditionedTranscoder


def expert_mlp_layer_name(layer_index: int) -> str:
    return f"expert_mlp_L{int(layer_index):02d}"


def parse_layer_name(layer_name: str) -> int:
    text = layer_name.strip()
    if text.startswith("expert_mlp_L"):
        return int(text.rsplit("L", 1)[-1])
    if text.startswith("expert_L"):
        return int(text.rsplit("L", 1)[-1])
    if text.isdigit():
        return int(text)
    raise ValueError(f"Unrecognized layer name: {layer_name!r}")


def per_token_topk(latent: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """Return ``(values, indices)`` with the last dim reduced to ``k``."""
    if latent.ndim < 1:
        raise ValueError(f"Expected at least a feature vector, got {tuple(latent.shape)}")
    width = latent.shape[-1]
    top_k = min(max(0, int(k)), width)
    if top_k == 0:
        empty_shape = (*latent.shape[:-1], 0)
        return (
            latent.new_empty(empty_shape),
            torch.empty(empty_shape, dtype=torch.long, device=latent.device),
        )
    values, indices = torch.topk(latent, k=top_k, dim=-1)
    return values, indices


def flatten_token_rows(indices: Tensor, values: Tensor, timestep: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Collapse ``[..., k]`` token top-k into ``[n_tokens, k]`` plus one t per token."""
    if indices.shape != values.shape:
        raise ValueError(f"indices {tuple(indices.shape)} != values {tuple(values.shape)}")
    token_indices = indices.reshape(-1, indices.shape[-1]).detach().cpu().long()
    token_values = values.reshape(-1, values.shape[-1]).detach().cpu().float()
    n_tokens = token_indices.shape[0]
    time = timestep.detach().float().reshape(-1).cpu()
    if time.numel() == 1:
        token_time = time.expand(n_tokens).contiguous()
    elif time.numel() == n_tokens:
        token_time = time
    else:
        repeats = n_tokens // max(time.numel(), 1)
        if time.numel() * repeats != n_tokens:
            raise ValueError(f"Cannot broadcast timestep {tuple(time.shape)} to {n_tokens} tokens")
        token_time = time.repeat_interleave(repeats)
    return token_indices, token_values, token_time


def pack_sparse_features(
    *,
    layer_index: int,
    d_features: int,
    k: int,
    indices: Tensor,
    values: Tensor,
    timesteps: Tensor,
) -> dict[str, Any]:
    return {
        "dictionary": "transcoder",
        "layer": expert_mlp_layer_name(layer_index),
        "layer_index": int(layer_index),
        "d_features": int(d_features),
        "k": int(k),
        "n_tokens": int(indices.shape[0]),
        "indices": indices.to(dtype=torch.int32),
        "values": values.to(dtype=torch.float16),
        "timesteps": timesteps.to(dtype=torch.float32),
    }


def save_sparse_features(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_sparse_features(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def sparse_to_dense(payload: dict[str, Any], *, max_tokens: int | None = None) -> Tensor:
    """Materialize sparse top-k rows as a dense ``[n_tokens, d_features]`` matrix."""
    indices = payload["indices"].long()
    values = payload["values"].float()
    d_features = int(payload["d_features"])
    n_tokens = int(indices.shape[0])
    if max_tokens is not None and n_tokens > max_tokens:
        choice = torch.randperm(n_tokens)[:max_tokens]
        indices = indices[choice]
        values = values[choice]
        n_tokens = int(max_tokens)
    dense = torch.zeros((n_tokens, d_features), dtype=torch.float32)
    if indices.numel() == 0:
        return dense
    row = torch.arange(n_tokens).unsqueeze(1).expand_as(indices)
    dense[row, indices] = values
    return dense


def atlas_activation_path(root: Path, task_id: int, episode: int, layer_index: int) -> Path:
    return root / f"task{int(task_id)}" / f"ep{int(episode)}" / f"{expert_mlp_layer_name(layer_index)}.pt"


def discover_task_ids(activations_dir: Path) -> list[int]:
    found: list[int] = []
    for child in sorted(activations_dir.glob("task*")):
        if child.is_dir() and child.name[4:].isdigit():
            found.append(int(child.name[4:]))
    return found


def load_task_features(
    activations_dir: Path,
    layer_name: str,
    *,
    max_tokens_per_task: int = 50000,
) -> dict[int, Tensor]:
    layer_index = parse_layer_name(layer_name)
    task_features: dict[int, Tensor] = {}
    for task_id in discover_task_ids(activations_dir):
        chunks: list[Tensor] = []
        for episode_dir in sorted((activations_dir / f"task{task_id}").glob("ep*")):
            path = episode_dir / f"{expert_mlp_layer_name(layer_index)}.pt"
            if not path.exists():
                continue
            chunks.append(sparse_to_dense(load_sparse_features(path)))
        if not chunks:
            continue
        combined = torch.cat(chunks, dim=0)
        if combined.shape[0] > max_tokens_per_task:
            combined = combined[torch.randperm(combined.shape[0])[:max_tokens_per_task]]
        task_features[task_id] = combined
    return task_features


def compute_concept_scores(
    task_features: dict[int, Tensor],
    suite: str,
    *,
    top_k: int = 20,
    space: str = "lerobot",
) -> dict[str, Any]:
    """Cohen's d x frequency, matching Action Atlas ``concept_id.py``."""
    concept_mapping = get_concept_task_mapping(suite, space=space)
    if not concept_mapping:
        raise ValueError(f"No Atlas concept table for suite {suite!r}")
    if len(task_features) < 2:
        raise ValueError(
            f"Need features from at least 2 tasks for contrastive scoring, got {sorted(task_features)}"
        )

    results: dict[str, Any] = {}
    for concept_type, concepts in concept_mapping.items():
        concept_results: dict[str, Any] = {}
        for concept_name, info in concepts.items():
            in_ids = [task_id for task_id in info["tasks"] if task_id in task_features]
            out_ids = [task_id for task_id in task_features if task_id not in info["tasks"]]
            if not in_ids or not out_ids:
                continue
            in_features = torch.cat([task_features[task_id] for task_id in in_ids], dim=0)
            out_features = torch.cat([task_features[task_id] for task_id in out_ids], dim=0)
            in_mean = in_features.mean(dim=0)
            out_mean = out_features.mean(dim=0)
            in_std = in_features.std(dim=0).clamp(min=1e-8)
            out_std = out_features.std(dim=0).clamp(min=1e-8)
            pooled_std = torch.sqrt((in_std.square() + out_std.square()) / 2).clamp(min=1e-8)
            cohens_d = (in_mean - out_mean) / pooled_std
            freq = (in_features.abs() > 0).float().mean(dim=0)
            score = cohens_d * freq
            keep = min(int(top_k), int(score.numel()))
            top_vals, top_idx = torch.topk(score.abs(), keep)
            top_features = []
            for rank, idx in enumerate(top_idx.tolist()):
                top_features.append(
                    {
                        "rank": rank,
                        "feature_idx": int(idx),
                        "score": float(score[idx]),
                        "cohens_d": float(cohens_d[idx]),
                        "frequency": float(freq[idx]),
                    }
                )
            concept_results[concept_name] = {
                "tasks": list(info["tasks"]),
                "tasks_in": sorted(in_ids),
                "tasks_out": sorted(out_ids),
                "n_in_samples": int(in_features.shape[0]),
                "n_out_samples": int(out_features.shape[0]),
                "top_features": top_features,
            }
        results[concept_type] = concept_results
    return results


def parse_int_list(value: str | Iterable[int] | None) -> list[int]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    return [int(item) for item in value]


class TranscoderDictionary:
    """Atlas SAE-shaped encode/decode over one time-conditioned transcoder."""

    def __init__(self, transcoder: TimeConditionedTranscoder):
        self.transcoder = transcoder

    def encode(self, x: Tensor, timesteps: Tensor) -> Tensor:
        _y_hat, latent = self.transcoder(x, timesteps)
        return latent

    def decode(self, latent: Tensor) -> Tensor:
        return self.transcoder.decoder(latent)

    def intervene(
        self,
        x: Tensor,
        timesteps: Tensor,
        *,
        ablate_features: Iterable[int] | None = None,
        steer_features: Iterable[int] | None = None,
        steer_strength: float = 0.0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(y_hat, y_intervened, z)``."""
        y_hat, latent = self.transcoder(x, timesteps)
        modified = latent
        ablate = list(ablate_features or [])
        steer = list(steer_features or [])
        if ablate or (steer and steer_strength != 0.0):
            modified = latent.clone()
            for feature in ablate:
                if feature < modified.shape[-1]:
                    modified[..., feature] = 0
            if steer_strength != 0.0:
                for feature in steer:
                    if feature < modified.shape[-1]:
                        modified[..., feature] = modified[..., feature] * (1.0 + float(steer_strength))
        y_intervened = self.decode(modified) if modified is not latent else y_hat
        return y_hat, y_intervened, latent


def merge_sparse_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("No sparse records to merge")
    first = records[0]
    return pack_sparse_features(
        layer_index=int(first["layer_index"]),
        d_features=int(first["d_features"]),
        k=int(first["k"]),
        indices=torch.cat([item["indices"] for item in records], dim=0),
        values=torch.cat([item["values"] for item in records], dim=0),
        timesteps=torch.cat([item["timesteps"] for item in records], dim=0),
    )


def group_records_by_layer(records: Iterable[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[int(record["layer_index"])].append(record)
    return grouped


def normalize_suite_name(suite: str) -> str:
    return normalize_suite(suite)

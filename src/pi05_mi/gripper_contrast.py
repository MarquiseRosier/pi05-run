"""Gripper open/close contrast on sparse transcoder latents.

The functions here do not load Pi0.5. They turn per-frame gripper commands
into matched sets, rank sparse features, and score steered action chunks.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

GRIPPER_INDEX = 6
ACTION_DIM_LABELS = ["dx", "dy", "dz", "dRx", "dRy", "dRz", "grip"]
N_PROGRESS_BINS = 5


@dataclass(frozen=True)
class GripperConvention:
    """Thresholds that map the gripper action dimension onto open and close.

    ``open_is_high`` is decided by which action mode has the larger finger gap.
    Values between ``low_threshold`` and ``high_threshold`` stay unlabeled.
    """

    open_is_high: bool
    low_center: float
    high_center: float
    low_threshold: float
    high_threshold: float
    finger_gap_correlation: float
    low_mode_finger_gap: float
    high_mode_finger_gap: float
    n_frames: int

    def label(self, grip: float) -> str | None:
        if grip >= self.high_threshold:
            return "open" if self.open_is_high else "close"
        if grip <= self.low_threshold:
            return "close" if self.open_is_high else "open"
        return None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrameRecord:
    index: int
    episode_id: int
    task_id: int
    frame_in_episode: int
    episode_length: int
    progress_bin: int
    label: str
    split: str


def finger_gap(state: np.ndarray) -> np.ndarray:
    """Return a per-frame finger opening from the trailing gripper state dims."""
    arr = np.asarray(state, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[-1] >= 2:
        gap = arr[..., -2:].mean(axis=-1)
    else:
        gap = arr[..., -1]
    return np.asarray(gap, dtype=np.float64).reshape(-1)


def _modes(grip: np.ndarray) -> tuple[float, float, np.ndarray]:
    values = np.asarray(grip, dtype=np.float64).reshape(-1)
    if values.size < 2:
        raise ValueError("Need at least two gripper actions to find open and close modes")
    low = float(np.quantile(values, 0.2))
    high = float(np.quantile(values, 0.8))
    if high < low:
        low, high = high, low
    assignment = np.zeros(values.shape[0], dtype=bool)
    for _ in range(25):
        assignment = np.abs(values - high) < np.abs(values - low)
        if not assignment.any() or assignment.all():
            break
        low = float(values[~assignment].mean())
        high = float(values[assignment].mean())
        if high < low:
            low, high = high, low
            assignment = ~assignment
    return low, high, assignment


def infer_gripper_convention(
    grip: np.ndarray,
    state: np.ndarray,
    *,
    margin_fraction: float = 0.15,
) -> GripperConvention:
    """Label the two gripper-action modes using finger opening, not a fixed sign."""
    values = np.asarray(grip, dtype=np.float64).reshape(-1)
    gap = finger_gap(state)
    if gap.shape[0] != values.shape[0]:
        raise ValueError(f"Expected one finger-gap value per gripper action, got {gap.shape[0]} and {values.shape[0]}")
    low, high, high_mask = _modes(values)
    span = high - low
    if not np.isfinite(span) or span <= 1e-6:
        raise ValueError(
            "Gripper action dimension is not bimodal, so open and close cannot be separated. "
            f"Mode centers were {low:.6f} and {high:.6f}."
        )
    margin = float(margin_fraction) * span
    midpoint = 0.5 * (low + high)
    low_gap = float(gap[~high_mask].mean()) if (~high_mask).any() else float("nan")
    high_gap = float(gap[high_mask].mean()) if high_mask.any() else float("nan")
    if not np.isfinite(low_gap) or not np.isfinite(high_gap):
        raise ValueError("Could not measure finger opening for both gripper-action modes")
    if abs(high_gap - low_gap) <= 1e-8:
        raise ValueError(
            "Finger opening does not differ between the two gripper-action modes, "
            "so the open/close sign cannot be determined."
        )
    correlation = float(np.corrcoef(values, gap)[0, 1]) if values.size > 1 else float("nan")
    return GripperConvention(
        open_is_high=bool(high_gap > low_gap),
        low_center=low,
        high_center=high,
        low_threshold=midpoint - margin,
        high_threshold=midpoint + margin,
        finger_gap_correlation=correlation,
        low_mode_finger_gap=low_gap,
        high_mode_finger_gap=high_gap,
        n_frames=int(values.size),
    )


def progress_bin(frame_in_episode: int, episode_length: int, n_bins: int = N_PROGRESS_BINS) -> int:
    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive, got {n_bins}")
    if episode_length <= 0:
        raise ValueError(f"episode_length must be positive, got {episode_length}")
    position = float(frame_in_episode) / float(episode_length)
    return min(n_bins - 1, max(0, int(position * n_bins)))


def stable_window_label(grip_window: Sequence[float], convention: GripperConvention) -> str | None:
    """Return open or close when every step in the window shares that label."""
    labels = [convention.label(float(value)) for value in grip_window]
    if not labels or any(label is None for label in labels):
        return None
    if all(label == "open" for label in labels):
        return "open"
    if all(label == "close" for label in labels):
        return "close"
    return None


def split_episode_ids(
    episode_ids: Iterable[int],
    *,
    train_fraction: float = 0.7,
    seed: int = 0,
) -> tuple[set[int], set[int]]:
    """Hold out whole episodes so later frames cannot leak into the direction."""
    ids = sorted({int(episode) for episode in episode_ids})
    if len(ids) < 2:
        raise ValueError(f"Need at least two episodes to hold one out, got {ids}")
    if not 0.0 < float(train_fraction) < 1.0:
        raise ValueError(f"train_fraction must be between 0 and 1, got {train_fraction}")
    order = np.random.default_rng(seed).permutation(len(ids))
    n_train = int(round(len(ids) * float(train_fraction)))
    n_train = min(max(n_train, 1), len(ids) - 1)
    train = {ids[int(index)] for index in order[:n_train]}
    holdout = {ids[int(index)] for index in order[n_train:]}
    return train, holdout


def assign_splits(records: Sequence[FrameRecord], train_episodes: set[int], holdout_episodes: set[int]) -> list[FrameRecord]:
    assigned: list[FrameRecord] = []
    for record in records:
        if record.episode_id in train_episodes:
            split = "train"
        elif record.episode_id in holdout_episodes:
            split = "holdout"
        else:
            continue
        assigned.append(
            FrameRecord(
                index=record.index,
                episode_id=record.episode_id,
                task_id=record.task_id,
                frame_in_episode=record.frame_in_episode,
                episode_length=record.episode_length,
                progress_bin=record.progress_bin,
                label=record.label,
                split=split,
            )
        )
    return assigned


def cap_records(
    records: Sequence[FrameRecord],
    *,
    max_per_group: int,
    max_holdout_per_label: int,
    seed: int = 0,
) -> list[FrameRecord]:
    """Subsample train rows inside each task/progress/label cell, and cap holdout."""
    rng = np.random.default_rng(seed)
    grouped: dict[tuple[int, int, str], list[FrameRecord]] = {}
    holdout: dict[str, list[FrameRecord]] = {"open": [], "close": []}
    for record in records:
        if record.split == "train":
            grouped.setdefault((record.task_id, record.progress_bin, record.label), []).append(record)
        elif record.split == "holdout" and record.label in holdout:
            holdout[record.label].append(record)
    kept: list[FrameRecord] = []
    for key in sorted(grouped):
        rows = list(grouped[key])
        rng.shuffle(rows)
        kept.extend(rows[: max(0, int(max_per_group))])
    for label in ("open", "close"):
        rows = list(holdout[label])
        rng.shuffle(rows)
        kept.extend(rows[: max(0, int(max_holdout_per_label))])
    return kept


def pair_records(records: Sequence[FrameRecord], *, seed: int = 0) -> list[tuple[FrameRecord, FrameRecord]]:
    """Pair train open/close rows that share a task and a progress bin."""
    rng = np.random.default_rng(seed)
    groups: dict[tuple[int, int], dict[str, list[FrameRecord]]] = {}
    for record in records:
        if record.split != "train":
            continue
        groups.setdefault((record.task_id, record.progress_bin), {"open": [], "close": []})
        if record.label in {"open", "close"}:
            groups[(record.task_id, record.progress_bin)][record.label].append(record)
    pairs: list[tuple[FrameRecord, FrameRecord]] = []
    for key in sorted(groups):
        opens = list(groups[key]["open"])
        closes = list(groups[key]["close"])
        rng.shuffle(opens)
        rng.shuffle(closes)
        pairs.extend(zip(opens, closes, strict=False))
    return pairs


def quantize_tau(tau: float) -> float:
    return round(float(tau), 5)


def mean_over_tokens(latent: np.ndarray, token_indices: Sequence[int]) -> np.ndarray:
    """Average selected action tokens of one STC latent into a feature vector."""
    arr = np.asarray(latent, dtype=np.float32)
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(f"Expected a single-example latent, got shape {arr.shape}")
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected latent shape [tokens, features], got {arr.shape}")
    indices = np.asarray(list(token_indices), dtype=np.int64)
    if indices.size == 0:
        raise ValueError("token_indices must select at least one action token")
    if int(indices.min()) < 0 or int(indices.max()) >= arr.shape[0]:
        raise IndexError(f"Token indices {indices.tolist()} do not fit latent length {arr.shape[0]}")
    return arr[indices].mean(axis=0)


class LatentBank:
    """Float16 store of reduced STC vectors, keyed by example, layer, and tau."""

    def __init__(self) -> None:
        self.vectors: dict[tuple[int, int, float], np.ndarray] = {}

    def add(self, example_id: int, layer: int, tau: float, vector: np.ndarray) -> None:
        self.vectors[(int(example_id), int(layer), quantize_tau(tau))] = np.asarray(vector, dtype=np.float16)

    def cells(self) -> list[tuple[int, float]]:
        return sorted({(layer, tau) for _example, layer, tau in self.vectors})

    def stack(self, example_ids: Sequence[int], layer: int, tau: float) -> np.ndarray:
        key_tau = quantize_tau(tau)
        rows = []
        for example_id in example_ids:
            key = (int(example_id), int(layer), key_tau)
            if key not in self.vectors:
                raise KeyError(f"Missing latent for example {example_id} layer {layer} tau {key_tau}")
            rows.append(self.vectors[key].astype(np.float32, copy=False))
        if not rows:
            raise ValueError("example_ids is empty")
        return np.stack(rows, axis=0)


def firing_frequency(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return np.zeros((vectors.shape[-1],), dtype=np.float64)
    return (vectors > 0).mean(axis=0).astype(np.float64)


def sign_consistency(open_vectors: np.ndarray, close_vectors: np.ndarray) -> np.ndarray:
    """Fraction of pairs whose open-minus-close sign matches the mean pair delta.

    A pair with zero difference does not count as agreement. One extreme
    activation therefore cannot look consistent just because the other pairs
    were ignored.
    """
    if open_vectors.shape != close_vectors.shape:
        raise ValueError(f"Pair shapes differ: {open_vectors.shape} vs {close_vectors.shape}")
    if open_vectors.shape[0] == 0:
        width = open_vectors.shape[-1] if open_vectors.ndim == 2 else 0
        return np.zeros((width,), dtype=np.float64)
    delta = open_vectors.astype(np.float64) - close_vectors.astype(np.float64)
    mean = delta.mean(axis=0)
    agree = (np.sign(delta) == np.sign(mean)) & (np.sign(mean) != 0)
    return agree.mean(axis=0).astype(np.float64)


def contrastive_direction(open_vectors: np.ndarray, close_vectors: np.ndarray) -> np.ndarray:
    if open_vectors.shape != close_vectors.shape:
        raise ValueError(f"Pair shapes differ: {open_vectors.shape} vs {close_vectors.shape}")
    if open_vectors.shape[0] == 0:
        raise ValueError("Need at least one open/close pair to build a direction")
    return (open_vectors.astype(np.float64) - close_vectors.astype(np.float64)).mean(axis=0)


def feature_scores(
    mean_open: np.ndarray,
    mean_close: np.ndarray,
    open_frequency: np.ndarray,
    close_frequency: np.ndarray,
    consistency: np.ndarray,
) -> np.ndarray:
    """Rank by contrast magnitude, pair agreement, and how often the feature fires.

    The frequency weight is the larger of the two class rates. Using the
    smaller rate would score a feature at zero when it is silent on one
    behavior, which is a valid open-versus-close contrast for a sparse code.
    """
    delta = mean_open - mean_close
    frequency = np.maximum(open_frequency, close_frequency)
    return np.abs(delta) * consistency * frequency


def cell_statistics(
    open_vectors: np.ndarray,
    close_vectors: np.ndarray,
    paired_open: np.ndarray,
    paired_close: np.ndarray,
) -> dict[str, np.ndarray]:
    mean_open = open_vectors.astype(np.float64).mean(axis=0)
    mean_close = close_vectors.astype(np.float64).mean(axis=0)
    consistency = sign_consistency(paired_open, paired_close)
    direction = contrastive_direction(paired_open, paired_close)
    open_frequency = firing_frequency(open_vectors)
    close_frequency = firing_frequency(close_vectors)
    delta = mean_open - mean_close
    score = feature_scores(mean_open, mean_close, open_frequency, close_frequency, consistency)
    return {
        "mean_open": mean_open,
        "mean_close": mean_close,
        "delta": delta,
        "open_firing_frequency": open_frequency,
        "close_firing_frequency": close_frequency,
        "sign_consistency": consistency,
        "score": score,
        "direction": direction,
    }


def feature_rows(layer: int, tau: float, stats: dict[str, np.ndarray], indices: Sequence[int] | None = None) -> list[dict[str, Any]]:
    if indices is None:
        indices = range(int(stats["score"].shape[0]))
    rows: list[dict[str, Any]] = []
    for index in indices:
        feature_id = int(index)
        rows.append(
            {
                "layer": int(layer),
                "tau": quantize_tau(tau),
                "feature_id": feature_id,
                "mean_open": float(stats["mean_open"][feature_id]),
                "mean_close": float(stats["mean_close"][feature_id]),
                "delta": float(stats["delta"][feature_id]),
                "open_firing_frequency": float(stats["open_firing_frequency"][feature_id]),
                "close_firing_frequency": float(stats["close_firing_frequency"][feature_id]),
                "sign_consistency": float(stats["sign_consistency"][feature_id]),
                "score": float(stats["score"][feature_id]),
            }
        )
    return rows


def sparsify_direction(
    direction: np.ndarray,
    consistency: np.ndarray,
    *,
    top_k: int = 32,
    min_consistency: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the strongest consistent coordinates and match the full vector's L2 norm."""
    vector = np.asarray(direction, dtype=np.float64)
    agree = np.asarray(consistency, dtype=np.float64)
    if vector.shape != agree.shape:
        raise ValueError(f"direction {vector.shape} and consistency {agree.shape} differ")
    eligible = np.flatnonzero(agree >= float(min_consistency))
    if eligible.size == 0:
        eligible = np.arange(vector.shape[0])
    order = eligible[np.argsort(-np.abs(vector[eligible]))]
    keep = order[: max(0, int(top_k))]
    sparse = np.zeros_like(vector)
    sparse[keep] = vector[keep]
    full_norm = float(np.linalg.norm(vector))
    sparse_norm = float(np.linalg.norm(sparse))
    if full_norm > 0.0 and sparse_norm > 0.0:
        sparse *= full_norm / sparse_norm
    return sparse, keep.astype(np.int64)


def permute_direction(direction: np.ndarray, seed: int) -> np.ndarray:
    """Shuffle feature identity while keeping the same values and sparsity."""
    vector = np.asarray(direction, dtype=np.float64).copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(vector)
    return vector


def select_best_cell(cell_scores: Sequence[tuple[int, float, float]]) -> tuple[int, float]:
    """Pick the ``(layer, tau)`` that contains the highest feature score."""
    if not cell_scores:
        raise ValueError("No layer/tau cells were scored")
    layer, tau, _score = max(cell_scores, key=lambda item: (item[2], -item[0], -item[1]))
    return int(layer), quantize_tau(tau)


def action_effect(
    baseline: np.ndarray,
    steered: np.ndarray,
    *,
    gripper_index: int = GRIPPER_INDEX,
    horizon: int = 5,
) -> dict[str, Any]:
    """Compare postprocessed action chunks on the gripper dim and the other dims."""
    base = _action_matrix(baseline)[:horizon]
    done = _action_matrix(steered)[:horizon]
    width = min(base.shape[1], done.shape[1], len(ACTION_DIM_LABELS))
    if gripper_index >= width:
        raise ValueError(f"Gripper index {gripper_index} does not fit action width {width}")
    base = base[:, :width]
    done = done[:, :width]
    signed = (done - base).mean(axis=0)
    absolute = np.abs(done - base).mean(axis=0)
    other = np.delete(absolute, gripper_index)
    result: dict[str, Any] = {
        "baseline_gripper": float(base[:, gripper_index].mean()),
        "steered_gripper": float(done[:, gripper_index].mean()),
        "delta_gripper": float(signed[gripper_index]),
        "other_dims_mean_abs": float(other.mean()) if other.size else 0.0,
    }
    for index, label in enumerate(ACTION_DIM_LABELS[:width]):
        result[f"delta_{label}"] = float(signed[index])
        result[f"abs_{label}"] = float(absolute[index])
    return result


def _action_matrix(actions: np.ndarray) -> np.ndarray:
    arr = np.asarray(actions, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected an action chunk, got shape {arr.shape}")
    return arr


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_figures(
    output_dir: Path,
    *,
    ranking_rows: Sequence[dict[str, Any]],
    steering_rows: Sequence[dict[str, Any]],
    grip_values: np.ndarray,
    grip_labels: Sequence[str | None],
    convention: GripperConvention,
) -> list[Path]:
    """Write the five comparison figures used to read the experiment."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    written = [
        _plot_histogram(output_dir / "gripper_action_histogram.png", grip_values, grip_labels, convention, plt),
        _plot_top_features(output_dir / "top_feature_deltas.png", ranking_rows, plt),
        _plot_alpha(output_dir / "gripper_effect_vs_alpha.png", steering_rows, plt),
        _plot_controls(output_dir / "gripper_vs_random.png", steering_rows, plt),
        _plot_dimensions(output_dir / "action_dimension_effects.png", steering_rows, plt),
    ]
    return written


def _plot_histogram(path: Path, grip_values: np.ndarray, grip_labels: Sequence[str | None], convention: GripperConvention, plt: Any) -> Path:
    values = np.asarray(grip_values, dtype=np.float64).reshape(-1)
    labels = list(grip_labels)
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, color in (("open", "#2a9d8f"), ("close", "#e76f51"), (None, "#adb5bd")):
        mask = [item == label for item in labels]
        selected = values[np.asarray(mask, dtype=bool)] if mask else np.empty(0)
        if selected.size:
            ax.hist(selected, bins=40, alpha=0.7, label="ambiguous" if label is None else label, color=color)
    ax.axvline(convention.low_threshold, color="black", linestyle="--", linewidth=1)
    ax.axvline(convention.high_threshold, color="black", linestyle="--", linewidth=1)
    side = "high action = open" if convention.open_is_high else "high action = close"
    ax.set_title(f"Gripper action dim {GRIPPER_INDEX} ({side})")
    ax.set_xlabel("action[:, 6]")
    ax.set_ylabel("frames")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _plot_top_features(path: Path, ranking_rows: Sequence[dict[str, Any]], plt: Any) -> Path:
    rows = sorted(ranking_rows, key=lambda row: float(row["score"]), reverse=True)[:20]
    fig, ax = plt.subplots(figsize=(8, 6))
    if rows:
        labels = [f"L{int(row['layer'])} t={float(row['tau']):.1f} f={int(row['feature_id'])}" for row in rows]
        deltas = [float(row["delta"]) for row in rows]
        colors = ["#2a9d8f" if value >= 0 else "#e76f51" for value in deltas]
        ax.barh(range(len(rows))[::-1], deltas[::-1], color=colors[::-1])
        ax.set_yticks(range(len(rows))[::-1], labels[::-1], fontsize=8)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("mean open - mean close")
    ax.set_title("Highest-scoring gripper features")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _mean_by(rows: Sequence[dict[str, Any]], *, kind: str, control: str) -> tuple[list[float], list[float]]:
    bucket: dict[float, list[float]] = {}
    for row in rows:
        if row.get("direction_kind") != kind or row.get("control") != control:
            continue
        alpha = float(row["alpha"])
        bucket.setdefault(alpha, []).append(float(row["delta_gripper"]))
    alphas = sorted(bucket)
    means = [float(np.mean(bucket[alpha])) for alpha in alphas]
    return alphas, means


def _plot_alpha(path: Path, steering_rows: Sequence[dict[str, Any]], plt: Any) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    for kind, control, style, label in (
        ("full", "plus", "-", "+v full"),
        ("full", "minus", "--", "-v full"),
        ("topk", "plus", "-", "+v top-k"),
        ("topk", "minus", "--", "-v top-k"),
    ):
        alphas, means = _mean_by(steering_rows, kind=kind, control=control)
        if alphas:
            ax.plot(alphas, means, linestyle=style, marker="o", label=label)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("alpha")
    ax.set_ylabel("mean delta gripper")
    ax.set_title("Gripper steering versus alpha")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _plot_controls(path: Path, steering_rows: Sequence[dict[str, Any]], plt: Any) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    for control, label in (("plus", "+v"), ("minus", "-v"), ("random", "random")):
        alphas, means = _mean_by(steering_rows, kind="full", control=control)
        if alphas:
            ax.plot(alphas, means, marker="o", label=label)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("alpha")
    ax.set_ylabel("mean delta gripper")
    ax.set_title("Full direction versus matched random features")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _plot_dimensions(path: Path, steering_rows: Sequence[dict[str, Any]], plt: Any) -> Path:
    positive = [row for row in steering_rows if float(row.get("alpha", 0.0)) > 0.0]
    alphas = sorted({float(row["alpha"]) for row in positive})
    target = alphas[len(alphas) // 2] if alphas else None
    fig, ax = plt.subplots(figsize=(8, 4))
    if target is not None:
        width = 0.35
        positions = np.arange(len(ACTION_DIM_LABELS))
        for offset, control, label in ((-width / 2, "plus", "+v"), (width / 2, "random", "random")):
            chosen = [
                row
                for row in steering_rows
                if row.get("direction_kind") == "full"
                and row.get("control") == control
                and abs(float(row["alpha"]) - target) < 1e-6
            ]
            if not chosen:
                continue
            heights = []
            for name in ACTION_DIM_LABELS:
                key = f"abs_{name}"
                heights.append(float(np.mean([float(row[key]) for row in chosen if key in row])))
            ax.bar(positions + offset, heights, width=width, label=label)
        ax.set_xticks(positions, ACTION_DIM_LABELS)
        ax.set_title(f"Mean absolute action change at alpha={target:g}")
    ax.set_ylabel("mean |delta|")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def selection_summary(
    *,
    convention: GripperConvention,
    n_frames: int,
    n_stable_open: int,
    n_stable_close: int,
    n_pairs: int,
    n_train: int,
    n_holdout: int,
    train_episodes: Iterable[int],
    holdout_episodes: Iterable[int],
    horizon: int,
) -> dict[str, Any]:
    return {
        "gripper_index": GRIPPER_INDEX,
        "action_dim_labels": ACTION_DIM_LABELS,
        "horizon": int(horizon),
        "n_frames": int(n_frames),
        "n_stable_open": int(n_stable_open),
        "n_stable_close": int(n_stable_close),
        "n_pairs": int(n_pairs),
        "n_train_probed": int(n_train),
        "n_holdout_probed": int(n_holdout),
        "train_episodes": sorted(int(episode) for episode in train_episodes),
        "holdout_episodes": sorted(int(episode) for episode in holdout_episodes),
        "convention": convention.to_dict(),
    }

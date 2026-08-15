#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import html
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import cv2
    import numpy as np
except ModuleNotFoundError:
    if os.environ.get("PI05_REPORT_NO_UV_REEXEC") != "1" and shutil.which("uv"):
        env = os.environ.copy()
        env["PI05_REPORT_NO_UV_REEXEC"] = "1"
        os.execvpe("uv", ["uv", "run", "python", __file__, *sys.argv[1:]], env)
    raise

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_pi05_analysis_video import (  # noqa: E402
    BG,
    GRID,
    MUTED,
    PANEL_BG,
    TEXT,
    build_scales,
    find_task_video,
    group_expert_events,
    letterbox,
    parse_metrics_from_log,
    read_capture,
    read_json,
    resolve_run,
    sequential_color,
    signed_color,
    trim_task,
)


FONT = cv2.FONT_HERSHEY_SIMPLEX
BG = (23, 13, 9)
PANEL_BG = (42, 23, 15)
GRID = (102, 82, 70)
TEXT = (249, 245, 241)
MUTED = (214, 197, 185)
ACCENT = (248, 189, 56)
YELLOW = (78, 196, 245)
GREEN = (128, 222, 74)
RED = (113, 113, 248)
ORANGE = (60, 146, 251)
SUBGRID = (41, 28, 21)
ROW_BG = (32, 20, 12)
ROW_BG_ALT = (39, 24, 16)
ACTION_DIM_LABELS = ["dx", "dy", "dz", "dRx", "dRy", "dRz", "grip"]


def put_text(
    img: np.ndarray,
    value: str,
    xy: tuple[int, int],
    *,
    scale: float = 0.52,
    color: tuple[int, int, int] = TEXT,
    thickness: int = 1,
) -> None:
    cv2.putText(img, value, xy, FONT, scale, color, thickness, cv2.LINE_AA)


def fit_text(value: str, max_width: int, *, scale: float = 0.5, thickness: int = 1) -> str:
    if cv2.getTextSize(value, FONT, scale, thickness)[0][0] <= max_width:
        return value
    suffix = "..."
    while value and cv2.getTextSize(value + suffix, FONT, scale, thickness)[0][0] > max_width:
        value = value[:-1]
    return value.rstrip() + suffix


def draw_frame(img: np.ndarray, x: int, y: int, w: int, h: int, color: tuple[int, int, int] = GRID) -> None:
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)


def load_capture_image(path: Path, width: int, height: int) -> np.ndarray:
    if path.suffix == ".npy":
        arr = np.load(path)
        frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    else:
        frame = cv2.imread(str(path))
    if frame is None:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = PANEL_BG
        put_text(frame, "missing", (18, 40), color=MUTED)
        return frame
    return letterbox(frame, width, height)


def video_frame_at(video_path: Path, frame_idx: int, width: int, height: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = PANEL_BG
        put_text(frame, f"no frame {frame_idx}", (18, 40), color=MUTED)
        return frame
    return letterbox(frame, width, height)


def sample_video_frames(
    video_path: Path,
    frame_indices: list[int],
    width: int,
    height: int,
) -> dict[int, np.ndarray]:
    targets = sorted(set(max(0, int(idx)) for idx in frame_indices))
    if not targets:
        return {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    frames: dict[int, np.ndarray] = {}
    next_target_idx = 0
    frame_idx = 0
    while next_target_idx < len(targets):
        ok, frame = cap.read()
        if not ok:
            break
        target = targets[next_target_idx]
        if frame_idx == target:
            frames[target] = letterbox(frame, width, height)
            next_target_idx += 1
        frame_idx += 1
    cap.release()
    for target in targets:
        if target not in frames:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:] = PANEL_BG
            put_text(frame, f"no frame {target}", (18, 40), color=MUTED)
            frames[target] = frame
    return frames


def chunks_with_capture(capture: dict[str, Any]) -> list[int]:
    chunks = set(capture["chunk_meta"].keys()) | set(capture["images"].keys()) | set(capture["actions"].keys())
    chunks |= set(capture["activations"].keys())
    return sorted(int(chunk) for chunk in chunks)


def expert_summary_for_chunk(capture: dict[str, Any], chunk: int) -> dict[str, Any]:
    events = capture["activations"].get(chunk, {}).get("expert", [])
    grouped = group_expert_events(events)
    abs_mat = grouped["abs"]
    layers = grouped["layers"]
    if abs_mat.size == 0:
        return {"layers": [], "abs": abs_mat, "mean": np.asarray([]), "final": np.asarray([]), "delta": np.asarray([])}
    initial = abs_mat[:, 0]
    final = abs_mat[:, -1]
    return {
        "layers": layers,
        "abs": abs_mat,
        "mean": np.nanmean(abs_mat, axis=1),
        "final": final,
        "delta": final - initial,
    }


def family_layer_matrix(
    capture: dict[str, Any],
    family: str,
    chunks: list[int],
    *,
    expert_metric: str = "mean",
) -> tuple[list[int], np.ndarray]:
    layer_values: dict[int, dict[int, float]] = defaultdict(dict)
    for chunk in chunks:
        if family == "expert":
            summary = expert_summary_for_chunk(capture, chunk)
            layers = summary["layers"]
            values = summary.get(expert_metric)
            if values is None or len(layers) == 0:
                continue
            for layer, value in zip(layers, values, strict=False):
                layer_values[int(layer)][chunk] = float(value)
            continue

        by_layer: dict[int, list[float]] = defaultdict(list)
        for event in capture["activations"].get(chunk, {}).get(family, []):
            layer = event.get("layer")
            if layer is None:
                continue
            by_layer[int(layer)].append(float(event.get("abs_mean", np.nan)))
        for layer, values in by_layer.items():
            arr = np.asarray(values, dtype=np.float32)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                layer_values[layer][chunk] = float(arr.mean())

    layers = sorted(layer_values)
    mat = np.full((len(layers), len(chunks)), np.nan, dtype=np.float32)
    chunk_to_col = {chunk: idx for idx, chunk in enumerate(chunks)}
    for row, layer in enumerate(layers):
        for chunk, value in layer_values[layer].items():
            mat[row, chunk_to_col[chunk]] = value
    return layers, mat


def safe_percentile(values: np.ndarray, pct: float, default: float = 1.0) -> float:
    arr = values.astype(np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return default
    value = float(np.percentile(arr, pct))
    if not math.isfinite(value) or value <= 0:
        return default
    return value


def matrix_to_json(matrix: np.ndarray) -> list[list[float | None]]:
    arr = np.asarray(matrix, dtype=np.float32)
    rows: list[list[float | None]] = []
    for row in arr:
        rows.append([None if not np.isfinite(value) else round(float(value), 6) for value in row])
    return rows


def draw_axis_ticks(
    img: np.ndarray,
    plot: tuple[int, int, int, int],
    *,
    x_labels: list[tuple[float, str]] | None = None,
    y_labels: list[tuple[float, str]] | None = None,
) -> None:
    x, y, w, h = plot
    for frac, label in x_labels or []:
        px = x + int(frac * w)
        cv2.line(img, (px, y), (px, y + h), SUBGRID, 1)
        put_text(img, label, (px - 8, y + h + 18), scale=0.36, color=MUTED)
    for frac, label in y_labels or []:
        py = y + int(frac * h)
        cv2.line(img, (x, py), (x + w, py), SUBGRID, 1)
        put_text(img, label, (x - 35, py + 4), scale=0.36, color=MUTED)


def draw_heatmap_panel(
    title: str,
    matrix: np.ndarray,
    *,
    width: int,
    height: int,
    cap: float,
    signed: bool = False,
    log_scale: bool = False,
    x_labels: list[tuple[float, str]] | None = None,
    y_labels: list[tuple[float, str]] | None = None,
) -> np.ndarray:
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = BG
    put_text(panel, title, (18, 34), scale=0.66, color=TEXT, thickness=2)
    plot = (58, 58, width - 78, height - 98)
    x, y, w, h = plot
    if matrix.size:
        if signed:
            heat = signed_color(np.nan_to_num(matrix, nan=0.0), w, h, cap)
        else:
            heat = sequential_color(matrix, w, h, cap, log_scale=log_scale)
        panel[y : y + h, x : x + w] = heat
    else:
        put_text(panel, "no data", (x + 18, y + 42), color=MUTED)
    draw_frame(panel, x, y, w, h)
    draw_axis_ticks(panel, plot, x_labels=x_labels, y_labels=y_labels)
    return panel


def action_window_for_chunk(actions: np.ndarray | None, n_action_steps: int) -> np.ndarray:
    if actions is None:
        return np.empty((0, len(ACTION_DIM_LABELS)), dtype=np.float32)
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return np.empty((0, len(ACTION_DIM_LABELS)), dtype=np.float32)
    return arr[:n_action_steps, : len(ACTION_DIM_LABELS)]


def action_feature_series(
    capture: dict[str, Any],
    chunks: list[int],
    n_action_steps: int,
) -> tuple[list[str], np.ndarray]:
    names = [f"mean {label}" for label in ACTION_DIM_LABELS] + ["step L2"]
    values = np.full((len(names), len(chunks)), np.nan, dtype=np.float32)
    for col, chunk in enumerate(chunks):
        window = action_window_for_chunk(capture["actions"].get(chunk), n_action_steps)
        if window.size == 0:
            continue
        means = np.nanmean(window, axis=0)
        values[: len(ACTION_DIM_LABELS), col] = means
        values[-1, col] = float(np.nanmean(np.linalg.norm(window, axis=1)))
    return names, values


def draw_polyline(
    img: np.ndarray,
    values: np.ndarray,
    plot: tuple[int, int, int, int],
    *,
    color: tuple[int, int, int],
    cap: float | None = None,
    zero_centered: bool = False,
) -> None:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return
    x, y, w, h = plot
    if cap is None:
        ymin = float(arr.min())
        ymax = float(arr.max())
        if abs(ymax - ymin) < 1e-6:
            ymin -= 0.5
            ymax += 0.5
    elif zero_centered:
        ymin = -cap
        ymax = cap
    else:
        ymin = 0.0
        ymax = cap
    points: list[tuple[int, int]] = []
    full = np.asarray(values, dtype=np.float32)
    for idx, value in enumerate(full):
        if not np.isfinite(value):
            continue
        px = x + int(idx / max(1, len(full) - 1) * w)
        py = y + h - int((float(value) - ymin) / max(1e-6, ymax - ymin) * h)
        points.append((px, py))
    for left, right in zip(points, points[1:], strict=False):
        cv2.line(img, left, right, color, 2)


def draw_action_tile(
    canvas: np.ndarray,
    actions: np.ndarray | None,
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    n_action_steps: int,
    cap: float,
) -> None:
    put_text(canvas, "executed actions: dim x step", (x, y + 14), scale=0.36, color=MUTED)
    window = action_window_for_chunk(actions, n_action_steps)
    label_w = 44
    plot_x = x + label_w
    plot_y = y + 26
    plot_w = w - label_w - 4
    plot_h = 104
    if window.size:
        heat = signed_color(window.T, plot_w, plot_h, cap)
        canvas[plot_y : plot_y + plot_h, plot_x : plot_x + plot_w] = heat
        for idx, label in enumerate(ACTION_DIM_LABELS[: window.shape[1]]):
            row_y = plot_y + int((idx + 0.58) / window.shape[1] * plot_h)
            put_text(canvas, label, (x, row_y), scale=0.34, color=TEXT)
        for step in range(window.shape[0]):
            sx = plot_x + int((step + 0.5) / window.shape[0] * plot_w)
            cv2.line(canvas, (sx, plot_y), (sx, plot_y + plot_h), SUBGRID, 1)
        put_text(canvas, "0", (plot_x, plot_y + plot_h + 17), scale=0.32, color=MUTED)
        put_text(canvas, str(window.shape[0] - 1), (plot_x + plot_w - 12, plot_y + plot_h + 17), scale=0.32, color=MUTED)
        put_text(canvas, "blue=-  red=+", (plot_x + 72, plot_y + plot_h + 17), scale=0.32, color=MUTED)
    else:
        put_text(canvas, "no action capture", (plot_x + 16, plot_y + 48), scale=0.42, color=MUTED)
    draw_frame(canvas, plot_x, plot_y, plot_w, plot_h)


def draw_action_summary_tile(
    canvas: np.ndarray,
    actions: np.ndarray | None,
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    n_action_steps: int,
    cap: float,
) -> None:
    put_text(canvas, "action summary", (x, y + 14), scale=0.38, color=MUTED)
    window = action_window_for_chunk(actions, n_action_steps)
    if window.size == 0:
        put_text(canvas, "no action capture", (x + 10, y + 54), scale=0.42, color=MUTED)
        return

    means = np.nanmean(window, axis=0)
    norms = np.linalg.norm(window, axis=1)
    order = np.argsort(-np.abs(means))[:3]
    text_y = y + 38
    for idx in order:
        label = ACTION_DIM_LABELS[idx]
        value = float(means[idx])
        signed = f"{value:+.3f}"
        put_text(canvas, f"{label:>4} {signed}", (x, text_y), scale=0.38, color=TEXT)
        bar_x = x + 92
        bar_w = max(1, int(min(1.0, abs(value) / max(cap, 1e-6)) * (w - 108)))
        color = RED if value >= 0 else ACCENT
        cv2.rectangle(canvas, (bar_x, text_y - 13), (bar_x + bar_w, text_y - 4), color, -1)
        text_y += 24

    plot = (x + 8, y + h - 52, w - 20, 34)
    draw_frame(canvas, *plot)
    draw_polyline(canvas, norms, plot, color=YELLOW, cap=max(cap, float(np.nanmax(norms))), zero_centered=False)
    put_text(canvas, "step action L2", (x + 8, y + h - 60), scale=0.32, color=MUTED)


def draw_layer_delta_tile(
    canvas: np.ndarray,
    summary: dict[str, Any],
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    cap: float,
) -> None:
    put_text(canvas, "expert Δ by layer", (x, y + 14), scale=0.38, color=MUTED)
    delta = np.asarray(summary.get("delta", []), dtype=np.float32)
    layers = summary.get("layers", [])
    plot_x = x + 42
    plot_y = y + 26
    plot_w = w - 48
    plot_h = h - 54
    if delta.size:
        heat = signed_color(delta[:, None], plot_w, plot_h, cap)
        canvas[plot_y : plot_y + plot_h, plot_x : plot_x + plot_w] = heat
        for idx, layer in enumerate(layers):
            if int(layer) % 3 == 0:
                row_y = plot_y + int((idx + 0.58) / len(layers) * plot_h)
                put_text(canvas, f"L{layer}", (x, row_y), scale=0.32, color=TEXT)
        peak_idx = int(np.nanargmax(np.abs(delta)))
        put_text(canvas, f"max |Δ| L{layers[peak_idx]} {delta[peak_idx]:+.2f}", (plot_x, y + h - 10), scale=0.32, color=MUTED)
    else:
        put_text(canvas, "no expert capture", (plot_x + 10, plot_y + 50), scale=0.42, color=MUTED)
    draw_frame(canvas, plot_x, plot_y, plot_w, plot_h)


def draw_denoise_delta_tile(
    canvas: np.ndarray,
    summary: dict[str, Any],
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    cap: float,
) -> None:
    put_text(canvas, "expert Δ over denoise", (x, y + 14), scale=0.38, color=MUTED)
    abs_mat = np.asarray(summary.get("abs", []), dtype=np.float32)
    layers = summary.get("layers", [])
    plot_x = x + 48
    plot_y = y + 26
    plot_w = w - 56
    plot_h = h - 54
    if abs_mat.size:
        delta_mat = abs_mat - abs_mat[:, [0]]
        heat = signed_color(delta_mat, plot_w, plot_h, cap)
        canvas[plot_y : plot_y + plot_h, plot_x : plot_x + plot_w] = heat
        for idx, layer in enumerate(layers):
            if int(layer) % 3 == 0:
                row_y = plot_y + int((idx + 0.58) / len(layers) * plot_h)
                put_text(canvas, f"L{layer}", (x, row_y), scale=0.32, color=TEXT)
        steps = abs_mat.shape[1]
        for frac, label in [(0.0, "0"), (0.5, str(max(0, steps // 2))), (0.98, str(max(0, steps - 1)))]:
            px = plot_x + int(frac * plot_w)
            cv2.line(canvas, (px, plot_y), (px, plot_y + plot_h), SUBGRID, 1)
            put_text(canvas, label, (px - 6, y + h - 10), scale=0.32, color=MUTED)
    else:
        put_text(canvas, "no denoise matrix", (plot_x + 10, plot_y + 50), scale=0.42, color=MUTED)
    draw_frame(canvas, plot_x, plot_y, plot_w, plot_h)


def draw_attribution_notes(
    canvas: np.ndarray,
    summary: dict[str, Any],
    actions: np.ndarray | None,
    x: int,
    y: int,
    w: int,
    *,
    n_action_steps: int,
) -> None:
    put_text(canvas, "alignment notes", (x, y + 14), scale=0.38, color=MUTED)
    window = action_window_for_chunk(actions, n_action_steps)
    lines: list[str] = []
    if window.size:
        means = np.nanmean(window, axis=0)
        order = np.argsort(-np.abs(means))[:2]
        lines.append("actions:")
        lines.extend(f"{ACTION_DIM_LABELS[idx]} {means[idx]:+.3f}" for idx in order)
    delta = np.asarray(summary.get("delta", []), dtype=np.float32)
    layers = summary.get("layers", [])
    if delta.size:
        order = np.argsort(-np.abs(delta))[:2]
        lines.append("layers:")
        lines.extend(f"L{layers[idx]} Δ{delta[idx]:+.2f}" for idx in order)
    if not lines:
        lines.append("no capture")
    text_y = y + 40
    for line in lines[:6]:
        put_text(canvas, fit_text(line, w - 4, scale=0.34), (x, text_y), scale=0.34, color=TEXT)
        text_y += 20


def make_action_layer_correlation(
    capture: dict[str, Any],
    chunks: list[int],
    out_dir: Path,
    *,
    task_id: int,
    episode: int,
    n_action_steps: int,
) -> dict[str, str]:
    action_names, actions = action_feature_series(capture, chunks, n_action_steps)
    layers, layer_delta = family_layer_matrix(capture, "expert", chunks, expert_metric="delta")
    if actions.size == 0 or layer_delta.size == 0 or len(chunks) < 3:
        return {}

    corr = np.full((len(action_names), len(layers)), np.nan, dtype=np.float32)
    for row in range(len(action_names)):
        x_values = actions[row]
        for col in range(len(layers)):
            y_values = layer_delta[col]
            valid = np.isfinite(x_values) & np.isfinite(y_values)
            if int(valid.sum()) < 3:
                continue
            x_centered = x_values[valid] - float(np.mean(x_values[valid]))
            y_centered = y_values[valid] - float(np.mean(y_values[valid]))
            denom = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
            if denom > 1e-8:
                corr[row, col] = float(np.dot(x_centered, y_centered) / denom)

    width = 1320
    height = 600
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = BG
    put_text(canvas, f"Action-to-Expert-Layer Alignment | task_id={task_id} episode={episode}", (18, 34), scale=0.72, color=TEXT, thickness=2)
    put_text(canvas, "Pearson r across policy chunks: action feature vs expert final-initial activation delta. Correlation, not causation.", (18, 62), scale=0.44, color=MUTED)
    plot = (150, 96, 920, 390)
    x, y, w, h = plot
    heat = signed_color(corr, w, h, 1.0)
    canvas[y : y + h, x : x + w] = heat
    draw_frame(canvas, x, y, w, h)
    for row, name in enumerate(action_names):
        row_y = y + int((row + 0.58) / len(action_names) * h)
        put_text(canvas, name, (22, row_y), scale=0.38, color=TEXT)
    for col, layer in enumerate(layers):
        if col % 2 == 0 or len(layers) <= 12:
            col_x = x + int((col + 0.42) / len(layers) * w)
            put_text(canvas, f"L{layer}", (col_x - 12, y + h + 24), scale=0.36, color=MUTED)
    put_text(canvas, "blue inverse", (x, y + h + 58), scale=0.4, color=ACCENT)
    put_text(canvas, "red same direction", (x + 160, y + h + 58), scale=0.4, color=RED)

    flat_order = np.argsort(-np.abs(np.nan_to_num(corr, nan=0.0)), axis=None)
    notes_x = 1100
    put_text(canvas, "strongest pairs", (notes_x, 116), scale=0.46, color=YELLOW, thickness=2)
    written = 0
    for flat_idx in flat_order:
        row, col = np.unravel_index(int(flat_idx), corr.shape)
        value = corr[row, col]
        if not np.isfinite(value) or abs(float(value)) < 0.05:
            continue
        put_text(canvas, f"{action_names[row]} -> L{layers[col]} r={value:+.2f}", (notes_x, 150 + written * 28), scale=0.38, color=TEXT)
        written += 1
        if written >= 10:
            break

    path = out_dir / f"task_{task_id}_episode_{episode}_action_layer_correlation.png"
    cv2.imwrite(str(path), canvas)
    return {"action_layer_correlation": str(path)}


def make_family_heatmaps(
    capture: dict[str, Any],
    chunks: list[int],
    out_dir: Path,
    *,
    task_id: int,
    episode: int,
) -> dict[str, str]:
    panels: list[np.ndarray] = []
    written: dict[str, str] = {}
    for family in ["vision", "prefix", "expert"]:
        layers, mat = family_layer_matrix(capture, family, chunks)
        if mat.size == 0:
            continue
        cap = safe_percentile(np.log1p(np.nan_to_num(mat, nan=0.0)), 98, 1.0)
        x_labels = [(0.0, "chunk 0"), (0.5, str(chunks[len(chunks) // 2])), (0.98, str(chunks[-1]))] if chunks else []
        y_labels = [((idx + 0.5) / max(1, len(layers)), str(layer)) for idx, layer in enumerate(layers) if layer % 3 == 0]
        panels.append(
            draw_heatmap_panel(
                f"{family} activation magnitude: layer x chunk",
                mat,
                width=980,
                height=320,
                cap=cap,
                log_scale=True,
                x_labels=x_labels,
                y_labels=y_labels,
            )
        )

    layers, delta = family_layer_matrix(capture, "expert", chunks, expert_metric="delta")
    if delta.size:
        cap = safe_percentile(np.abs(np.nan_to_num(delta, nan=0.0)), 98, 1.0)
        x_labels = [(0.0, "chunk 0"), (0.5, str(chunks[len(chunks) // 2])), (0.98, str(chunks[-1]))] if chunks else []
        y_labels = [((idx + 0.5) / max(1, len(layers)), str(layer)) for idx, layer in enumerate(layers) if layer % 3 == 0]
        panels.append(
            draw_heatmap_panel(
                "expert activation delta: final denoise - initial denoise",
                delta,
                width=980,
                height=320,
                cap=cap,
                signed=True,
                x_labels=x_labels,
                y_labels=y_labels,
            )
        )

    if panels:
        gap = 18
        canvas = np.zeros((sum(p.shape[0] for p in panels) + gap * (len(panels) - 1), 980, 3), dtype=np.uint8)
        canvas[:] = PANEL_BG
        y = 0
        for panel in panels:
            canvas[y : y + panel.shape[0], : panel.shape[1]] = panel
            y += panel.shape[0] + gap
        path = out_dir / f"task_{task_id}_episode_{episode}_activation_family_heatmaps.png"
        cv2.imwrite(str(path), canvas)
        written["family_heatmaps"] = str(path)
    return written


def draw_line_chart(
    title: str,
    series: dict[str, np.ndarray],
    chunks: list[int],
    *,
    width: int = 900,
    height: int = 320,
) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = BG
    put_text(img, title, (18, 34), scale=0.66, color=TEXT, thickness=2)
    plot = (64, 62, width - 98, height - 112)
    x, y, w, h = plot
    draw_frame(img, x, y, w, h)
    colors = {"initial": MUTED, "mean": ACCENT, "final": GREEN, "delta": YELLOW}

    all_values = np.concatenate([np.asarray(v, dtype=np.float32).flatten() for v in series.values() if len(v)])
    all_values = all_values[np.isfinite(all_values)]
    if all_values.size == 0:
        put_text(img, "no data", (x + 18, y + 44), color=MUTED)
        return img
    ymin = float(np.min(all_values))
    ymax = float(np.max(all_values))
    if abs(ymax - ymin) < 1e-6:
        ymax = ymin + 1.0
    pad = 0.08 * (ymax - ymin)
    ymin -= pad
    ymax += pad
    put_text(img, f"{ymax:.2f}", (14, y + 5), scale=0.36, color=MUTED)
    put_text(img, f"{ymin:.2f}", (14, y + h), scale=0.36, color=MUTED)
    if chunks:
        put_text(img, f"chunk {chunks[0]}", (x, y + h + 22), scale=0.38, color=MUTED)
        put_text(img, f"chunk {chunks[-1]}", (x + w - 72, y + h + 22), scale=0.38, color=MUTED)

    for name, values in series.items():
        arr = np.asarray(values, dtype=np.float32)
        points = []
        for idx, value in enumerate(arr):
            if not np.isfinite(value):
                continue
            px = x + int(idx / max(1, len(arr) - 1) * w)
            py = y + h - int((float(value) - ymin) / (ymax - ymin) * h)
            points.append((px, py))
        for left, right in zip(points, points[1:], strict=False):
            cv2.line(img, left, right, colors.get(name, TEXT), 2)
        legend_x = x + 12 + list(series).index(name) * 140
        cv2.line(img, (legend_x, height - 30), (legend_x + 28, height - 30), colors.get(name, TEXT), 3)
        put_text(img, name, (legend_x + 36, height - 24), scale=0.42, color=TEXT)
    return img


def make_expert_layer_graphs(
    capture: dict[str, Any],
    chunks: list[int],
    out_dir: Path,
    *,
    task_id: int,
    episode: int,
) -> dict[str, Any]:
    layer_series: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    all_layers: set[int] = set()
    for chunk in chunks:
        summary = expert_summary_for_chunk(capture, chunk)
        layers = [int(layer) for layer in summary["layers"]]
        all_layers.update(layers)
        abs_mat = summary["abs"]
        if abs_mat.size == 0:
            for layer in sorted(all_layers):
                for key in ["initial", "mean", "final", "delta"]:
                    layer_series[layer][key].append(np.nan)
            continue
        for row, layer in enumerate(layers):
            initial = float(abs_mat[row, 0])
            final = float(abs_mat[row, -1])
            layer_series[layer]["initial"].append(initial)
            layer_series[layer]["mean"].append(float(np.nanmean(abs_mat[row])))
            layer_series[layer]["final"].append(final)
            layer_series[layer]["delta"].append(final - initial)

    layer_dir = out_dir / f"task_{task_id}_episode_{episode}_expert_layers"
    layer_dir.mkdir(parents=True, exist_ok=True)
    layer_paths: list[str] = []
    small_panels: list[np.ndarray] = []
    for layer in sorted(layer_series):
        series = {key: np.asarray(values, dtype=np.float32) for key, values in layer_series[layer].items()}
        chart = draw_line_chart(f"expert layer {layer}: activation over policy chunks", series, chunks)
        path = layer_dir / f"expert_layer_{layer:02d}_over_chunks.png"
        cv2.imwrite(str(path), chart)
        layer_paths.append(str(path))
        small = cv2.resize(chart, (450, 160), interpolation=cv2.INTER_AREA)
        small_panels.append(small)

    if small_panels:
        cols = 2
        rows = math.ceil(len(small_panels) / cols)
        grid = np.zeros((rows * 184 + 54, cols * 468, 3), dtype=np.uint8)
        grid[:] = PANEL_BG
        put_text(grid, "Expert Transformer Layers Over Time", (18, 36), scale=0.72, color=TEXT, thickness=2)
        for idx, panel in enumerate(small_panels):
            row = idx // cols
            col = idx % cols
            x = col * 468 + 9
            y = row * 184 + 54
            grid[y : y + panel.shape[0], x : x + panel.shape[1]] = panel
        grid_path = out_dir / f"task_{task_id}_episode_{episode}_expert_layers_grid.png"
        cv2.imwrite(str(grid_path), grid)
    else:
        grid_path = None
    return {"expert_layers_grid": str(grid_path) if grid_path else "", "expert_layer_graphs": layer_paths}


def make_chunk_matrix(
    *,
    run_dir: Path,
    video_path: Path,
    capture: dict[str, Any],
    chunks: list[int],
    out_dir: Path,
    task_id: int,
    episode: int,
    n_action_steps: int,
    max_rows: int,
) -> str:
    chunks = chunks[:max_rows]
    row_h = 270
    header_h = 124
    width = 2280
    height = header_h + row_h * max(1, len(chunks))
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = BG

    eval_info = read_json(run_dir / "eval_info.json")
    metrics = eval_info.get("overall") or parse_metrics_from_log(run_dir / "run.log")
    task = ""
    for chunk in chunks:
        task = capture["chunk_meta"].get(chunk, {}).get("task", "")
        if task:
            break
    put_text(canvas, f"Pi0.5 LIBERO Chunk Matrix | task_id={task_id} episode={episode}", (18, 32), scale=0.76, color=TEXT, thickness=2)
    put_text(
        canvas,
        f"success={metrics.get('pc_success', 'n/a')}% | rows={len(chunks)} | one row = one policy call, first {n_action_steps} actions executed before replanning",
        (18, 62),
        scale=0.48,
        color=MUTED,
    )
    put_text(
        canvas,
        "activation panels use signed deltas: red=increase, blue=decrease relative to first denoise pass",
        (18, 88),
        scale=0.42,
        color=MUTED,
    )

    columns = [
        (18, 140, "chunk/task"),
        (170, 190, "sim third-person"),
        (380, 190, "input image"),
        (590, 190, "input image2"),
        (815, 360, "executed 10-step action"),
        (1198, 230, "action summary"),
        (1460, 250, "expert layer delta"),
        (1740, 340, "expert denoise delta"),
        (2100, 160, "top action/layer links"),
    ]
    for x, _w, label in columns:
        put_text(canvas, label, (x, header_h - 16), scale=0.44, color=YELLOW)

    scales = build_scales(capture)
    expert_delta_cap = scales.get("expert_delta", 1.0)
    action_cap = scales.get("action", 1.0)
    frame_indices = [chunk * n_action_steps for chunk in chunks]
    rollout_frames = sample_video_frames(video_path, frame_indices, 190, 176)
    for row_idx, chunk in enumerate(chunks):
        y0 = header_h + row_idx * row_h
        color = ROW_BG if row_idx % 2 == 0 else ROW_BG_ALT
        cv2.rectangle(canvas, (0, y0), (width, y0 + row_h), color, -1)
        frame_idx = chunk * n_action_steps
        put_text(canvas, f"chunk {chunk}", (18, y0 + 32), scale=0.55, color=TEXT, thickness=2)
        put_text(canvas, f"frames {frame_idx}-{frame_idx + n_action_steps - 1}", (18, y0 + 58), scale=0.38, color=MUTED)
        task_text = trim_task(capture["chunk_meta"].get(chunk, {}).get("task", task), 95)
        put_text(canvas, fit_text(task_text, 145, scale=0.36), (18, y0 + 92), scale=0.36, color=MUTED)

        rollout = rollout_frames[frame_idx]
        canvas[y0 + 38 : y0 + 214, 170 : 360] = rollout
        draw_frame(canvas, 170, y0 + 38, 190, 176)

        images = capture["images"].get(chunk, {})
        image1 = images.get("observation.images.image") or images.get("image")
        image2 = images.get("observation.images.image2") or images.get("image2")
        for x, path in [(380, image1), (590, image2)]:
            tile = load_capture_image(path, 190, 176) if path else np.zeros((176, 190, 3), dtype=np.uint8)
            canvas[y0 + 38 : y0 + 214, x : x + 190] = tile
            draw_frame(canvas, x, y0 + 38, 190, 176)

        actions = capture["actions"].get(chunk)
        draw_action_tile(canvas, actions, 815, y0 + 36, 360, 178, n_action_steps=n_action_steps, cap=action_cap)
        draw_action_summary_tile(canvas, actions, 1198, y0 + 36, 230, 178, n_action_steps=n_action_steps, cap=action_cap)

        summary = expert_summary_for_chunk(capture, chunk)
        draw_layer_delta_tile(canvas, summary, 1460, y0 + 36, 250, 178, cap=expert_delta_cap)
        draw_denoise_delta_tile(canvas, summary, 1740, y0 + 36, 340, 178, cap=expert_delta_cap)
        draw_attribution_notes(canvas, summary, actions, 2100, y0 + 36, 160, n_action_steps=n_action_steps)

    path = out_dir / f"task_{task_id}_episode_{episode}_chunk_matrix.png"
    cv2.imwrite(str(path), canvas)
    return str(path)


def activation_series_data(capture: dict[str, Any], chunks: list[int], n_action_steps: int = 10) -> dict[str, Any]:
    data: dict[str, Any] = {"chunks": chunks, "families": {}, "actions": {}}
    action_names, action_values = action_feature_series(capture, chunks, n_action_steps)
    for row, name in enumerate(action_names):
        data["actions"][name] = [
            None if not np.isfinite(value) else round(float(value), 6)
            for value in action_values[row].tolist()
        ]
    for family in ["vision", "prefix"]:
        layers, mat = family_layer_matrix(capture, family, chunks)
        if mat.size:
            data["families"][family] = {
                "layers": layers,
                "metrics": {"abs_mean": matrix_to_json(mat)},
            }

    expert_metrics: dict[str, list[list[float | None]]] = {}
    expert_layers: list[int] = []
    for metric in ["initial", "mean", "final", "delta"]:
        layers, mat = family_layer_matrix(capture, "expert", chunks, expert_metric=metric)
        if mat.size:
            expert_layers = layers
            expert_metrics[metric] = matrix_to_json(mat)
    if expert_metrics:
        data["families"]["expert"] = {"layers": expert_layers, "metrics": expert_metrics}
    return data


def _img_data_uri(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def write_interactive_html(
    *,
    out_dir: Path,
    task_id: int,
    episode: int,
    run_dir: Path,
    artifacts: dict[str, Any],
    chunks: list[int],
    capture: dict[str, Any],
    n_action_steps: int,
) -> str:
    task = ""
    for chunk in chunks:
        task = capture["chunk_meta"].get(chunk, {}).get("task", "")
        if task:
            break

    data = activation_series_data(capture, chunks, n_action_steps)
    images = {
        "Chunk Matrix": _img_data_uri(artifacts.get("chunk_matrix", "")),
        "Action-Layer Correlation": _img_data_uri(artifacts.get("action_layer_correlation", "")),
        "Family Heatmaps": _img_data_uri(artifacts.get("family_heatmaps", "")),
        "Expert Layer Grid": _img_data_uri(artifacts.get("expert_layers_grid", "")),
    }
    layer_images = {
        Path(path).stem.replace("_", " "): _img_data_uri(path)
        for path in artifacts.get("expert_layer_graphs", [])
        if Path(path).exists()
    }
    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Pi0.5 LIBERO Interactive Report</title>
  <style>
    :root {{ color-scheme: dark; }}
    body {{ margin: 0; font-family: Arial, sans-serif; background: #080d1a; color: #e5e7eb; }}
    main {{ padding: 18px; max-width: 1500px; margin: 0 auto; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    h2 {{ margin-top: 26px; color: #f8fafc; }}
    .muted {{ color: #9ca3af; }}
    .controls {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin: 16px 0; }}
    select, button {{ background: #111827; color: #e5e7eb; border: 1px solid #374151; border-radius: 6px; padding: 8px 10px; }}
    .panel {{ background: #0f172a; border: 1px solid #263142; border-radius: 8px; padding: 14px; margin: 14px 0; }}
    svg {{ width: 100%; height: 360px; background: #0b1020; border: 1px solid #263142; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #263142; background: #111827; }}
    .tabs {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }}
    .tabs button.active {{ background: #0369a1; border-color: #38bdf8; }}
    .hidden {{ display: none; }}
    code {{ color: #bae6fd; }}
  </style>
</head>
<body>
<main>
  <h1>Pi0.5 LIBERO Interactive Report</h1>
  <div class="muted">task_id={task_id} episode={episode} | run={html.escape(str(run_dir))}</div>
  <p>{html.escape(trim_task(task, 260))}</p>
  <div class="panel">
    <h2>Activation Plot</h2>
    <div class="muted">Select a captured model family, layer, and metric. Expert layers are the action/diffusion path.</div>
    <div class="controls">
      <label>Family <select id="family"></select></label>
      <label>Metric <select id="metric"></select></label>
      <label>Layer <select id="layer"></select></label>
      <label>Action overlay <select id="action"></select></label>
    </div>
    <svg id="plot" viewBox="0 0 1000 360" role="img"></svg>
  </div>
  <div class="panel">
    <h2>Standalone Views</h2>
    <div class="tabs" id="tabs"></div>
    <div id="imagePanels"></div>
  </div>
</main>
<script>
const report = {json.dumps(data)};
const images = {json.dumps(images)};
const layerImages = {json.dumps(layer_images)};
const familySelect = document.getElementById('family');
const metricSelect = document.getElementById('metric');
const layerSelect = document.getElementById('layer');
const actionSelect = document.getElementById('action');
const plot = document.getElementById('plot');

function option(value, label) {{
  const opt = document.createElement('option');
  opt.value = value;
  opt.textContent = label ?? value;
  return opt;
}}

function initControls() {{
  Object.keys(report.families).forEach(name => familySelect.appendChild(option(name)));
  familySelect.value = report.families.expert ? 'expert' : Object.keys(report.families)[0];
  actionSelect.appendChild(option('', 'none'));
  Object.keys(report.actions || {{}}).forEach(name => actionSelect.appendChild(option(name)));
  updateMetricLayer();
  familySelect.addEventListener('change', updateMetricLayer);
  metricSelect.addEventListener('change', drawPlot);
  layerSelect.addEventListener('change', drawPlot);
  actionSelect.addEventListener('change', drawPlot);
}}

function updateMetricLayer() {{
  const family = report.families[familySelect.value];
  metricSelect.innerHTML = '';
  Object.keys(family.metrics).forEach(name => metricSelect.appendChild(option(name)));
  if (family.metrics.mean) metricSelect.value = 'mean';
  layerSelect.innerHTML = '';
  family.layers.forEach((layer, idx) => layerSelect.appendChild(option(String(idx), `layer ${{layer}}`)));
  drawPlot();
}}

function drawPlot() {{
  const family = report.families[familySelect.value];
  const metric = metricSelect.value;
  const layerIdx = Number(layerSelect.value || 0);
  const values = (family.metrics[metric] || [])[layerIdx] || [];
  const actionName = actionSelect.value;
  const actionValues = actionName ? (report.actions[actionName] || []) : [];
  const chunks = report.chunks;
  const finite = values.filter(v => Number.isFinite(v));
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  const yMin = Number.isFinite(min) ? min - Math.max(1e-6, (max - min) * 0.08) : 0;
  const yMax = Number.isFinite(max) && max !== yMin ? max + Math.max(1e-6, (max - min) * 0.08) : 1;
  const finiteAction = actionValues.filter(v => Number.isFinite(v));
  const aMinRaw = Math.min(...finiteAction);
  const aMaxRaw = Math.max(...finiteAction);
  const aPad = Number.isFinite(aMaxRaw - aMinRaw) ? Math.max(1e-6, Math.abs(aMaxRaw - aMinRaw) * 0.08) : 1;
  const aMin = Number.isFinite(aMinRaw) ? aMinRaw - aPad : 0;
  const aMax = Number.isFinite(aMaxRaw) && aMaxRaw !== aMinRaw ? aMaxRaw + aPad : 1;
  const W = 1000, H = 360, left = 72, right = 86, top = 34, bottom = 54;
  const pw = W - left - right, ph = H - top - bottom;
  const pts = values.map((v, i) => {{
    const x = left + (values.length <= 1 ? 0 : i / (values.length - 1)) * pw;
    const y = top + (1 - (v - yMin) / (yMax - yMin)) * ph;
    return [x, y, v, chunks[i]];
  }}).filter(p => Number.isFinite(p[2]));
  const actionPts = actionValues.map((v, i) => {{
    const x = left + (actionValues.length <= 1 ? 0 : i / (actionValues.length - 1)) * pw;
    const y = top + (1 - (v - aMin) / (aMax - aMin)) * ph;
    return [x, y, v, chunks[i]];
  }}).filter(p => Number.isFinite(p[2]));
  const poly = pts.map(p => `${{p[0].toFixed(1)}},${{p[1].toFixed(1)}}`).join(' ');
  const actionPoly = actionPts.map(p => `${{p[0].toFixed(1)}},${{p[1].toFixed(1)}}`).join(' ');
  const circles = pts.map(p => `<circle cx="${{p[0].toFixed(1)}}" cy="${{p[1].toFixed(1)}}" r="4"><title>chunk ${{p[3]}}: ${{p[2].toFixed(4)}}</title></circle>`).join('');
  const actionCircles = actionPts.map(p => `<circle cx="${{p[0].toFixed(1)}}" cy="${{p[1].toFixed(1)}}" r="3"><title>${{actionName}} chunk ${{p[3]}}: ${{p[2].toFixed(4)}}</title></circle>`).join('');
  plot.innerHTML = `
    <rect x="0" y="0" width="${{W}}" height="${{H}}" fill="#0b1020"/>
    <text x="18" y="24" fill="#e5e7eb" font-size="18">${{familySelect.value}} layer ${{family.layers[layerIdx]}} ${{metric}} over chunks</text>
    <line x1="${{left}}" y1="${{top}}" x2="${{left}}" y2="${{top+ph}}" stroke="#334155"/>
    <line x1="${{left}}" y1="${{top+ph}}" x2="${{left+pw}}" y2="${{top+ph}}" stroke="#334155"/>
    <line x1="${{left+pw}}" y1="${{top}}" x2="${{left+pw}}" y2="${{top+ph}}" stroke="#334155"/>
    <text x="8" y="${{top+8}}" fill="#9ca3af" font-size="12">${{yMax.toFixed(3)}}</text>
    <text x="8" y="${{top+ph}}" fill="#9ca3af" font-size="12">${{yMin.toFixed(3)}}</text>
    <text x="${{left+pw+8}}" y="${{top+8}}" fill="#facc15" font-size="12">${{actionName ? aMax.toFixed(3) : ''}}</text>
    <text x="${{left+pw+8}}" y="${{top+ph}}" fill="#facc15" font-size="12">${{actionName ? aMin.toFixed(3) : ''}}</text>
    <text x="${{left}}" y="${{H-18}}" fill="#9ca3af" font-size="12">chunk ${{chunks[0] ?? ''}}</text>
    <text x="${{left+pw-70}}" y="${{H-18}}" fill="#9ca3af" font-size="12">chunk ${{chunks[chunks.length-1] ?? ''}}</text>
    <polyline points="${{poly}}" fill="none" stroke="#38bdf8" stroke-width="3"/>
    <g fill="#38bdf8">${{circles}}</g>
    ${{actionName ? `<polyline points="${{actionPoly}}" fill="none" stroke="#facc15" stroke-width="2.5"/><g fill="#facc15">${{actionCircles}}</g><text x="${{left+pw-220}}" y="24" fill="#facc15" font-size="13">overlay: ${{actionName}}</text>` : ''}}`;
}}

function initImages() {{
  const tabs = document.getElementById('tabs');
  const panels = document.getElementById('imagePanels');
  const merged = {{...images, ...layerImages}};
  Object.entries(merged).forEach(([name, uri], idx) => {{
    if (!uri) return;
    const btn = document.createElement('button');
    btn.textContent = name;
    btn.onclick = () => showImage(name);
    tabs.appendChild(btn);
    const div = document.createElement('div');
    div.id = 'panel-' + name.replaceAll(' ', '-');
    div.className = 'hidden';
    div.innerHTML = `<h3>${{name}}</h3><img src="${{uri}}" />`;
    panels.appendChild(div);
    if (idx === 0) showImage(name);
  }});
}}

function showImage(name) {{
  [...document.querySelectorAll('#tabs button')].forEach(btn => btn.classList.toggle('active', btn.textContent === name));
  [...document.querySelectorAll('#imagePanels > div')].forEach(div => div.classList.add('hidden'));
  const panel = document.getElementById('panel-' + name.replaceAll(' ', '-'));
  if (panel) panel.classList.remove('hidden');
}}

initControls();
initImages();
</script>
</body>
</html>"""
    path = out_dir / f"task_{task_id}_episode_{episode}_interactive.html"
    path.write_text(html_text)
    return str(path)


def write_html(
    *,
    out_dir: Path,
    task_id: int,
    episode: int,
    run_dir: Path,
    artifacts: dict[str, Any],
    chunks: list[int],
    capture: dict[str, Any],
) -> str:
    task = ""
    for chunk in chunks:
        task = capture["chunk_meta"].get(chunk, {}).get("task", "")
        if task:
            break
    rows = [
        "<html><body style='font-family:Arial,sans-serif;background:#0b1020;color:#e5e7eb;'>",
        f"<h1>Pi0.5 LIBERO Report: task {task_id}, episode {episode}</h1>",
        f"<p><b>Run:</b> {html.escape(str(run_dir))}</p>",
        f"<p><b>Task:</b> {html.escape(trim_task(task, 220))}</p>",
        "<p>Each chunk row is one policy call. LeRobot executes the first 10 predicted actions, then replans from fresh simulator observations.</p>",
        "<ul>",
    ]
    for key, value in artifacts.items():
        if isinstance(value, list):
            rows.append(f"<li>{html.escape(key)}: {len(value)} files</li>")
        elif value:
            rows.append(f"<li>{html.escape(key)}: {html.escape(str(value))}</li>")
    rows.extend(["</ul>", "</body></html>"])
    path = out_dir / f"task_{task_id}_episode_{episode}_report.html"
    path.write_text("\n".join(rows))
    return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create Colab-visible granular reports for Pi0.5 LIBERO activation captures.")
    parser.add_argument("--run", default="latest", help="Run id, run directory, or 'latest'.")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--n-action-steps", type=int, default=10)
    parser.add_argument("--max-rows", type=int, default=80)
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    run_dir = resolve_run(args.run)
    video_path = find_task_video(run_dir, args.task_id, args.episode)
    capture = read_capture(run_dir)
    chunks = chunks_with_capture(capture)
    if not chunks:
        raise SystemExit("No activation capture chunks found. Rerun with CAPTURE_ACTIVATIONS=1.")

    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "analysis" / f"task_{args.task_id}_episode_{args.episode}_colab_report"
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, Any] = {}
    artifacts["chunk_matrix"] = make_chunk_matrix(
        run_dir=run_dir,
        video_path=video_path,
        capture=capture,
        chunks=chunks,
        out_dir=out_dir,
        task_id=args.task_id,
        episode=args.episode,
        n_action_steps=args.n_action_steps,
        max_rows=args.max_rows,
    )
    artifacts.update(
        make_action_layer_correlation(
            capture,
            chunks,
            out_dir,
            task_id=args.task_id,
            episode=args.episode,
            n_action_steps=args.n_action_steps,
        )
    )
    artifacts.update(make_family_heatmaps(capture, chunks, out_dir, task_id=args.task_id, episode=args.episode))
    artifacts.update(make_expert_layer_graphs(capture, chunks, out_dir, task_id=args.task_id, episode=args.episode))
    artifacts["interactive_html"] = write_interactive_html(
        out_dir=out_dir,
        task_id=args.task_id,
        episode=args.episode,
        run_dir=run_dir,
        artifacts=artifacts,
        chunks=chunks,
        capture=capture,
        n_action_steps=args.n_action_steps,
    )
    artifacts["html"] = write_html(
        out_dir=out_dir,
        task_id=args.task_id,
        episode=args.episode,
        run_dir=run_dir,
        artifacts=artifacts,
        chunks=chunks,
        capture=capture,
    )

    manifest_path = out_dir / f"task_{args.task_id}_episode_{args.episode}_manifest.json"
    manifest_path.write_text(json.dumps(artifacts, indent=2))
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

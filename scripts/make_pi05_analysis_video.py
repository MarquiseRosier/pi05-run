#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import cv2
    import numpy as np
except ModuleNotFoundError:
    if os.environ.get("PI05_ANALYSIS_NO_UV_REEXEC") != "1" and shutil.which("uv"):
        env = os.environ.copy()
        env["PI05_ANALYSIS_NO_UV_REEXEC"] = "1"
        os.execvpe("uv", ["uv", "run", "python", __file__, *sys.argv[1:]], env)
    raise


ROOT = Path("outputs/eval/pi05_libero")
CANVAS_W = 1920
CANVAS_H = 1080
PANEL_W = CANVAS_W // 2
PANEL_H = CANVAS_H // 2
HEADER_H = 52
FONT = cv2.FONT_HERSHEY_SIMPLEX

BG = (9, 13, 23)
PANEL_BG = (15, 23, 42)
HEADER_BG = (42, 23, 13)
TEXT = (241, 245, 249)
MUTED = (185, 197, 214)
GRID = (70, 82, 102)
ACCENT = (56, 189, 248)


def resolve_run(run: str) -> Path:
    path = Path(run)
    if path.is_dir():
        return path
    if run == "latest":
        runs = sorted(p for p in ROOT.glob("*") if p.is_dir())
        if not runs:
            raise SystemExit(f"No local runs found under {ROOT}")
        return runs[-1]
    path = ROOT / run
    if not path.is_dir():
        raise SystemExit(f"Run not found: {path}")
    return path


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def strip_ansi(value: str) -> str:
    return re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", value)


def parse_metrics_from_log(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    lines = strip_ansi(path.read_text(errors="replace")).splitlines()
    for idx, line in enumerate(lines):
        if "Aggregated Metrics for overall:" not in line:
            continue
        for candidate in lines[idx + 1 : idx + 6]:
            start = candidate.find("{")
            if start == -1:
                continue
            try:
                value = ast.literal_eval(candidate[start:])
                if isinstance(value, dict):
                    return value
            except (SyntaxError, ValueError):
                continue
    return {}


def find_task_video(run_dir: Path, task_id: int, episode: int) -> Path:
    eval_info = read_json(run_dir / "eval_info.json")
    for item in eval_info.get("per_task", []):
        if int(item.get("task_id", -1)) != task_id:
            continue
        paths = item.get("metrics", {}).get("video_paths", [])
        if episode >= len(paths):
            continue
        original = Path(paths[episode])
        candidates = [
            original,
            run_dir / original.name,
            run_dir / "videos" / f"{item.get('task_group')}_{task_id}" / original.name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
    matches = sorted((run_dir / "videos").glob(f"*_{task_id}/eval_episode_{episode}.mp4"))
    if matches:
        return matches[0]
    raise SystemExit(f"No video found for task_id={task_id}, episode={episode} in {run_dir}")


def read_capture(run_dir: Path) -> dict[str, Any]:
    events_path = run_dir / "activation_capture" / "events.jsonl"
    capture_root = events_path.parent
    result: dict[str, Any] = {
        "events_path": events_path,
        "activations": defaultdict(lambda: defaultdict(list)),
        "actions": {},
        "images": defaultdict(dict),
        "chunk_meta": {},
        "hooks": [],
        "counts": Counter(),
    }
    if not events_path.exists():
        return result

    for raw in events_path.read_text(errors="replace").splitlines():
        if not raw.strip():
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        event_type = event.get("type")
        result["counts"][event_type] += 1
        if event_type == "hooks_installed":
            result["hooks"] = event.get("modules", [])
            continue
        chunk = event.get("chunk")
        if chunk is None:
            continue
        chunk = int(chunk)
        if event_type == "chunk_start":
            result["chunk_meta"][chunk] = event
        elif event_type == "activation":
            result["activations"][chunk][event.get("family", "unknown")].append(event)
        elif event_type == "action_chunk":
            result["actions"][chunk] = np.asarray(event.get("values", []), dtype=np.float32)
        elif event_type == "image":
            rel = Path(event.get("path", ""))
            result["images"][chunk][event.get("key", "image")] = capture_root / rel
    return result


def put_text(
    panel: np.ndarray,
    value: str,
    xy: tuple[int, int],
    scale: float = 0.62,
    color: tuple[int, int, int] = TEXT,
    thickness: int = 1,
) -> None:
    x, y = xy
    cv2.putText(panel, value, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def text_lines(
    panel: np.ndarray,
    lines: list[str],
    origin: tuple[int, int],
    scale: float = 0.58,
    color: tuple[int, int, int] = TEXT,
    line_height: int | None = None,
) -> None:
    x, y = origin
    step = line_height or max(20, int(32 * scale / 0.58))
    for line in lines:
        put_text(panel, line, (x, y), scale=scale, color=color)
        y += step


def draw_header(panel: np.ndarray, title: str, subtitle: str = "") -> None:
    cv2.rectangle(panel, (0, 0), (panel.shape[1], HEADER_H), HEADER_BG, -1)
    put_text(panel, title, (18, 34), scale=0.82, color=TEXT, thickness=2)
    if subtitle:
        subtitle = fit_text(subtitle, 390, scale=0.5)
        text_w = cv2.getTextSize(subtitle, FONT, 0.5, 1)[0][0]
        put_text(panel, subtitle, (panel.shape[1] - text_w - 20, 34), scale=0.5, color=MUTED)


def fit_text(value: str, max_width: int, scale: float = 0.58, thickness: int = 1) -> str:
    if cv2.getTextSize(value, FONT, scale, thickness)[0][0] <= max_width:
        return value
    suffix = "..."
    while value and cv2.getTextSize(value + suffix, FONT, scale, thickness)[0][0] > max_width:
        value = value[:-1]
    return value.rstrip() + suffix


def trim_task(value: str, limit: int = 110) -> str:
    value = value.replace("\n", " ")
    value = re.sub(r"\s+", " ", value)
    value = value.removeprefix("Task: ").split(", State:")[0].strip()
    if len(value) > limit:
        value = value[: limit - 1].rstrip() + "..."
    return value


def letterbox(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    out = np.zeros((height, width, 3), dtype=np.uint8)
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x0 = (width - new_w) // 2
    y0 = (height - new_h) // 2
    out[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return out


def safe_percentile(values: list[float] | np.ndarray, pct: float, default: float = 1.0) -> float:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return default
    value = float(np.percentile(arr, pct))
    if not math.isfinite(value) or value <= 0:
        return default
    return value


def sequential_color(
    matrix: np.ndarray,
    width: int,
    height: int,
    cap: float,
    *,
    log_scale: bool = False,
) -> np.ndarray:
    data = matrix.astype(np.float32)
    valid = np.isfinite(data)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data = np.clip(data, 0.0, None)
    if log_scale:
        data = np.log1p(data)
    norm = np.clip(data / max(cap, 1e-6), 0.0, 1.0)
    img = cv2.resize((norm * 255).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    colored = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)
    if not np.all(valid):
        mask = cv2.resize(valid.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) == 0
        colored[mask] = PANEL_BG
    return colored


def signed_color(matrix: np.ndarray, width: int, height: int, cap: float) -> np.ndarray:
    data = np.nan_to_num(matrix.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.clip(data / max(cap, 1e-6), -1.0, 1.0)
    pos = np.clip(norm, 0.0, 1.0)
    neg = np.clip(-norm, 0.0, 1.0)
    base = np.full((*norm.shape, 3), 34, dtype=np.float32)
    base[..., 0] += 210 * neg
    base[..., 1] += 90 * np.maximum(pos, neg)
    base[..., 2] += 220 * pos
    img = np.clip(base, 0, 255).astype(np.uint8)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)


def draw_frame(panel: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    cv2.rectangle(panel, (x, y), (x + w, y + h), GRID, 1)


def draw_heatmap(
    panel: np.ndarray,
    matrix: np.ndarray,
    rect: tuple[int, int, int, int],
    title: str,
    *,
    mode: str,
    cap: float,
    log_scale: bool = False,
    x_label: str = "",
    y_label: str = "",
    x_ticks: list[tuple[float, str]] | None = None,
    y_ticks: list[tuple[float, str]] | None = None,
) -> tuple[int, int, int, int]:
    x, y, w, h = rect
    put_text(panel, fit_text(title, w - 8, scale=0.54), (x, y), scale=0.54, color=TEXT)
    plot_x = x + 56
    plot_y = y + 28
    plot_w = w - 66
    plot_h = h - 58
    if matrix.size == 0:
        cv2.rectangle(panel, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), PANEL_BG, -1)
        draw_frame(panel, plot_x, plot_y, plot_w, plot_h)
        put_text(panel, "no data", (plot_x + 18, plot_y + 36), scale=0.54, color=MUTED)
        return plot_x, plot_y, plot_w, plot_h
    if mode == "signed":
        heat = signed_color(matrix, plot_w, plot_h, cap)
    else:
        heat = sequential_color(matrix, plot_w, plot_h, cap, log_scale=log_scale)
    panel[plot_y : plot_y + plot_h, plot_x : plot_x + plot_w] = heat
    draw_frame(panel, plot_x, plot_y, plot_w, plot_h)
    rows, cols = matrix.shape[:2]
    for frac, label in x_ticks or []:
        px = int(plot_x + frac * plot_w)
        cv2.line(panel, (px, plot_y), (px, plot_y + plot_h), (20, 25, 35), 1)
        put_text(panel, label, (px - 8, plot_y + plot_h + 20), scale=0.42, color=MUTED)
    for frac, label in y_ticks or []:
        py = int(plot_y + frac * plot_h)
        cv2.line(panel, (plot_x, py), (plot_x + plot_w, py), (20, 25, 35), 1)
        put_text(panel, label, (x + 6, py + 4), scale=0.42, color=MUTED)
    if x_label:
        put_text(panel, x_label, (plot_x + plot_w - 105, plot_y + plot_h + 20), scale=0.42, color=MUTED)
    if y_label:
        put_text(panel, y_label, (x + 6, plot_y - 4), scale=0.42, color=MUTED)
    if rows > 0 and cols > 0:
        return plot_x, plot_y, plot_w, plot_h
    return plot_x, plot_y, plot_w, plot_h


def placeholder(title: str, lines: list[str]) -> np.ndarray:
    panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    panel[:] = PANEL_BG
    draw_header(panel, title)
    text_lines(panel, lines, origin=(28, 92), scale=0.62, color=MUTED)
    return panel


def chunk_for_frame(frame_idx: int, n_action_steps: int) -> int:
    return max(0, frame_idx // max(1, n_action_steps))


def phase_for_frame(frame_idx: int, n_action_steps: int) -> int:
    return frame_idx % max(1, n_action_steps)


def group_expert_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    expert_events = [event for event in events if event.get("layer") is not None]
    expert_events.sort(key=lambda event: int(event.get("call", 0)))
    if not expert_events:
        return {"layers": [], "abs": np.empty((0, 0)), "tokens": np.empty((0, 0, 0))}

    layers = sorted({int(event["layer"]) for event in expert_events})
    layer_to_row = {layer: idx for idx, layer in enumerate(layers)}
    groups: list[dict[int, dict[str, Any]]] = []
    current: dict[int, dict[str, Any]] = {}
    for event in expert_events:
        layer = int(event["layer"])
        if layer in current:
            groups.append(current)
            current = {}
        current[layer] = event
    if current:
        groups.append(current)

    max_tokens = 0
    for group in groups:
        for event in group.values():
            max_tokens = max(max_tokens, len(event.get("token_abs") or []))
    abs_mat = np.full((len(layers), len(groups)), np.nan, dtype=np.float32)
    max_mat = np.full((len(layers), len(groups)), np.nan, dtype=np.float32)
    token_cube = np.full((len(groups), len(layers), max_tokens), np.nan, dtype=np.float32)

    for step, group in enumerate(groups):
        for layer, event in group.items():
            row = layer_to_row[layer]
            abs_mat[row, step] = float(event.get("abs_mean", np.nan))
            max_mat[row, step] = float(event.get("max_abs", np.nan))
            token_abs = np.asarray(event.get("token_abs") or [], dtype=np.float32)
            if token_abs.size:
                token_cube[step, row, : token_abs.size] = token_abs

    return {"layers": layers, "abs": abs_mat, "max": max_mat, "tokens": token_cube}


def build_scales(capture: dict[str, Any]) -> dict[str, float]:
    action_values: list[float] = []
    expert_abs: list[float] = []
    expert_delta: list[float] = []
    token_values: list[float] = []
    for values in capture["actions"].values():
        if values.size:
            action_values.extend(np.abs(values).flatten().tolist())
    for chunk_events in capture["activations"].values():
        grouped = group_expert_events(chunk_events.get("expert", []))
        abs_mat = grouped["abs"]
        if abs_mat.size:
            expert_abs.extend(np.nan_to_num(abs_mat, nan=0.0).flatten().tolist())
            baseline = abs_mat[:, [0]]
            expert_delta.extend(np.nan_to_num(abs_mat - baseline, nan=0.0).flatten().tolist())
        tokens = grouped["tokens"]
        if tokens.size:
            token_values.extend(np.nan_to_num(tokens, nan=0.0).flatten().tolist())
    return {
        "action": safe_percentile(action_values, 98, 1.0),
        "expert_log": safe_percentile(np.log1p(np.asarray(expert_abs, dtype=np.float32)), 98, 1.0),
        "expert_delta": safe_percentile(np.abs(expert_delta), 98, 1.0),
        "token_log": safe_percentile(np.log1p(np.asarray(token_values, dtype=np.float32)), 98, 1.0),
    }


def render_rollout(
    frame: np.ndarray,
    label: str,
    frame_idx: int,
    chunk: int,
    phase: int,
    metrics: dict[str, Any],
    task: str,
) -> np.ndarray:
    panel = letterbox(frame, PANEL_W, PANEL_H)
    cv2.rectangle(panel, (0, 0), (PANEL_W, 118), (0, 0, 0), -1)
    lines = [
        f"{label}",
        f"frame {frame_idx} | chunk {chunk} | executed action {phase}/9 | success {metrics.get('pc_success', 'n/a')}%",
        fit_text(trim_task(task, 180), PANEL_W - 48, scale=0.62),
    ]
    text_lines(panel, lines, origin=(24, 34), scale=0.62, color=TEXT, line_height=34)
    return panel


def render_inputs(capture: dict[str, Any], chunk: int) -> np.ndarray:
    images = capture["images"].get(chunk, {})
    task = capture["chunk_meta"].get(chunk, {}).get("task", "")
    if not images:
        return placeholder(
            "Input Cameras",
            ["No captured camera tensors for this chunk.", "Rerun with CAPTURE_ACTIVATIONS=1."],
        )
    panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    panel[:] = BG
    draw_header(panel, f"Input Cameras - chunk {chunk}", "policy tensors before forward pass")
    slots = [
        (20, 72, (PANEL_W - 52) // 2, 344),
        (PANEL_W // 2 + 6, 72, (PANEL_W - 52) // 2, 344),
    ]
    for (key, path), (x, y, w, h) in zip(sorted(images.items()), slots, strict=False):
        if path.suffix == ".npy":
            img = np.load(path)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img = cv2.imread(str(path))
        if img is None:
            continue
        tile = letterbox(img, w, h)
        panel[y : y + h, x : x + w] = tile
        cv2.rectangle(panel, (x, y), (x + w, y + 32), (0, 0, 0), -1)
        put_text(panel, key.replace("observation.images.", ""), (x + 12, y + 23), scale=0.54, color=TEXT)
    cv2.rectangle(panel, (0, 432), (PANEL_W, PANEL_H), (16, 24, 39), -1)
    text_lines(
        panel,
        ["Task", fit_text(trim_task(task, 180), PANEL_W - 48, scale=0.58)],
        origin=(24, 468),
        scale=0.58,
        color=MUTED,
        line_height=34,
    )
    return panel


def render_action(capture: dict[str, Any], chunk: int, phase: int, n_action_steps: int, scales: dict[str, float]) -> np.ndarray:
    values = capture["actions"].get(chunk)
    if values is None or values.size == 0:
        return placeholder(
            "Signed Action Chunk",
            ["No captured action chunk for this frame.", "The default eval video only has behavior."],
        )
    panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    panel[:] = PANEL_BG
    draw_header(panel, "Signed Policy Action Chunk", f"chunk {chunk} | model predicts 50; first {n_action_steps} are postprocessed and executed")
    dims = ["dx", "dy", "dz", "dRx", "dRy", "dRz", "grip"]
    matrix = values.T
    plot = draw_heatmap(
        panel,
        matrix,
        (22, 82, 900, 318),
        "blue = negative, red = positive",
        mode="signed",
        cap=scales["action"],
        x_ticks=[(0.0, "0"), (0.2, "10"), (0.5, "25"), (0.98, "49")],
        y_ticks=[((idx + 0.5) / len(dims), name) for idx, name in enumerate(dims)],
        x_label="future step",
        y_label="action dim",
    )
    px, py, pw, ph = plot
    current_frac = min(max((phase + 0.5) / values.shape[0], 0), 1)
    executed_frac = min(max(n_action_steps / values.shape[0], 0), 1)
    cv2.line(panel, (int(px + current_frac * pw), py), (int(px + current_frac * pw), py + ph), (255, 255, 255), 2)
    cv2.line(panel, (int(px + executed_frac * pw), py), (int(px + executed_frac * pw), py + ph), ACCENT, 2)
    put_text(panel, "white = current env step", (px, py + ph + 34), scale=0.44, color=TEXT)
    put_text(panel, "cyan = end of queue window", (px + 260, py + ph + 34), scale=0.44, color=ACCENT)

    norms = np.linalg.norm(values, axis=1)
    chart_x, chart_y, chart_w, chart_h = 80, 434, 805, 72
    draw_frame(panel, chart_x, chart_y, chart_w, chart_h)
    max_norm = max(float(norms.max()), 1e-6)
    points = []
    for idx, value in enumerate(norms):
        x = chart_x + int(idx / max(1, len(norms) - 1) * chart_w)
        y = chart_y + chart_h - int(float(value) / max_norm * (chart_h - 8)) - 4
        points.append((x, y))
    for left, right in zip(points, points[1:], strict=False):
        cv2.line(panel, left, right, ACCENT, 2)
    marker_x = chart_x + int(phase / max(1, len(norms) - 1) * chart_w)
    cv2.line(panel, (marker_x, chart_y), (marker_x, chart_y + chart_h), (255, 255, 255), 1)
    put_text(panel, "action L2 norm across predicted horizon", (chart_x, chart_y - 10), scale=0.48, color=MUTED)
    put_text(panel, f"scale +/-{scales['action']:.2f}", (chart_x + chart_w - 140, chart_y - 10), scale=0.44, color=MUTED)
    return panel


def render_activation(capture: dict[str, Any], chunk: int, scales: dict[str, float]) -> np.ndarray:
    events = capture["activations"].get(chunk, {}).get("expert", [])
    grouped = group_expert_events(events)
    abs_mat = grouped["abs"]
    if abs_mat.size == 0:
        return placeholder(
            "Expert Activation Diagnostics",
            ["No expert-layer hook trace for this chunk.", "Use the capture runner for forward-pass heatmaps."],
        )
    panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    panel[:] = PANEL_BG
    steps = abs_mat.shape[1]
    layers = grouped["layers"]
    counts = capture.get("counts", {})
    draw_header(
        panel,
        "Expert Activation Diagnostics",
        f"chunk {chunk} | {len(layers)} layers x {steps} denoise | {counts.get('activation', 0)} events",
    )

    x_ticks = [(0.0, "0"), (0.5, str(max(0, steps // 2))), (0.98, str(max(0, steps - 1)))]
    y_ticks = [((layer + 0.5) / max(1, len(layers)), str(layer)) for layer in layers if layer % 3 == 0]
    draw_heatmap(
        panel,
        abs_mat,
        (22, 80, 430, 202),
        "activation magnitude: layer x denoise",
        mode="seq",
        cap=scales["expert_log"],
        log_scale=True,
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        x_label="denoise",
        y_label="layer",
    )
    delta = abs_mat - abs_mat[:, [0]]
    draw_heatmap(
        panel,
        delta,
        (490, 80, 430, 202),
        "change from denoise 0: layer x denoise",
        mode="signed",
        cap=scales["expert_delta"],
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        x_label="denoise",
        y_label="layer",
    )

    tokens = grouped["tokens"]
    final_tokens = tokens[-1] if tokens.size else np.empty((0, 0), dtype=np.float32)
    token_ticks = [(0.0, "0"), (0.2, "10"), (0.5, "25"), (0.98, "49")]
    draw_heatmap(
        panel,
        final_tokens,
        (22, 320, 898, 156),
        "final denoise activation by layer x action-token",
        mode="seq",
        cap=scales["token_log"],
        log_scale=True,
        x_ticks=token_ticks,
        y_ticks=y_ticks,
        x_label="action token",
        y_label="layer",
    )

    peak_row, peak_step = np.unravel_index(np.nanargmax(abs_mat), abs_mat.shape)
    final_delta = float(np.nanmean(delta[:, -1])) if delta.size else 0.0
    lines = [
        f"peak layer={layers[peak_row]} denoise={peak_step} abs_mean={abs_mat[peak_row, peak_step]:.3f}",
        f"mean final-vs-initial delta={final_delta:+.3f}",
        "fixed global scale across video; magnitude uses log1p to avoid late-layer washout",
    ]
    cv2.rectangle(panel, (0, 492), (PANEL_W, PANEL_H), (16, 24, 39), -1)
    text_lines(panel, lines[:2], origin=(24, 514), scale=0.46, color=MUTED, line_height=22)
    return panel


def open_file(path: Path) -> None:
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        elif sys.platform.startswith("linux"):
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception:
        pass


def render_canvas(
    frame: np.ndarray,
    *,
    label: str,
    frame_idx: int,
    n_action_steps: int,
    metrics: dict[str, Any],
    capture: dict[str, Any],
    scales: dict[str, float],
) -> np.ndarray:
    chunk = chunk_for_frame(frame_idx, n_action_steps)
    phase = phase_for_frame(frame_idx, n_action_steps)
    task = capture["chunk_meta"].get(chunk, {}).get("task", "")
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    canvas[:PANEL_H, :PANEL_W] = render_rollout(frame, label, frame_idx, chunk, phase, metrics, task)
    canvas[:PANEL_H, PANEL_W:] = render_inputs(capture, chunk)
    canvas[PANEL_H:, :PANEL_W] = render_action(capture, chunk, phase, n_action_steps, scales)
    canvas[PANEL_H:, PANEL_W:] = render_activation(capture, chunk, scales)
    cv2.line(canvas, (PANEL_W, 0), (PANEL_W, CANVAS_H), (3, 7, 18), 2)
    cv2.line(canvas, (0, PANEL_H), (CANVAS_W, PANEL_H), (3, 7, 18), 2)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a diagnostic activation video for Pi0.5 LIBERO runs.")
    parser.add_argument("--run", default="latest", help="Run id, run directory, or 'latest'.")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--n-action-steps", type=int, default=10)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--out", default="")
    parser.add_argument("--preview-frame", type=int, default=-1)
    parser.add_argument("--open", action="store_true")
    parser.add_argument("--open-preview", action="store_true")
    args = parser.parse_args()

    run_dir = resolve_run(args.run)
    video_path = find_task_video(run_dir, args.task_id, args.episode)
    metrics = read_json(run_dir / "eval_info.json").get("overall") or parse_metrics_from_log(run_dir / "run.log")
    capture = read_capture(run_dir)
    scales = build_scales(capture)

    out_dir = run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / f"task_{args.task_id}_episode_{args.episode}_analysis.mp4"
    preview_path = out_path.with_name(f"{out_path.stem}_frame{max(args.preview_frame, 0):04d}.png")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (CANVAS_W, CANVAS_H))
    if not writer.isOpened():
        raise SystemExit(f"Could not write video: {out_path}")

    label = f"{video_path.parent.name} episode={args.episode}"
    frame_idx = 0
    saved_preview = False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        canvas = render_canvas(
            frame,
            label=label,
            frame_idx=frame_idx,
            n_action_steps=args.n_action_steps,
            metrics=metrics,
            capture=capture,
            scales=scales,
        )
        writer.write(canvas)
        if args.preview_frame >= 0 and frame_idx == args.preview_frame:
            cv2.imwrite(str(preview_path), canvas)
            saved_preview = True
        frame_idx += 1

    cap.release()
    writer.release()
    print(out_path)
    if args.preview_frame >= 0:
        if not saved_preview:
            print(f"Preview frame {args.preview_frame} was outside the video length", file=sys.stderr)
        else:
            print(preview_path)
            if args.open_preview:
                open_file(preview_path)
    if args.open:
        open_file(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

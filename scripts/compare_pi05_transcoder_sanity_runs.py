#!/usr/bin/env python
"""Compare Pi0.5 original/probe and transcoder-replace LIBERO sanity runs.

The script auto-discovers the latest run folders, summarizes eval success,
checks transcoder trace completeness, compares action chunks and layer latent
activity, and writes a small HTML report plus machine-readable artifacts.
"""

from __future__ import annotations

import argparse
import collections
import csv
import html
import json
import math
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - local minimal Python envs may omit plotting deps.
    plt = None


DEFAULT_BASE_DIRS = [
    Path("outputs/eval/pi05_libero"),
    Path("/content/groot-run/outputs/eval/pi05_libero"),
    Path("/content/drive/MyDrive/groot-run-shared-programmer908/outputs/eval/pi05_libero"),
]


@dataclass
class ActionChunk:
    chunk: int
    values: np.ndarray
    norm_per_step: list[float] = field(default_factory=list)
    abs_per_dim: list[float] = field(default_factory=list)


@dataclass
class RunSummary:
    label: str
    path: Path
    eval_info: dict[str, Any]
    pc_success: float | None
    n_episodes: int | None
    eval_s: float | None
    videos: list[Path]
    event_counts: dict[str, int]
    layers: list[int]
    chunks: list[int]
    action_chunks: dict[int, ActionChunk]
    layer_l1_mean: dict[int, float]
    layer_l0_mean: dict[int, float]
    diffusion_counts: dict[str, int]
    first_success_step: int | None
    max_rollout_steps: int | None
    useful_log_tail: list[str]
    warnings: list[str] = field(default_factory=list)

    @property
    def success_count(self) -> int | None:
        if self.pc_success is None or self.n_episodes is None:
            return None
        return int(round((self.pc_success / 100.0) * self.n_episodes))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        action="append",
        default=[],
        help="Base directory containing original-probe-sanity/ and replace-sanity/. May be passed more than once.",
    )
    parser.add_argument("--original-label", default="original-probe-sanity")
    parser.add_argument("--replace-label", default="replace-sanity")
    parser.add_argument("--pure-original-label", default="original-sanity")
    parser.add_argument("--original-run", type=Path, default=None, help="Explicit original/probe run directory.")
    parser.add_argument("--replace-run", type=Path, default=None, help="Explicit replace run directory.")
    parser.add_argument("--pure-original-run", type=Path, default=None, help="Optional pure original run directory.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-action-steps", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level for confidence intervals.")
    parser.add_argument("--make-video", action="store_true", help="Try to render a side-by-side mp4 with ffmpeg.")
    parser.add_argument("--no-video", action="store_true", help="Disable side-by-side video even if --make-video is set.")
    return parser.parse_args()


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return str(value)


def find_latest_run(label: str, base_dirs: list[Path]) -> Path | None:
    candidates: list[Path] = []
    for base in base_dirs:
        root = base / label
        if root.exists():
            candidates.extend(path for path in root.iterdir() if path.is_dir())
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: path.stat().st_mtime)[-1]


def load_eval_info(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "eval_info.json"
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def _overall(info: dict[str, Any]) -> dict[str, Any]:
    overall = info.get("overall")
    return overall if isinstance(overall, dict) else {}


def _find_videos(run_dir: Path) -> list[Path]:
    video_root = run_dir / "videos"
    if not video_root.exists():
        return []
    return sorted(video_root.glob("**/*.mp4"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _parse_log(run_dir: Path) -> tuple[int | None, int | None, list[str]]:
    log_path = run_dir / "run.log"
    if not log_path.exists():
        return None, None, []
    lines = log_path.read_text(errors="replace").splitlines()
    useful_keys = [
        "Overall Aggregated Metrics",
        "running_success_rate",
        "End of eval",
        "Traceback",
        "Error",
        "Exception",
        "Saved results",
    ]
    useful = [line for line in lines if any(key in line for key in useful_keys)]
    first_success_step = None
    max_steps = None
    progress_re = re.compile(r"Running rollout with at most\s+(\d+)\s+steps:.*?\|\s*(\d+)/(?:\s*)?(\d+).*?running_success_rate=([0-9.]+)%")
    for line in lines:
        match = progress_re.search(line)
        if not match:
            continue
        max_steps = _safe_int(match.group(1)) or max_steps
        step = _safe_int(match.group(2))
        success = _safe_float(match.group(4))
        if first_success_step is None and step is not None and success is not None and success > 0:
            first_success_step = step
    return first_success_step, max_steps, useful[-30:]


def _events_summary(events: list[dict[str, Any]]) -> tuple[
    dict[str, int],
    list[int],
    list[int],
    dict[int, ActionChunk],
    dict[int, float],
    dict[int, float],
    dict[str, int],
]:
    counts = collections.Counter()
    layers = collections.Counter()
    chunks: set[int] = set()
    action_chunks: dict[int, ActionChunk] = {}
    layer_l1_values: dict[int, list[float]] = collections.defaultdict(list)
    layer_l0_values: dict[int, list[float]] = collections.defaultdict(list)
    diffusion_counts = collections.Counter()

    for event in events:
        typ = str(event.get("type"))
        counts[typ] += 1
        if typ == "transcoder_latent":
            layer = _safe_int(event.get("layer"))
            chunk = _safe_int(event.get("chunk"))
            if layer is not None:
                layers[layer] += 1
                l1 = _safe_float(event.get("l1_mean"))
                l0 = _safe_float(event.get("l0_mean"))
                if l1 is not None:
                    layer_l1_values[layer].append(l1)
                if l0 is not None:
                    layer_l0_values[layer].append(l0)
            if chunk is not None:
                chunks.add(chunk)
        elif typ == "action_chunk":
            chunk = _safe_int(event.get("chunk"))
            values = event.get("values")
            if chunk is not None and isinstance(values, list):
                action_chunks[chunk] = ActionChunk(
                    chunk=chunk,
                    values=np.asarray(values, dtype=np.float64),
                    norm_per_step=[float(item) for item in event.get("norm_per_step", [])],
                    abs_per_dim=[float(item) for item in event.get("abs_per_dim", [])],
                )
        elif typ == "diffusion_state":
            diffusion_counts[str(event.get("kind", "unknown"))] += 1

    layer_l1_mean = {layer: float(np.mean(values)) for layer, values in layer_l1_values.items() if values}
    layer_l0_mean = {layer: float(np.mean(values)) for layer, values in layer_l0_values.items() if values}
    return (
        dict(counts),
        sorted(layers),
        sorted(chunks),
        action_chunks,
        layer_l1_mean,
        layer_l0_mean,
        dict(sorted(diffusion_counts.items())),
    )


def summarize_run(label: str, run_dir: Path) -> RunSummary:
    info = load_eval_info(run_dir)
    overall = _overall(info)
    events = _read_jsonl(run_dir / "transcoder_capture" / "events.jsonl")
    (
        event_counts,
        layers,
        chunks,
        action_chunks,
        layer_l1_mean,
        layer_l0_mean,
        diffusion_counts,
    ) = _events_summary(events)
    first_success_step, max_rollout_steps, useful_log_tail = _parse_log(run_dir)
    return RunSummary(
        label=label,
        path=run_dir,
        eval_info=info,
        pc_success=_safe_float(overall.get("pc_success")),
        n_episodes=_safe_int(overall.get("n_episodes")),
        eval_s=_safe_float(overall.get("eval_s")),
        videos=_find_videos(run_dir),
        event_counts=event_counts,
        layers=layers,
        chunks=chunks,
        action_chunks=action_chunks,
        layer_l1_mean=layer_l1_mean,
        layer_l0_mean=layer_l0_mean,
        diffusion_counts=diffusion_counts,
        first_success_step=first_success_step,
        max_rollout_steps=max_rollout_steps,
        useful_log_tail=useful_log_tail,
    )


def wilson_interval(successes: int, n: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (math.nan, math.nan)
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    margin = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _comb(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def mcnemar_exact_p(discordant_ab: int, discordant_ba: int) -> float | None:
    n = discordant_ab + discordant_ba
    if n == 0:
        return 1.0
    tail = sum(_comb(n, i) for i in range(0, min(discordant_ab, discordant_ba) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    # Table: [[a, b], [c, d]]. Two-sided p by summing tables with probability <= observed.
    row1 = a + b
    row2 = c + d
    col1 = a + c
    total = row1 + row2

    def hypergeom(x: int) -> float:
        return (_comb(col1, x) * _comb(total - col1, row1 - x)) / _comb(total, row1)

    min_x = max(0, row1 - (total - col1))
    max_x = min(row1, col1)
    observed = hypergeom(a)
    return min(1.0, sum(hypergeom(x) for x in range(min_x, max_x + 1) if hypergeom(x) <= observed + 1e-15))


def significance_summary(original: RunSummary, replace: RunSummary) -> dict[str, Any]:
    orig_n = original.n_episodes or 0
    repl_n = replace.n_episodes or 0
    orig_successes = original.success_count
    repl_successes = replace.success_count
    result: dict[str, Any] = {
        "original_n": orig_n,
        "replace_n": repl_n,
        "original_successes": orig_successes,
        "replace_successes": repl_successes,
        "interpretation": "insufficient data",
    }
    if orig_successes is None or repl_successes is None or orig_n <= 0 or repl_n <= 0:
        return result

    result["original_wilson_95"] = wilson_interval(orig_successes, orig_n)
    result["replace_wilson_95"] = wilson_interval(repl_successes, repl_n)
    result["original_rate"] = orig_successes / orig_n
    result["replace_rate"] = repl_successes / repl_n

    if orig_n == repl_n:
        # Without per-episode identity in eval_info, assume aggregate-only paired
        # data. If the aggregate counts are equal, discordants are unknown but
        # the observed aggregate difference is zero.
        if orig_successes == repl_successes:
            result["paired_note"] = (
                "Aggregate success counts match. Per-episode discordants are not available, "
                "so an exact paired test cannot prove equivalence."
            )
            result["mcnemar_exact_p"] = 1.0
        else:
            result["paired_note"] = (
                "Per-episode identities are not available. Report Fisher exact on aggregate counts only."
            )

    result["fisher_exact_p_aggregate"] = fisher_exact_two_sided(
        orig_successes,
        orig_n - orig_successes,
        repl_successes,
        repl_n - repl_successes,
    )

    if min(orig_n, repl_n) < 5:
        result["interpretation"] = (
            "Descriptive sanity check only. n is too small for a meaningful statistical significance claim."
        )
    elif result["fisher_exact_p_aggregate"] < 0.05:
        result["interpretation"] = "Aggregate success rates differ at p < 0.05."
    else:
        result["interpretation"] = (
            "No statistically significant aggregate success-rate difference detected. "
            "This is not an equivalence proof."
        )
    return result


def trace_completeness_summary(run: RunSummary) -> dict[str, Any]:
    chunk_count = len(run.chunks)
    layer_count = len(run.layers)
    latent_records = run.event_counts.get("transcoder_latent", 0)
    inferred_steps = None
    expected_latents = None
    if chunk_count and layer_count:
        inferred_steps = latent_records / float(chunk_count * layer_count)
        expected_latents = chunk_count * layer_count * round(inferred_steps)
    return {
        "chunks": chunk_count,
        "layers": layer_count,
        "latent_records": latent_records,
        "inferred_denoise_steps": inferred_steps,
        "expected_latents_at_rounded_steps": expected_latents,
        "latent_record_ratio": None if not expected_latents else latent_records / expected_latents,
        "diffusion_counts": run.diffusion_counts,
    }


def compare_actions(original: RunSummary, replace: RunSummary, *, n_action_steps: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for chunk in sorted(set(original.action_chunks) & set(replace.action_chunks)):
        orig = original.action_chunks[chunk].values
        repl = replace.action_chunks[chunk].values
        rows.append(_compare_action_array(chunk, orig, repl, n_action_steps=n_action_steps))
    return rows


def action_comparison_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if "error" not in row]
    if not valid:
        return {"paired_chunks": 0}

    def values(key: str) -> np.ndarray:
        return np.asarray([float(row[key]) for row in valid if row.get(key) is not None], dtype=np.float64)

    result: dict[str, Any] = {"paired_chunks": len(valid)}
    for key in ["rmse", "mae", "max_abs", "cosine", "executed_rmse", "executed_mae"]:
        arr = values(key)
        if arr.size == 0:
            continue
        result[f"{key}_mean"] = float(arr.mean())
        result[f"{key}_median"] = float(np.median(arr))
        result[f"{key}_max"] = float(arr.max())
        result[f"{key}_std"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return result


def _compare_action_array(chunk: int, original: np.ndarray, replace: np.ndarray, *, n_action_steps: int) -> dict[str, Any]:
    steps = min(original.shape[0], replace.shape[0])
    dims = min(original.shape[1] if original.ndim > 1 else 0, replace.shape[1] if replace.ndim > 1 else 0)
    if steps == 0 or dims == 0:
        return {"chunk": chunk, "error": "empty action arrays"}
    orig = original[:steps, :dims]
    repl = replace[:steps, :dims]
    diff = repl - orig
    flat_o = orig.reshape(-1)
    flat_r = repl.reshape(-1)
    denom = float(np.linalg.norm(flat_o) * np.linalg.norm(flat_r))
    executed_steps = min(n_action_steps, steps)
    executed_diff = diff[:executed_steps]
    return {
        "chunk": chunk,
        "steps": steps,
        "dims": dims,
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "mae": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "cosine": None if denom == 0 else float(np.dot(flat_o, flat_r) / denom),
        "executed_rmse": float(np.sqrt(np.mean(executed_diff**2))),
        "executed_mae": float(np.mean(np.abs(executed_diff))),
        "original_norm_mean": float(np.mean(np.linalg.norm(orig, axis=-1))),
        "replace_norm_mean": float(np.mean(np.linalg.norm(repl, axis=-1))),
    }


def compare_layers(original: RunSummary, replace: RunSummary) -> list[dict[str, Any]]:
    rows = []
    for layer in sorted(set(original.layer_l1_mean) | set(replace.layer_l1_mean)):
        orig_l1 = original.layer_l1_mean.get(layer)
        repl_l1 = replace.layer_l1_mean.get(layer)
        orig_l0 = original.layer_l0_mean.get(layer)
        repl_l0 = replace.layer_l0_mean.get(layer)
        rows.append(
            {
                "layer": layer,
                "original_l1_mean": orig_l1,
                "replace_l1_mean": repl_l1,
                "l1_delta": None if orig_l1 is None or repl_l1 is None else repl_l1 - orig_l1,
                "l1_ratio": None if orig_l1 in (None, 0) or repl_l1 is None else repl_l1 / orig_l1,
                "original_l0_mean": orig_l0,
                "replace_l0_mean": repl_l0,
                "l0_delta": None if orig_l0 is None or repl_l0 is None else repl_l0 - orig_l0,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_action_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if plt is None:
        return
    if not rows:
        return
    chunks = [row["chunk"] for row in rows]
    rmse = [row.get("rmse") or 0.0 for row in rows]
    executed = [row.get("executed_rmse") or 0.0 for row in rows]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(chunks, rmse, marker="o", label="full 50-step chunk RMSE")
    ax.plot(chunks, executed, marker="o", label="executed first-window RMSE")
    ax.set_xlabel("policy chunk")
    ax.set_ylabel("action RMSE")
    ax.set_title("Original/probe vs replace action chunk distance")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_layer_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if plt is None:
        return
    if not rows:
        return
    layers = [row["layer"] for row in rows]
    deltas = [row.get("l1_delta") or 0.0 for row in rows]
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#2f6fba" if value >= 0 else "#b05a2a" for value in deltas]
    ax.bar(layers, deltas, color=colors)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_xlabel("action-expert layer")
    ax.set_ylabel("replace - original/probe L1 mean")
    ax.set_title("Transcoder latent L1 difference by layer")
    ax.set_xticks(layers)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_success(path: Path, original: RunSummary, replace: RunSummary, stats: dict[str, Any]) -> None:
    if plt is None:
        return
    if original.success_count is None or replace.success_count is None or not original.n_episodes or not replace.n_episodes:
        return
    labels = ["original/probe", "replace"]
    rates = [original.success_count / original.n_episodes, replace.success_count / replace.n_episodes]
    intervals = [stats.get("original_wilson_95"), stats.get("replace_wilson_95")]
    lower = [0.0 if item is None else rates[idx] - item[0] for idx, item in enumerate(intervals)]
    upper = [0.0 if item is None else item[1] - rates[idx] for idx, item in enumerate(intervals)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, rates, color=["#4f7cac", "#5e9f69"])
    ax.errorbar(labels, rates, yerr=[lower, upper], fmt="none", color="#222222", capsize=6)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("success rate")
    ax.set_title("Success rate with Wilson 95% interval")
    for idx, rate in enumerate(rates):
        ax.text(idx, min(1.02, rate + 0.04), f"{rate:.0%}", ha="center")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _relative_link(path: Path, root: Path) -> str:
    try:
        return html.escape(str(path.relative_to(root)))
    except ValueError:
        return html.escape(str(path))


def _html_table(rows: list[dict[str, Any]], *, limit: int | None = None) -> str:
    if not rows:
        return "<p>No rows.</p>"
    shown = rows if limit is None else rows[:limit]
    fieldnames = list(shown[0].keys())
    parts = ["<table><thead><tr>"]
    parts.extend(f"<th>{html.escape(str(field))}</th>" for field in fieldnames)
    parts.append("</tr></thead><tbody>")
    for row in shown:
        parts.append("<tr>")
        for field in fieldnames:
            value = row.get(field)
            if isinstance(value, float):
                text = f"{value:.6g}"
            else:
                text = "" if value is None else str(value)
            parts.append(f"<td>{html.escape(text)}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def _run_row(run: RunSummary) -> dict[str, Any]:
    return {
        "label": run.label,
        "path": str(run.path),
        "success": run.pc_success,
        "episodes": run.n_episodes,
        "eval_s": run.eval_s,
        "first_success_step": run.first_success_step,
        "videos": len(run.videos),
        "events": bool(run.event_counts),
        "chunks": len(run.chunks),
        "latent_records": run.event_counts.get("transcoder_latent", 0),
        "layers": ",".join(str(item) for item in run.layers),
    }


def _render_html(
    path: Path,
    *,
    original: RunSummary,
    replace: RunSummary,
    pure_original: RunSummary | None,
    action_rows: list[dict[str, Any]],
    layer_rows: list[dict[str, Any]],
    stats: dict[str, Any],
    artifacts: dict[str, Path],
    verdict: str,
) -> None:
    run_rows = [_run_row(original), _run_row(replace)]
    if pure_original is not None:
        run_rows.insert(0, _run_row(pure_original))
    video_links = []
    original_video = artifacts.get("original_video") or (original.videos[0] if original.videos else None)
    replace_video = artifacts.get("replace_video") or (replace.videos[0] if replace.videos else None)
    if original_video:
        video_links.append(f"<li>Original/probe video: <a href='{_relative_link(original_video, path.parent)}'>{html.escape(original_video.name)}</a></li>")
    if replace_video:
        video_links.append(f"<li>Replace video: <a href='{_relative_link(replace_video, path.parent)}'>{html.escape(replace_video.name)}</a></li>")
    if artifacts.get("side_by_side_video") and artifacts["side_by_side_video"].exists():
        link = _relative_link(artifacts["side_by_side_video"], path.parent)
        video_links.append(f"<li>Side-by-side video: <a href='{link}'>{html.escape(artifacts['side_by_side_video'].name)}</a></li>")
    image_parts = []
    for key, title in [
        ("success_plot", "Success Rate"),
        ("action_plot", "Action Chunk Distance"),
        ("layer_plot", "Layer L1 Difference"),
    ]:
        artifact = artifacts.get(key)
        if artifact and artifact.exists():
            image_parts.append(f"<h2>{title}</h2><img src='{_relative_link(artifact, path.parent)}' alt='{html.escape(title)}'>")
    action_summary = action_comparison_summary(action_rows)
    trace_rows = [
        {"label": "original/probe", **trace_completeness_summary(original)},
        {"label": "replace", **trace_completeness_summary(replace)},
    ]

    html_text = "\n".join(
        [
            "<!doctype html>",
            "<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>",
            "<title>Pi0.5 Transcoder Sanity Comparison</title>",
            "<style>",
            "body{font-family:Arial,Helvetica,sans-serif;margin:24px;color:#15171a;background:white}",
            "h1{margin-bottom:4px} h2{margin-top:28px}",
            "p{max-width:980px;line-height:1.45;color:#374151}",
            "table{border-collapse:collapse;width:100%;font-size:13px;margin-top:10px}",
            "th,td{border-bottom:1px solid #e5e9f0;padding:7px 8px;text-align:left;vertical-align:top}",
            "th{background:#f5f7fa;color:#273447}",
            "code{background:#f4f6f8;padding:1px 4px;border-radius:4px}",
            "img{max-width:1100px;width:100%;height:auto;border:1px solid #d8dee8}",
            ".verdict{padding:12px 14px;border-left:4px solid #2f6fba;background:#f5f9ff;max-width:980px}",
            ".warn{color:#8a4b00}",
            "</style></head><body>",
            "<h1>Pi0.5 Transcoder Sanity Comparison</h1>",
            f"<p>Generated: {html.escape(time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()))}</p>",
            f"<div class='verdict'><b>Verdict:</b> {html.escape(verdict)}</div>",
            "<h2>Runs</h2>",
            _html_table(run_rows),
            "<h2>Statistical Significance</h2>",
            f"<p>{html.escape(str(stats.get('interpretation', 'n/a')))}</p>",
            _html_table([{key: value for key, value in stats.items() if key != "interpretation"}]),
            "<h2>Trace Completeness</h2>",
            _html_table(trace_rows),
            "<h2>Action Error Summary</h2>",
            "<p>These chunk metrics compare original/probe and replace chunks by chunk index from the closed-loop rollouts. For a stricter same-observation estimate, run the paired action-equivalence probe.</p>",
            _html_table([action_summary]),
            "<h2>Videos</h2>",
            "<ul>" + "\n".join(video_links or ["<li>No videos found.</li>"]) + "</ul>",
            *image_parts,
            "<h2>Action Chunk Comparison</h2>",
            _html_table(action_rows),
            "<h2>Layer L1 Comparison</h2>",
            _html_table(layer_rows),
            "<h2>Useful Log Tail</h2>",
            "<h3>Original/probe</h3><pre>" + html.escape("\n".join(original.useful_log_tail[-20:])) + "</pre>",
            "<h3>Replace</h3><pre>" + html.escape("\n".join(replace.useful_log_tail[-20:])) + "</pre>",
            "</body></html>",
        ]
    )
    path.write_text(html_text, encoding="utf-8")


def _copy_first_video(run: RunSummary, output_dir: Path, prefix: str) -> Path | None:
    if not run.videos:
        return None
    target = output_dir / f"{prefix}_{run.videos[0].name}"
    if run.videos[0].resolve() != target.resolve():
        shutil.copy2(run.videos[0], target)
    return target


def _make_side_by_side_video(original_video: Path | None, replace_video: Path | None, output_path: Path) -> bool:
    if original_video is None or replace_video is None:
        return False
    if shutil.which("ffmpeg") is None:
        return False
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(original_video),
        "-i",
        str(replace_video),
        "-filter_complex",
        "[0:v]scale=640:-2,setpts=PTS-STARTPTS[left];[1:v]scale=640:-2,setpts=PTS-STARTPTS[right];[left][right]hstack=inputs=2[v]",
        "-map",
        "[v]",
        "-an",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output_path.exists()
    except Exception:
        return False


def _default_output_dir(base_dirs: list[Path]) -> Path:
    for base in base_dirs:
        if base.exists():
            return base / "sanity-comparison" / time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    return Path("outputs/eval/pi05_libero/sanity-comparison") / time.strftime("%Y%m%d-%H%M%S", time.gmtime())


def _verdict(original: RunSummary, replace: RunSummary, action_rows: list[dict[str, Any]], stats: dict[str, Any]) -> str:
    original_ok = original.pc_success == 100.0
    replace_ok = replace.pc_success == 100.0
    if original_ok and replace_ok:
        chunk_text = "unknown action distance"
        if action_rows:
            mean_rmse = float(np.mean([row.get("executed_rmse") or 0.0 for row in action_rows]))
            mean_cos = float(np.mean([row.get("cosine") for row in action_rows if row.get("cosine") is not None]))
            chunk_text = f"mean executed-window RMSE {mean_rmse:.4g}, mean chunk cosine {mean_cos:.4g}"
        return (
            "Sanity check passed: original/probe and replace both succeeded. "
            f"Trace completeness is comparable; {chunk_text}. "
            f"Statistical note: {stats.get('interpretation', 'n/a')}"
        )
    if original_ok and not replace_ok:
        return "Replacement failed where original/probe succeeded. Treat as a replacement-fidelity problem."
    if not original_ok and replace_ok:
        return "Replace succeeded while original/probe did not; rerun because the baseline did not validate the task."
    return "Both runs failed. This task/run is not useful as a replacement sanity check."


def main() -> None:
    args = parse_args()
    base_dirs = args.base_dir or DEFAULT_BASE_DIRS
    original_run = args.original_run or find_latest_run(args.original_label, base_dirs)
    replace_run = args.replace_run or find_latest_run(args.replace_label, base_dirs)
    pure_original_run = args.pure_original_run or find_latest_run(args.pure_original_label, base_dirs)
    if original_run is None:
        raise FileNotFoundError(f"Could not find run for {args.original_label!r} under {[str(p) for p in base_dirs]}")
    if replace_run is None:
        raise FileNotFoundError(f"Could not find run for {args.replace_label!r} under {[str(p) for p in base_dirs]}")

    output_dir = args.output_dir or _default_output_dir(base_dirs)
    output_dir.mkdir(parents=True, exist_ok=True)

    original = summarize_run(args.original_label, original_run)
    replace = summarize_run(args.replace_label, replace_run)
    pure_original = summarize_run(args.pure_original_label, pure_original_run) if pure_original_run else None
    action_rows = compare_actions(original, replace, n_action_steps=args.n_action_steps)
    action_summary = action_comparison_summary(action_rows)
    layer_rows = compare_layers(original, replace)
    stats = significance_summary(original, replace)
    verdict = _verdict(original, replace, action_rows, stats)

    action_csv = output_dir / "action_chunk_comparison.csv"
    layer_csv = output_dir / "layer_l1_comparison.csv"
    summary_json = output_dir / "sanity_comparison_summary.json"
    report_html = output_dir / "sanity_comparison_report.html"
    action_plot = output_dir / "action_rmse_by_chunk.png"
    layer_plot = output_dir / "layer_l1_delta.png"
    success_plot = output_dir / "success_rate_ci.png"

    _write_csv(action_csv, action_rows)
    _write_csv(layer_csv, layer_rows)
    _plot_action_rows(action_plot, action_rows)
    _plot_layer_rows(layer_plot, layer_rows)
    _plot_success(success_plot, original, replace, stats)

    copied_original_video = _copy_first_video(original, output_dir, "original_probe")
    copied_replace_video = _copy_first_video(replace, output_dir, "replace")
    side_by_side_video = output_dir / "original_probe_vs_replace.mp4"
    if args.make_video and not args.no_video:
        _make_side_by_side_video(copied_original_video, copied_replace_video, side_by_side_video)

    artifacts = {
        "html": report_html,
        "summary_json": summary_json,
        "action_csv": action_csv,
        "layer_csv": layer_csv,
        "action_plot": action_plot,
        "layer_plot": layer_plot,
        "success_plot": success_plot,
    }
    if copied_original_video is not None:
        artifacts["original_video"] = copied_original_video
    if copied_replace_video is not None:
        artifacts["replace_video"] = copied_replace_video
    if side_by_side_video.exists():
        artifacts["side_by_side_video"] = side_by_side_video

    payload = {
        "verdict": verdict,
        "original": _run_row(original),
        "replace": _run_row(replace),
        "pure_original": _run_row(pure_original) if pure_original else None,
        "significance": stats,
        "trace_completeness": {
            "original": trace_completeness_summary(original),
            "replace": trace_completeness_summary(replace),
        },
        "action_summary": action_summary,
        "action_comparison": action_rows,
        "layer_comparison": layer_rows,
        "artifacts": {key: str(value) for key, value in artifacts.items()},
    }
    summary_json.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    _render_html(
        report_html,
        original=original,
        replace=replace,
        pure_original=pure_original,
        action_rows=action_rows,
        layer_rows=layer_rows,
        stats=stats,
        artifacts=artifacts,
        verdict=verdict,
    )

    print(verdict)
    print("Original/probe run:", original.path)
    print("Replace run:", replace.path)
    if pure_original:
        print("Pure original run:", pure_original.path)
    print("Report:", report_html)
    print("Summary:", summary_json)
    print("Action comparison:", action_csv)
    print("Layer comparison:", layer_csv)


if __name__ == "__main__":
    main()

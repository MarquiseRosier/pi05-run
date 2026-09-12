#!/usr/bin/env python
"""Render a DifFRACT-style feature inspection dashboard for Pi0.5 transcoders."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, default=None)
    parser.add_argument("--candidate-csv", type=Path, default=None)
    parser.add_argument("--candidate-json", type=Path, default=None)
    parser.add_argument("--policy-path", default="lerobot/pi05_libero_finetuned")
    parser.add_argument("--max-features", type=int, default=200)
    parser.add_argument("--top-examples", type=int, default=20)
    parser.add_argument(
        "--features",
        default=None,
        help="Explicit layer:timestep:feature or layer:feature list, e.g. `12:0.4:7342,7:4317`.",
    )
    parser.add_argument(
        "--sort-by",
        choices=("interesting", "max", "topk_mean", "mean", "std", "frequency", "top_m_frequency"),
        default="interesting",
    )
    parser.add_argument("--min-frequency", type=float, default=0.0)
    parser.add_argument("--max-frequency", type=float, default=1.0)
    parser.add_argument("--min-max-score", type=float, default=0.0)
    parser.add_argument("--save-thumbnails", action="store_true")
    parser.add_argument("--thumbnail-camera", default=None, help="Camera key to render. Default uses first dataset camera.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--policy-dtype", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _read_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    observations: dict[int, dict[str, Any]] = {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            observations[int(row["observation_id"])] = row
    return observations


def _parse_features(raw: str) -> list[tuple[int, str | None, int]]:
    features: list[tuple[int, str | None, int]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) == 2:
            layer, feature = parts
            features.append((int(layer), None, int(feature)))
        elif len(parts) == 3:
            layer, timestep, feature = parts
            features.append((int(layer), f"{float(timestep):.8f}", int(feature)))
        else:
            raise ValueError(f"Expected layer:feature or layer:timestep:feature, got {item!r}")
    return features


def _layer_name_for_index(topk_payload: dict[str, Any], layer: int) -> str:
    for name in topk_payload["layer_names"]:
        if int(topk_payload["layer_indices"][name]) == layer:
            return name
    raise KeyError(f"No layer {layer} in feature artifact")


def _timesteps_for_feature(topk_payload: dict[str, Any], name: str, timestep: str | None) -> list[str]:
    timesteps = sorted(topk_payload["topk"][name])
    if timestep is None:
        return timesteps
    if timestep in topk_payload["topk"][name]:
        return [timestep]
    raise KeyError(f"No timestep {float(timestep):.4g} for {name}")


def _metric_tensor(
    *,
    topk_payload: dict[str, Any],
    stats_payload: dict[str, Any],
    name: str,
    timestep: str,
    sort_by: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    store = topk_payload["topk"][name][timestep]
    scores = store["scores"].float()
    finite_scores = torch.where(torch.isfinite(scores), scores, torch.zeros_like(scores))
    valid = torch.isfinite(scores)
    valid_count = valid.sum(dim=1).clamp_min(1)
    topk_mean = finite_scores.sum(dim=1) / valid_count
    max_score = torch.where(valid[:, 0], scores[:, 0], torch.zeros_like(scores[:, 0]))

    stats = stats_payload["stats"][name][timestep]
    mean = stats["mean"].float()
    std = stats["std"].float()
    frequency = stats["firing_frequency"].float()
    top_m_frequency = stats.get("top_m_frequency", torch.zeros_like(frequency)).float()

    metrics = {
        "max": max_score,
        "topk_mean": topk_mean,
        "mean": mean,
        "std": std,
        "frequency": frequency,
        "top_m_frequency": top_m_frequency,
        "interesting": topk_mean / (std + 1e-6),
        "valid_topk": valid.sum(dim=1).float(),
    }
    return metrics, metrics[sort_by]


def _single_feature_metrics(
    *,
    topk_payload: dict[str, Any],
    stats_payload: dict[str, Any],
    name: str,
    timestep: str,
    feature: int,
    sort_by: str,
) -> dict[str, float | int]:
    metrics, rank_tensor = _metric_tensor(
        topk_payload=topk_payload,
        stats_payload=stats_payload,
        name=name,
        timestep=timestep,
        sort_by=sort_by,
    )
    stats = stats_payload["stats"][name][timestep]
    return {
        "rank_score": float(rank_tensor[feature]),
        "max": float(metrics["max"][feature]),
        "topk_mean": float(metrics["topk_mean"][feature]),
        "mean": float(metrics["mean"][feature]),
        "std": float(metrics["std"][feature]),
        "frequency": float(metrics["frequency"][feature]),
        "top_m_frequency": float(metrics["top_m_frequency"][feature]),
        "valid_topk": int(metrics["valid_topk"][feature]),
        "count": int(stats["count"]),
        "active_count": int(stats["active_count"][feature]),
        "top_m_count": int(stats.get("top_m_count", torch.zeros_like(stats["active_count"]))[feature]),
    }


def _candidate_row(
    *,
    topk_payload: dict[str, Any],
    stats_payload: dict[str, Any],
    name: str,
    layer: int,
    timestep: str,
    feature: int,
    sort_by: str,
) -> dict[str, Any]:
    values = _single_feature_metrics(
        topk_payload=topk_payload,
        stats_payload=stats_payload,
        name=name,
        timestep=timestep,
        feature=feature,
        sort_by=sort_by,
    )
    row: dict[str, Any] = {
        "rank": 0,
        "layer": layer,
        "layer_name": name,
        "timestep_key": timestep,
        "timestep": float(timestep),
        "feature": int(feature),
        "sort_by": sort_by,
    }
    row.update(values)
    row["feature_key"] = f"L{layer:02d}:tau{float(timestep):.4g}:F{feature}"
    return row


def _rank_features(
    *,
    topk_payload: dict[str, Any],
    stats_payload: dict[str, Any],
    sort_by: str,
    max_features: int,
    min_frequency: float,
    max_frequency: float,
    min_max_score: float,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    per_bucket_k = max(1, min(max_features, int(topk_payload["d_features"])))
    for name in topk_payload["layer_names"]:
        layer = int(topk_payload["layer_indices"][name])
        for timestep in sorted(topk_payload["topk"][name]):
            metrics, ranking_scores = _metric_tensor(
                topk_payload=topk_payload,
                stats_payload=stats_payload,
                name=name,
                timestep=timestep,
                sort_by=sort_by,
            )
            mask = (
                (metrics["frequency"] >= min_frequency)
                & (metrics["frequency"] <= max_frequency)
                & (metrics["max"] >= min_max_score)
                & torch.isfinite(ranking_scores)
            )
            ranking_scores = torch.where(mask, ranking_scores, torch.full_like(ranking_scores, -torch.inf))
            top_values, top_indices = torch.topk(ranking_scores, k=per_bucket_k)
            for value, feature in zip(top_values.tolist(), top_indices.tolist(), strict=False):
                if value == float("-inf"):
                    continue
                ranked.append(
                    _candidate_row(
                        topk_payload=topk_payload,
                        stats_payload=stats_payload,
                        name=name,
                        layer=layer,
                        timestep=timestep,
                        feature=int(feature),
                        sort_by=sort_by,
                    )
                )
    ranked.sort(key=lambda item: float(item["rank_score"]), reverse=True)
    ranked = ranked[:max_features]
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def _explicit_features(
    *,
    topk_payload: dict[str, Any],
    stats_payload: dict[str, Any],
    raw_features: str,
    sort_by: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for layer, timestep, feature in _parse_features(raw_features):
        name = _layer_name_for_index(topk_payload, layer)
        for selected_timestep in _timesteps_for_feature(topk_payload, name, timestep):
            selected.append(
                _candidate_row(
                    topk_payload=topk_payload,
                    stats_payload=stats_payload,
                    name=name,
                    layer=layer,
                    timestep=selected_timestep,
                    feature=feature,
                    sort_by=sort_by,
                )
            )
    for rank, row in enumerate(selected, start=1):
        row["rank"] = rank
    return selected


def _dataset_index(dataset: Any, observation: dict[str, Any]) -> int | None:
    if "index" in observation and isinstance(observation["index"], int):
        return observation["index"]
    if "episode_index" not in observation or "frame_index" not in observation:
        return None
    episode = int(observation["episode_index"])
    frame = int(observation["frame_index"])
    from_indices = dataset.meta.episodes["dataset_from_index"]
    absolute_index = int(from_indices[episode]) + frame
    absolute_to_relative = getattr(dataset, "absolute_to_relative_idx", None)
    if absolute_to_relative is None:
        return absolute_index
    return absolute_to_relative.get(absolute_index)


def _tensor_to_image(tensor: torch.Tensor):
    from PIL import Image

    array = tensor.detach().cpu()
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = array.permute(1, 2, 0)
    array = array.float()
    if array.numel() == 0:
        raise ValueError("empty image tensor")
    if float(array.max()) <= 1.5:
        array = array * 255.0
    array = array.clamp(0, 255).byte().numpy()
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[:, :, 0]
    return Image.fromarray(array)


def _save_thumbnail(dataset: Any, observation: dict[str, Any], *, camera_key: str, path: Path) -> str | None:
    index = _dataset_index(dataset, observation)
    if index is None:
        return None
    if path.exists():
        return str(path)
    item = dataset[index]
    if camera_key not in item:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    image = _tensor_to_image(item[camera_key])
    image.thumbnail((224, 168))
    image.save(path, quality=85)
    return str(path)


def _required_thumbnail_observations(
    *,
    topk_payload: dict[str, Any],
    observations: dict[int, dict[str, Any]],
    candidates: list[dict[str, Any]],
    top_examples: int,
) -> dict[int, dict[str, Any]]:
    required: dict[int, dict[str, Any]] = {}
    for candidate in candidates:
        name = candidate["layer_name"]
        timestep = candidate["timestep_key"]
        feature = int(candidate["feature"])
        store = topk_payload["topk"][name][timestep]
        limit = min(top_examples, store["scores"].shape[1])
        for rank in range(limit):
            score = float(store["scores"][feature, rank])
            if not torch.isfinite(torch.tensor(score)):
                continue
            observation_id = int(store["observation_ids"][feature, rank])
            if observation_id < 0:
                continue
            required[observation_id] = observations.get(observation_id, {"observation_id": observation_id})
    return required


def _materialize_thumbnails(
    *,
    dataset: Any,
    camera_key: str,
    observations: dict[int, dict[str, Any]],
    output_html: Path,
) -> dict[int, str]:
    thumb_root = output_html.parent / "feature_report_thumbnails"
    indexed: list[tuple[int, int, dict[str, Any]]] = []
    for observation_id, observation in observations.items():
        dataset_index = _dataset_index(dataset, observation)
        if dataset_index is None:
            continue
        indexed.append((int(dataset_index), observation_id, observation))
    indexed.sort()

    paths: dict[int, str] = {}
    total = len(indexed)
    print(f"saving thumbnails for {total} unique observations", flush=True)
    for offset, (_dataset_index_value, observation_id, observation) in enumerate(indexed, start=1):
        thumb_path = thumb_root / f"observation_{observation_id:08d}.jpg"
        saved = _save_thumbnail(dataset, observation, camera_key=camera_key, path=thumb_path)
        if saved is not None:
            paths[observation_id] = str(Path(saved).relative_to(output_html.parent))
        if offset == 1 or offset % 100 == 0 or offset == total:
            print(f"thumbnail {offset}/{total}", flush=True)
    return paths


def _load_thumbnail_dataset(args: argparse.Namespace, config: dict[str, Any]) -> tuple[Any, str]:
    from lerobot.datasets.factory import make_dataset
    from train_pi05_transcoders import (
        _config_with_episodes,
        _configure_train_config,
        patch_pi05_checkpoint_key_compat,
        patch_transformers_causal_mask_compat,
        resolve_device,
        resolve_policy_dtype,
    )

    patch_transformers_causal_mask_compat()
    patch_pi05_checkpoint_key_compat()
    device = resolve_device(args.device)
    args.resolved_device = device
    args.resolved_policy_dtype = resolve_policy_dtype(args.policy_dtype, device)
    cfg = _configure_train_config(args, episodes=None)
    cfg = _config_with_episodes(cfg, None if config.get("episodes") is None else ",".join(str(x) for x in config["episodes"]))
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    dataset = make_dataset(cfg)
    camera_key = args.thumbnail_camera or dataset.meta.camera_keys[0]
    return dataset, camera_key


def _top_observations(
    *,
    topk_payload: dict[str, Any],
    observations: dict[int, dict[str, Any]],
    candidate: dict[str, Any],
    top_examples: int,
    output_html: Path,
    thumbnail_paths: dict[int, str] | None,
) -> list[dict[str, Any]]:
    name = candidate["layer_name"]
    timestep = candidate["timestep_key"]
    feature = int(candidate["feature"])
    layer = int(candidate["layer"])
    store = topk_payload["topk"][name][timestep]
    limit = min(top_examples, store["scores"].shape[1])
    thumb_root = output_html.parent / "feature_report_thumbnails"
    rows: list[dict[str, Any]] = []
    for rank in range(limit):
        score = float(store["scores"][feature, rank])
        if not torch.isfinite(torch.tensor(score)):
            continue
        observation_id = int(store["observation_ids"][feature, rank])
        if observation_id < 0:
            continue
        observation = observations.get(observation_id, {"observation_id": observation_id})
        position = int(store["action_positions"][feature, rank])
        flow_timestep = float(store["flow_timesteps"][feature, rank])
        image = None if thumbnail_paths is None else thumbnail_paths.get(observation_id)
        rows.append(
            {
                "rank": rank + 1,
                "score": score,
                "observation_id": observation_id,
                "action_position": position,
                "flow_timestep": flow_timestep,
                "episode_index": observation.get("episode_index"),
                "frame_index": observation.get("frame_index"),
                "timestamp": observation.get("timestamp"),
                "task": observation.get("task", ""),
                "image": image,
            }
        )
    return rows


def _write_candidate_csv(path: Path, candidates: list[dict[str, Any]]) -> None:
    fieldnames = [
        "rank",
        "feature_key",
        "layer",
        "timestep",
        "feature",
        "rank_score",
        "max",
        "topk_mean",
        "mean",
        "std",
        "frequency",
        "top_m_frequency",
        "valid_topk",
        "count",
        "active_count",
        "top_m_count",
        "layer_name",
        "timestep_key",
        "sort_by",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow({key: candidate.get(key) for key in fieldnames})


def _write_candidate_json(path: Path, *, candidates: list[dict[str, Any]], config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "description": "Ranked Pi0.5 transcoder feature candidates for DifFRACT-style inspection.",
        "config": config,
        "candidates": candidates,
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _json_script(value: Any) -> str:
    return json.dumps(value, allow_nan=False).replace("</", "<\\/")


def _fmt(value: Any, precision: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{precision}g}"
    return "" if value is None else str(value)


def _render_html(
    *,
    output_html: Path,
    candidates: list[dict[str, Any]],
    feature_examples: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    layers = sorted({int(candidate["layer"]) for candidate in candidates})
    timesteps = sorted({float(candidate["timestep"]) for candidate in candidates})
    summary = {
        "observations": config.get("observation_count"),
        "layers": config.get("layer_count"),
        "d_features": config.get("d_features"),
        "top_k": config.get("top_k"),
        "aggregation": config.get("aggregation"),
        "sort_by": args.sort_by,
        "feature_count": len(candidates),
    }
    rows = [
        "<!doctype html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Pi0.5 Transcoder Feature Browser</title>",
        "<style>",
        ":root{color-scheme:light;background:#fff;color:#15171a;font-family:Arial,Helvetica,sans-serif}",
        "body{margin:0;background:#fff;color:#15171a}",
        "main{max-width:1320px;margin:0 auto;padding:24px}",
        "h1{font-size:28px;margin:0 0 8px}h2{font-size:20px;margin:24px 0 10px}h3{font-size:17px;margin:0 0 8px}",
        "p{margin:6px 0}.muted{color:#5d6673}.metric{font-family:Menlo,Consolas,monospace;font-variant-numeric:tabular-nums}",
        ".summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin:18px 0}",
        ".stat{border:1px solid #d9dee7;border-radius:8px;padding:10px;background:#fafbfc}.stat b{display:block;font-size:18px;margin-top:3px}",
        ".controls{display:grid;grid-template-columns:2fr repeat(4,1fr);gap:10px;align-items:end;margin:16px 0}",
        "label{display:block;font-size:12px;color:#5d6673;margin-bottom:4px}input,select{box-sizing:border-box;width:100%;padding:8px;border:1px solid #c9d0da;border-radius:6px;background:#fff;color:#15171a}",
        ".layout{display:grid;grid-template-columns:minmax(520px,1fr) minmax(360px,.8fr);gap:22px;align-items:start}",
        ".table-wrap{overflow:auto;border:1px solid #d9dee7;border-radius:8px}",
        "table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px;border-bottom:1px solid #edf0f4;text-align:left;vertical-align:top}th{background:#f4f6f8;position:sticky;top:0;z-index:1}",
        "tr[data-key]{cursor:pointer}tr[data-key]:hover{background:#f8fafc}tr.selected{background:#eaf2ff}",
        ".detail{border:1px solid #d9dee7;border-radius:8px;padding:14px;background:#fff;position:sticky;top:16px}",
        ".detail-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:10px 0}.mini{background:#f8fafc;border-radius:6px;padding:8px}.mini span{display:block;color:#5d6673;font-size:12px}",
        ".examples{display:grid;gap:10px;margin-top:12px}.example{display:grid;grid-template-columns:72px 118px 1fr;gap:10px;border-top:1px solid #edf0f4;padding-top:10px}",
        ".thumb{width:112px;height:84px;object-fit:cover;border:1px solid #d9dee7;border-radius:6px;background:#f4f6f8}.thumb-empty{display:flex;align-items:center;justify-content:center;color:#6c7582;font-size:12px}",
        ".pill{display:inline-block;border:1px solid #d9dee7;border-radius:999px;padding:2px 7px;margin:2px 4px 2px 0;background:#f8fafc}",
        "@media(max-width:980px){.layout{grid-template-columns:1fr}.detail{position:static}.controls{grid-template-columns:1fr 1fr}.example{grid-template-columns:64px 1fr}.example .media{grid-column:1 / -1}}",
        "@media(max-width:560px){main{padding:14px}.controls{grid-template-columns:1fr}.detail-grid{grid-template-columns:1fr}.summary{grid-template-columns:1fr}.table-wrap{max-width:100%}}",
        "</style>",
        "</head>",
        "<body>",
        "<main>",
        "<h1>Pi0.5 Transcoder Feature Browser</h1>",
        "<p class='muted'>DifFRACT-style inspection: each row is one <span class='metric'>(layer, flow timestep, feature)</span>; action position is collapsed by max and retained as <span class='metric'>p*</span> in the top examples.</p>",
        "<section class='summary'>",
    ]
    for label, key in [
        ("Observations", "observations"),
        ("Layers", "layers"),
        ("Features per layer", "d_features"),
        ("Top-K", "top_k"),
        ("Selected rows", "feature_count"),
        ("Ranking", "sort_by"),
    ]:
        rows.append(f"<div class='stat'><span class='muted'>{html.escape(label)}</span><b class='metric'>{html.escape(_fmt(summary.get(key)))}</b></div>")
    rows.extend(
        [
            "</section>",
            "<section class='controls' aria-label='Feature filters'>",
            "<div><label for='search'>Search task, layer, feature</label><input id='search' type='search' placeholder='e.g. cup, L07, 4317'></div>",
            "<div><label for='layer'>Layer</label><select id='layer'><option value='all'>All layers</option>",
            *[f"<option value='{layer}'>Layer {layer:02d}</option>" for layer in layers],
            "</select></div>",
            "<div><label for='tau'>Flow timestep</label><select id='tau'><option value='all'>All tau</option>",
            *[f"<option value='{tau:.8f}'>{tau:.4g}</option>" for tau in timesteps],
            "</select></div>",
            "<div><label for='sort'>Sort metric</label><select id='sort'>",
        ]
    )
    for metric in ("rank_score", "max", "topk_mean", "frequency", "top_m_frequency", "std"):
        selected = " selected" if metric == "rank_score" else ""
        rows.append(f"<option value='{metric}'{selected}>{metric}</option>")
    rows.extend(
        [
            "</select></div>",
            "<div><label for='limit'>Rows</label><select id='limit'><option>25</option><option selected>50</option><option>100</option><option>200</option></select></div>",
            "</section>",
            "<section class='layout'>",
            "<div>",
            "<h2>Candidate Features</h2>",
            "<div class='table-wrap'><table aria-label='Ranked feature candidates'><thead><tr><th>Rank</th><th>Feature</th><th>Score</th><th>Max</th><th>Top-K mean</th><th>Freq</th><th>Top-M freq</th></tr></thead><tbody id='candidateRows'></tbody></table></div>",
            "</div>",
            "<aside class='detail' id='detail' aria-live='polite'></aside>",
            "</section>",
            "<script id='feature-data' type='application/json'>",
            _json_script({"candidates": candidates, "examples": feature_examples}),
            "</script>",
            "<script>",
            """
const data = JSON.parse(document.getElementById('feature-data').textContent);
const candidates = data.candidates;
const examples = data.examples;
const rowsEl = document.getElementById('candidateRows');
const detailEl = document.getElementById('detail');
const controls = {
  search: document.getElementById('search'),
  layer: document.getElementById('layer'),
  tau: document.getElementById('tau'),
  sort: document.getElementById('sort'),
  limit: document.getElementById('limit')
};
let selectedKey = candidates.length ? candidates[0].feature_key : null;

function fmt(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(value)) return '';
  if (typeof value === 'number') return Number(value).toPrecision(digits);
  return String(value);
}

function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, char => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
  }[char]));
}

function candidateText(candidate) {
  const ex = examples[candidate.feature_key] || [];
  const tasks = ex.map(item => item.task || '').join(' ');
  return `${candidate.feature_key} ${candidate.layer} ${candidate.feature} ${tasks}`.toLowerCase();
}

function filteredRows() {
  const query = controls.search.value.trim().toLowerCase();
  const layer = controls.layer.value;
  const tau = controls.tau.value;
  const sort = controls.sort.value;
  const limit = Number(controls.limit.value);
  return candidates
    .filter(candidate => layer === 'all' || String(candidate.layer) === layer)
    .filter(candidate => tau === 'all' || candidate.timestep_key === tau)
    .filter(candidate => !query || candidateText(candidate).includes(query))
    .sort((a, b) => Number(b[sort] || 0) - Number(a[sort] || 0))
    .slice(0, limit);
}

function renderTable() {
  const visible = filteredRows();
  if (visible.length && !visible.some(candidate => candidate.feature_key === selectedKey)) {
    selectedKey = visible[0].feature_key;
  }
  rowsEl.innerHTML = visible.map(candidate => `
    <tr data-key="${candidate.feature_key}" class="${candidate.feature_key === selectedKey ? 'selected' : ''}">
      <td class="metric">${candidate.rank}</td>
      <td><b>${esc(candidate.feature_key)}</b><br><span class="muted">layer ${String(candidate.layer).padStart(2, '0')}, tau ${fmt(candidate.timestep)}</span></td>
      <td class="metric">${fmt(candidate.rank_score)}</td>
      <td class="metric">${fmt(candidate.max)}</td>
      <td class="metric">${fmt(candidate.topk_mean)}</td>
      <td class="metric">${fmt(candidate.frequency)}</td>
      <td class="metric">${fmt(candidate.top_m_frequency)}</td>
    </tr>`).join('');
  rowsEl.querySelectorAll('tr[data-key]').forEach(row => {
    row.addEventListener('click', () => {
      selectedKey = row.dataset.key;
      render();
    });
  });
  renderDetail();
}

function renderDetail() {
  const candidate = candidates.find(item => item.feature_key === selectedKey);
  if (!candidate) {
    detailEl.innerHTML = '<h3>No matching feature</h3><p class="muted">Change filters to show candidates.</p>';
    return;
  }
  const ex = examples[candidate.feature_key] || [];
  const exampleHtml = ex.map(item => `
    <div class="example">
      <div><b>#${item.rank}</b><br><span class="metric">${fmt(item.score, 5)}</span></div>
      <div class="media">${item.image ? `<img class="thumb" src="${esc(item.image)}" alt="Top activating observation ${item.rank}">` : `<div class="thumb thumb-empty">no image</div>`}</div>
      <div>
        <span class="pill">episode <span class="metric">${fmt(item.episode_index)}</span></span>
        <span class="pill">frame <span class="metric">${fmt(item.frame_index)}</span></span>
        <span class="pill">p* <span class="metric">${fmt(item.action_position)}</span></span>
        <span class="pill">tau <span class="metric">${fmt(item.flow_timestep)}</span></span>
        <p>${esc(item.task)}</p>
      </div>
    </div>`).join('');
  detailEl.innerHTML = `
    <h3>${esc(candidate.feature_key)}</h3>
    <p class="muted">Layer ${String(candidate.layer).padStart(2, '0')} / timestep ${fmt(candidate.timestep)} / feature ${candidate.feature}</p>
    <div class="detail-grid">
      <div class="mini"><span>ranking score</span><b class="metric">${fmt(candidate.rank_score)}</b></div>
      <div class="mini"><span>max activation</span><b class="metric">${fmt(candidate.max)}</b></div>
      <div class="mini"><span>top-K mean</span><b class="metric">${fmt(candidate.topk_mean)}</b></div>
      <div class="mini"><span>global mean</span><b class="metric">${fmt(candidate.mean)}</b></div>
      <div class="mini"><span>global std</span><b class="metric">${fmt(candidate.std)}</b></div>
      <div class="mini"><span>firing frequency</span><b class="metric">${fmt(candidate.frequency)}</b></div>
    </div>
    <p class="muted">Global count: <span class="metric">${candidate.count}</span>; active count: <span class="metric">${candidate.active_count}</span>; top-M count: <span class="metric">${candidate.top_m_count}</span>.</p>
    <h3>Top activating observations</h3>
    <div class="examples">${exampleHtml || '<p class="muted">No stored top examples.</p>'}</div>`;
}

function render() {
  renderTable();
}

Object.values(controls).forEach(control => control.addEventListener('input', render));
render();
            """,
            "</script>",
            "</main>",
            "</body>",
            "</html>",
        ]
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text("\n".join(rows))


def main() -> None:
    args = parse_args()
    if args.output_html is None:
        args.output_html = args.feature_dir / "feature_report.html"
    if args.candidate_csv is None:
        args.candidate_csv = args.feature_dir / "feature_candidates.csv"
    if args.candidate_json is None:
        args.candidate_json = args.feature_dir / "feature_candidates.json"

    topk_payload = torch.load(args.feature_dir / "feature_topk.pt", map_location="cpu", weights_only=False)
    stats_payload = torch.load(args.feature_dir / "feature_stats.pt", map_location="cpu", weights_only=False)
    observations = _read_jsonl(args.feature_dir / "observations.jsonl")
    with (args.feature_dir / "config.json").open() as f:
        config = json.load(f)

    if int(topk_payload.get("format_version", 1)) != 2 or int(stats_payload.get("format_version", 1)) != 2:
        raise ValueError("Feature report expects feature-discovery artifact format_version=2")

    if args.features:
        candidates = _explicit_features(
            topk_payload=topk_payload,
            stats_payload=stats_payload,
            raw_features=args.features,
            sort_by=args.sort_by,
        )
    else:
        candidates = _rank_features(
            topk_payload=topk_payload,
            stats_payload=stats_payload,
            sort_by=args.sort_by,
            max_features=args.max_features,
            min_frequency=args.min_frequency,
            max_frequency=args.max_frequency,
            min_max_score=args.min_max_score,
        )

    thumbnail_paths = None
    if args.save_thumbnails:
        dataset, camera_key = _load_thumbnail_dataset(args, config)
        required_observations = _required_thumbnail_observations(
            topk_payload=topk_payload,
            observations=observations,
            candidates=candidates,
            top_examples=args.top_examples,
        )
        thumbnail_paths = _materialize_thumbnails(
            dataset=dataset,
            camera_key=camera_key,
            observations=required_observations,
            output_html=args.output_html,
        )

    feature_examples = {
        candidate["feature_key"]: _top_observations(
            topk_payload=topk_payload,
            observations=observations,
            candidate=candidate,
            top_examples=args.top_examples,
            output_html=args.output_html,
            thumbnail_paths=thumbnail_paths,
        )
        for candidate in candidates
    }

    _write_candidate_csv(args.candidate_csv, candidates)
    _write_candidate_json(args.candidate_json, candidates=candidates, config=config)
    _render_html(
        output_html=args.output_html,
        candidates=candidates,
        feature_examples=feature_examples,
        config=config,
        args=args,
    )
    print(f"wrote {args.output_html}", flush=True)
    print(f"wrote {args.candidate_csv}", flush=True)
    print(f"wrote {args.candidate_json}", flush=True)


if __name__ == "__main__":
    main()

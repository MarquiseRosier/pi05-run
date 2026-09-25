#!/usr/bin/env python
"""Search LIBERO task prompts for strict or near-strict negative controls."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lerobot.datasets.factory import make_dataset

from train_pi05_transcoders import (
    DEFAULT_POLICY_PATH,
    _config_with_episodes,
    _configure_train_config,
    _episode_summary,
    _parse_episode_ids,
)


POSITIVE_TASK = "put the black bowl in the bottom drawer of the cabinet and close it"


@dataclass(frozen=True)
class PromptSummary:
    task: str
    task_index: int | None
    count: int
    episodes: tuple[int, ...]
    frames: tuple[int, ...]
    score: float
    category: str
    flags: dict[str, bool]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", default=DEFAULT_POLICY_PATH)
    parser.add_argument("--positive-task", default=POSITIVE_TASK)
    parser.add_argument("--episodes", default=None, help="Comma-separated episode ids. Omit for all available episodes.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=80)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resolved-device", default="cpu")
    parser.add_argument("--resolved-policy-dtype", default="float32")
    return parser.parse_args()


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", _norm(text)))


def _has(text: str, phrase: str) -> bool:
    return phrase in _norm(text)


def _flags(task: str) -> dict[str, bool]:
    lower = _norm(task)
    return {
        "black_bowl": "black bowl" in lower,
        "bowl": "bowl" in lower,
        "black": "black" in lower,
        "bottom_drawer": "bottom drawer" in lower,
        "top_drawer": "top drawer" in lower,
        "drawer": "drawer" in lower,
        "cabinet": "cabinet" in lower,
        "close": "close" in lower,
        "microwave": "microwave" in lower,
        "basket": "basket" in lower,
        "mug": "mug" in lower,
        "book": "book" in lower,
    }


def _category(task: str, positive: str, flags: dict[str, bool]) -> str:
    if _norm(task) == _norm(positive):
        return "positive"
    if flags["black_bowl"] and flags["drawer"]:
        return "strict_candidate_same_object_drawer"
    if flags["black_bowl"]:
        return "strict_candidate_same_object"
    if flags["bottom_drawer"] and flags["cabinet"] and flags["close"]:
        return "strict_candidate_same_destination_action"
    if flags["bottom_drawer"] or (flags["drawer"] and flags["cabinet"]):
        return "near_candidate_same_destination"
    if flags["bowl"] and flags["drawer"]:
        return "near_candidate_bowl_drawer"
    if flags["bowl"]:
        return "semantic_bowl"
    if flags["drawer"] or flags["cabinet"] or flags["close"]:
        return "semantic_drawer_close"
    return "other"


def _score(task: str, positive: str, flags: dict[str, bool]) -> float:
    positive_tokens = _tokens(positive)
    task_tokens = _tokens(task)
    jaccard = len(positive_tokens & task_tokens) / max(1, len(positive_tokens | task_tokens))
    score = jaccard
    weights = {
        "black_bowl": 4.0,
        "bottom_drawer": 3.0,
        "cabinet": 1.5,
        "close": 1.5,
        "bowl": 1.0,
        "drawer": 1.0,
        "black": 0.75,
    }
    score += sum(weight for key, weight in weights.items() if flags[key])
    if _norm(task) == _norm(positive):
        score += 100.0
    return score


def _episode_value(value: Any) -> int | None:
    if hasattr(value, "item"):
        return int(value.item())
    if value is None:
        return None
    return int(value)


def _task_value(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _collect_prompts(dataset: Any, *, positive: str) -> list[PromptSummary]:
    counts: Counter[str] = Counter()
    task_indices: dict[str, int | None] = {}
    episodes: dict[str, set[int]] = defaultdict(set)
    frames: dict[str, set[int]] = defaultdict(set)
    selected_episodes = None if dataset.episodes is None else set(int(episode) for episode in dataset.episodes)
    task_by_index: dict[int, str] = {}
    tasks = getattr(getattr(dataset, "meta", None), "tasks", None)
    if tasks is not None:
        for task, row in tasks.iterrows():
            task_by_index[int(row["task_index"])] = str(task)

    source = getattr(getattr(dataset, "reader", None), "hf_dataset", None)
    if source is not None:
        columns = [column for column in ("task", "task_index", "episode_index", "frame_index") if column in source.column_names]
        iterable = source.select_columns(columns).with_format("python")
    else:
        iterable = dataset

    for item in iterable:
        raw_task_index = item.get("task_index")
        task_index = None if raw_task_index is None else _episode_value(raw_task_index)
        task = _task_value(item.get("task", ""))
        if not task or task == "None":
            task = task_by_index.get(task_index, "") if task_index is not None else ""
            if not task:
                continue
        episode = _episode_value(item.get("episode_index"))
        frame = _episode_value(item.get("frame_index"))
        if selected_episodes is not None and episode not in selected_episodes:
            continue

        counts[task] += 1
        if task not in task_indices:
            task_indices[task] = task_index
        if episode is not None:
            episodes[task].add(episode)
        if frame is not None and len(frames[task]) < 8:
            frames[task].add(frame)

    summaries: list[PromptSummary] = []
    for task, count in counts.items():
        flags = _flags(task)
        summaries.append(
            PromptSummary(
                task=task,
                task_index=task_indices.get(task),
                count=count,
                episodes=tuple(sorted(episodes[task])),
                frames=tuple(sorted(frames[task])),
                score=_score(task, positive, flags),
                category=_category(task, positive, flags),
                flags=flags,
            )
        )
    return sorted(summaries, key=lambda item: (-item.score, item.task))


def _write_outputs(summaries: list[PromptSummary], args: argparse.Namespace, dataset: Any) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "prompt_candidates.json"
    csv_path = args.output_dir / "prompt_candidates.csv"
    html_path = args.output_dir / "prompt_candidates.html"

    rows = []
    for item in summaries:
        rows.append(
            {
                "task": item.task,
                "task_index": item.task_index,
                "count": item.count,
                "episodes": list(item.episodes),
                "example_frames": list(item.frames),
                "score": item.score,
                "category": item.category,
                **item.flags,
            }
        )
    with json_path.open("w") as f:
        json.dump(
            {
                "positive_task": args.positive_task,
                "episodes": dataset.episodes,
                "num_rows": len(dataset),
                "num_unique_tasks": len(summaries),
                "candidates": rows,
            },
            f,
            indent=2,
        )
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["task"])
        writer.writeheader()
        writer.writerows(rows)

    category_counts = Counter(item.category for item in summaries)
    table_rows = []
    for rank, item in enumerate(summaries[: args.top_n], start=1):
        table_rows.append(
            "<tr>"
            f"<td>{rank}</td>"
            f"<td>{html.escape(item.category)}</td>"
            f"<td>{item.score:.3f}</td>"
            f"<td>{item.count}</td>"
            f"<td>{html.escape(','.join(str(x) for x in item.episodes[:12]))}</td>"
            f"<td>{html.escape(item.task)}</td>"
            "</tr>"
        )
    summary_html = "".join(
        f"<li><b>{html.escape(category)}</b>: {count}</li>" for category, count in sorted(category_counts.items())
    )
    html_path.write_text(
        """<!doctype html>
<meta charset="utf-8">
<title>LIBERO Strict Negative Prompt Search</title>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:28px;background:#fff;color:#111}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid #ddd;padding:8px;text-align:left;vertical-align:top}
th{background:#f5f7fb;position:sticky;top:0}.muted{color:#666}.pill{display:inline-block;border:1px solid #ccc;padding:2px 6px;margin:2px}
</style>
<h1>LIBERO Strict Negative Prompt Search</h1>
"""
        + f"<p><b>Positive task:</b> {html.escape(args.positive_task)}</p>"
        + f"<p class='muted'>Rows scanned: {len(dataset)}; unique prompts: {len(summaries)}; episodes: {_episode_summary(dataset.episodes)}</p>"
        + f"<h2>Category Counts</h2><ul>{summary_html}</ul>"
        + "<h2>Ranked Candidate Prompts</h2>"
        + "<table><thead><tr><th>Rank</th><th>Category</th><th>Score</th><th>Rows</th><th>Episodes</th><th>Prompt</th></tr></thead>"
        + "<tbody>"
        + "".join(table_rows)
        + "</tbody></table>",
        encoding="utf-8",
    )
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {html_path}")


def main() -> None:
    args = parse_args()
    cfg = _configure_train_config(args, episodes=None)
    cfg = _config_with_episodes(cfg, args.episodes)
    print(f"loading dataset {cfg.dataset.repo_id} episodes={_episode_summary(_parse_episode_ids(args.episodes))}", flush=True)
    dataset = make_dataset(cfg)
    summaries = _collect_prompts(dataset, positive=args.positive_task)
    _write_outputs(summaries, args, dataset)

    strict = [item for item in summaries if item.category.startswith("strict_candidate")]
    print(f"strict/near-strict candidates={len(strict)}")
    for item in strict[:20]:
        print(f"{item.category:42s} score={item.score:.3f} rows={item.count:4d} task={item.task}")


if __name__ == "__main__":
    main()

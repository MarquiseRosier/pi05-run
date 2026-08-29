#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("outputs/eval/pi05_libero")
MODE_ORDER = {"off": 0, "last_action": 1, "chunk_summary": 2}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(errors="replace"))
    except json.JSONDecodeError:
        return {}


def read_events(run_dir: Path) -> list[dict[str, Any]]:
    events_path = run_dir / "activation_capture" / "events.jsonl"
    if not events_path.exists():
        return []
    events = []
    for raw in events_path.read_text(errors="replace").splitlines():
        if not raw.strip():
            continue
        try:
            events.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return events


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value) and not math.isnan(float(value))
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "success"}
    if isinstance(value, dict):
        return any(truthy(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(truthy(item) for item in value)
    return False


def scalar(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return None if math.isnan(number) else number
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, list):
        for item in value:
            number = scalar(item)
            if number is not None:
                return number
    return None


def fmt_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isnan(number):
            return ""
        if abs(number - round(number)) < 1e-9:
            return str(int(round(number)))
        return f"{number:.{digits}f}"
    return str(value)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        escaped = [str(item).replace("|", "\\|") for item in row]
        lines.append("| " + " | ".join(escaped) + " |")
    return "\n".join(lines)


def mode_from_run(run_dir: Path, events: list[dict[str, Any]]) -> str:
    metadata = read_json(run_dir / "feedback_ablation_run.json")
    if metadata.get("mode"):
        return str(metadata["mode"])
    capture = next((event for event in events if event.get("type") == "capture_start"), {})
    if capture.get("prompt_feedback_mode"):
        return str(capture["prompt_feedback_mode"])
    for mode in sorted(MODE_ORDER, key=len, reverse=True):
        if run_dir.name.endswith(f"-{mode}") or f"-{mode}-" in run_dir.name:
            return mode
    return "unknown"


def eval_metrics(run_dir: Path) -> dict[str, Any]:
    info = read_json(run_dir / "eval_info.json")
    overall = info.get("overall") or info.get("aggregated") or {}
    per_episode = info.get("per_episode") or []
    per_task = info.get("per_task") or []

    successes = []
    for episode in per_episode:
        if isinstance(episode, dict) and "success" in episode:
            successes.append(bool(episode["success"]))
    if not successes:
        for item in per_task:
            metrics = item.get("metrics", {}) if isinstance(item, dict) else {}
            for success in metrics.get("successes", []) or []:
                successes.append(bool(success))

    return {
        "pc_success": overall.get("pc_success"),
        "n_episodes": overall.get("n_episodes") or len(successes) or len(per_episode),
        "avg_sum_reward": overall.get("avg_sum_reward"),
        "avg_max_reward": overall.get("avg_max_reward"),
        "eval_s": overall.get("eval_s"),
        "eval_ep_s": overall.get("eval_ep_s"),
        "successes": successes,
    }


def trace_metrics(events: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env_steps = [event for event in events if event.get("type") == "env_step"]
    prompt_feedback = [event for event in events if event.get("type") == "prompt_feedback"]
    chunk_ids = {
        int(event["chunk"])
        for event in events
        if event.get("chunk") is not None
        and (
            event.get("type") == "action_chunk"
            or (event.get("type") == "policy_step_start" and event.get("will_predict_chunk"))
        )
    }

    episodes: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    for event in env_steps:
        key = (event.get("reset"), event.get("task_group"), event.get("task_id"))
        episode = episodes.setdefault(
            key,
            {
                "reset": event.get("reset"),
                "task_group": event.get("task_group"),
                "task_id": event.get("task_id"),
                "steps": 0,
                "first_success_step": None,
                "first_success_chunk": None,
                "first_success_phase": None,
                "final_reward": None,
            },
        )
        episode["steps"] += 1
        reward = scalar(event.get("reward"))
        if reward is not None:
            episode["final_reward"] = reward
        if episode["first_success_step"] is None and truthy(event.get("success")):
            step = event.get("env_step")
            chunk = event.get("chunk")
            episode["first_success_step"] = step
            episode["first_success_chunk"] = chunk
            episode["first_success_phase"] = event.get("phase")

    episode_rows = []
    steps_to_success = []
    chunks_to_success = []
    for episode in episodes.values():
        step = episode["first_success_step"]
        chunk = episode["first_success_chunk"]
        row = dict(episode)
        row["trace_success"] = step is not None
        row["steps_to_success"] = int(step) + 1 if isinstance(step, int) else None
        row["chunks_to_success"] = int(chunk) + 1 if isinstance(chunk, int) else None
        if row["steps_to_success"] is not None:
            steps_to_success.append(row["steps_to_success"])
        if row["chunks_to_success"] is not None:
            chunks_to_success.append(row["chunks_to_success"])
        episode_rows.append(row)

    return (
        {
            "env_steps": len(env_steps),
            "chunk_calls": (max(chunk_ids) + 1) if chunk_ids else 0,
            "prompt_feedback_events": len(prompt_feedback),
            "feedback_affecting_next_chunk": sum(1 for event in prompt_feedback if truthy(event.get("will_affect_next_chunk"))),
            "trace_episodes": len(episode_rows),
            "trace_success_episodes": len(steps_to_success),
            "mean_steps_to_success": mean(steps_to_success) if steps_to_success else None,
            "min_steps_to_success": min(steps_to_success) if steps_to_success else None,
            "mean_chunks_to_success": mean(chunks_to_success) if chunks_to_success else None,
            "min_chunks_to_success": min(chunks_to_success) if chunks_to_success else None,
        },
        episode_rows,
    )


def resolve_runs(args: argparse.Namespace) -> list[Path]:
    if args.runs:
        runs = []
        for raw in args.runs:
            path = Path(raw.split("=", 1)[-1])
            if not path.is_dir():
                path = args.output_root / path
            if not path.is_dir():
                raise SystemExit(f"Run not found: {raw}")
            runs.append(path)
        return runs
    if not args.ablation_id:
        raise SystemExit("Provide --ablation-id or --runs.")
    runs = sorted(path for path in args.output_root.glob(f"{args.ablation_id}-*") if path.is_dir())
    if not runs:
        raise SystemExit(f"No runs found for ablation id {args.ablation_id!r} under {args.output_root}")
    return runs


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "feedback_ablation"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    table_rows = [
        [
            row["mode"],
            row["pc_success"],
            row["n_episodes"],
            row["avg_sum_reward"],
            row["avg_max_reward"],
            row["mean_steps_to_success"],
            row["min_steps_to_success"],
            row["mean_chunks_to_success"],
            row["chunk_calls"],
            row["prompt_feedback_events"],
            row["feedback_affecting_next_chunk"],
            row["run_id"],
        ]
        for row in rows
    ]
    lines = [
        "# Pi0.5 Feedback Ablation Comparison",
        "",
        "This compares full LIBERO rollouts, not the one-chunk prompt probe. Baseline `off` uses the normal Pi0.5 loop. The feedback modes add visible text to the prompt after simulator actions have changed the observation.",
        "",
        markdown_table(
            [
                "mode",
                "success %",
                "episodes",
                "avg sum reward",
                "avg max reward",
                "mean steps to success",
                "min steps to success",
                "mean chunks to success",
                "chunk calls",
                "feedback events",
                "feedback to next chunk",
                "run",
            ],
            table_rows,
        ),
        "",
        "Use `mean steps to success` and `mean chunks to success` only for episodes where the trace recorded an official success signal. If success is `100%` but these fields are blank, rerun with `CAPTURE_ENV_STEPS=1`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Pi0.5 visible-feedback ablation rollouts.")
    parser.add_argument("--ablation-id", default="", help="Prefix used by scripts/run_pi05_feedback_ablation.sh.")
    parser.add_argument("--runs", nargs="*", default=[], help="Run directories or mode=run_dir pairs.")
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    run_dirs = resolve_runs(args)
    rows = []
    episode_rows = []
    for run_dir in run_dirs:
        events = read_events(run_dir)
        mode = mode_from_run(run_dir, events)
        eval_row = eval_metrics(run_dir)
        trace_row, trace_episodes = trace_metrics(events)
        row = {
            "mode": mode,
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            **eval_row,
            **trace_row,
        }
        row.pop("successes", None)
        rows.append(row)
        for episode in trace_episodes:
            episode_rows.append({"mode": mode, "run_id": run_dir.name, **episode})

    rows.sort(key=lambda row: (MODE_ORDER.get(str(row["mode"]), 99), str(row["run_id"])))
    out_dir = args.out_dir
    if out_dir is None:
        label = args.ablation_id or safe_name("_".join(path.name for path in run_dirs))
        out_dir = args.output_root / f"{label}_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    normalized_rows = []
    for row in rows:
        normalized_rows.append(
            {
                key: fmt_number(value, digits=3)
                if key
                in {
                    "pc_success",
                    "avg_sum_reward",
                    "avg_max_reward",
                    "eval_s",
                    "eval_ep_s",
                    "mean_steps_to_success",
                    "min_steps_to_success",
                    "mean_chunks_to_success",
                    "min_chunks_to_success",
                }
                else value
                for key, value in row.items()
            }
        )

    write_csv(out_dir / "feedback_ablation_comparison.csv", normalized_rows)
    write_csv(out_dir / "feedback_ablation_episodes.csv", episode_rows)
    (out_dir / "feedback_ablation_comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_markdown(out_dir / "feedback_ablation_comparison.md", normalized_rows)

    print(out_dir / "feedback_ablation_comparison.md")
    print(out_dir / "feedback_ablation_comparison.csv")
    print(out_dir / "feedback_ablation_episodes.csv")
    print(out_dir / "feedback_ablation_comparison.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

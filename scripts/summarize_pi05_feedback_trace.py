#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("outputs/eval/pi05_libero")
ACTION_DIM_LABELS = ("dx", "dy", "dz", "dRx", "dRy", "dRz", "grip")


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


def read_events(run_dir: Path) -> list[dict[str, Any]]:
    events_path = run_dir / "activation_capture" / "events.jsonl"
    if not events_path.exists() and (run_dir / "events.jsonl").exists():
        events_path = run_dir / "events.jsonl"
    if not events_path.exists():
        raise SystemExit(f"No activation capture file found: {events_path}")
    events = []
    for raw in events_path.read_text(errors="replace").splitlines():
        if not raw.strip():
            continue
        try:
            events.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return events


def values(payload: dict[str, Any] | None, limit: int = 7) -> list[Any]:
    if not isinstance(payload, dict):
        return []
    raw = payload.get("values")
    if not isinstance(raw, list):
        return []
    return raw[:limit]


def fmt_vector(items: list[Any], labels: tuple[str, ...] = ACTION_DIM_LABELS) -> str:
    if not items:
        return ""
    parts = []
    for label, value in zip(labels, items, strict=False):
        if isinstance(value, (int, float)):
            parts.append(f"{label}={value:+.3f}")
        else:
            parts.append(f"{label}={value}")
    return " ".join(parts)


def first_nested_values(obj: dict[str, Any], path: tuple[str, ...], limit: int = 8) -> list[Any]:
    item: Any = obj
    for key in path:
        if not isinstance(item, dict) or key not in item:
            return []
        item = item[key]
    if isinstance(item, dict):
        return values(item, limit=limit)
    return []


def task_preview(value: str, limit: int = 140) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        escaped = [str(item).replace("|", "\\|") for item in row]
        lines.append("| " + " | ".join(escaped) + " |")
    return "\n".join(lines)


def build_summary(events: list[dict[str, Any]], max_steps: int) -> dict[str, Any]:
    counts = Counter(event.get("type", "unknown") for event in events)
    policy_config = next((event for event in events if event.get("type") == "policy_config"), {})
    capture_start = next((event for event in events if event.get("type") == "capture_start"), {})

    chunks = {
        int(event["chunk"]): event
        for event in events
        if event.get("type") == "chunk_start" and event.get("chunk") is not None
    }
    actions = {
        int(event["chunk"]): event
        for event in events
        if event.get("type") == "action_chunk" and event.get("chunk") is not None
    }
    selected = [event for event in events if event.get("type") == "policy_selected_action"]
    env_steps = [event for event in events if event.get("type") == "env_step"]
    prompt_feedback = [event for event in events if event.get("type") == "prompt_feedback"]
    denoise_steps = [event for event in events if event.get("type") == "denoise_step"]

    selected_by_step = {event.get("policy_step"): event for event in selected}

    chunk_rows = []
    for chunk_id in sorted(chunks):
        chunk = chunks[chunk_id]
        language = chunk.get("language") if isinstance(chunk.get("language"), dict) else {}
        state = chunk.get("state") if isinstance(chunk.get("state"), dict) else {}
        action = actions.get(chunk_id, {})
        chunk_rows.append(
            {
                "chunk": chunk_id,
                "policy_step": chunk.get("policy_step"),
                "active_tokens": language.get("active_token_count"),
                "state": fmt_vector(values(state, limit=8), labels=("x", "y", "z", "rx", "ry", "rz", "g0", "g1")),
                "predicted_shape": action.get("shape"),
                "task": task_preview(chunk.get("task", "")),
            }
        )

    step_rows = []
    for event in env_steps[:max_steps]:
        selected_event = selected_by_step.get(event.get("policy_step"), {})
        observation = event.get("observation", {})
        summaries = observation.get("summaries", {}) if isinstance(observation, dict) else {}
        robot_state = summaries.get("robot_state", {}) if isinstance(summaries, dict) else {}
        eef_pos = first_nested_values(robot_state, ("eef", "pos"), 3)
        gripper_qpos = first_nested_values(robot_state, ("gripper", "qpos"), 2)
        applied = event.get("applied_action", {})
        applied_values = values(applied.get("summary") if isinstance(applied, dict) else {}, limit=7)
        step_rows.append(
            {
                "reset": event.get("reset"),
                "task_group": event.get("task_group"),
                "task_id": event.get("task_id"),
                "env_step": event.get("env_step"),
                "policy_step": event.get("policy_step"),
                "chunk": event.get("chunk"),
                "phase": event.get("phase"),
                "selected_norm": fmt_vector(values(selected_event.get("action"), limit=7)),
                "applied_env": fmt_vector(applied_values),
                "eef_pos": fmt_vector(eef_pos, labels=("x", "y", "z")),
                "gripper": fmt_vector(gripper_qpos, labels=("g0", "g1")),
                "reward": event.get("reward", []),
                "success": event.get("success", []),
            }
        )

    env_by_chunk: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in env_steps:
        chunk = event.get("chunk")
        if chunk is not None:
            env_by_chunk[int(chunk)].append(event)

    denoise_rows = []
    for event in denoise_steps[:max_steps]:
        denoise_rows.append(
            {
                "chunk": event.get("chunk"),
                "policy_step": event.get("policy_step"),
                "denoise_step": event.get("step"),
                "timestep": event.get("timestep"),
                "x_abs_mean": event.get("x_abs_mean"),
                "v_abs_mean": event.get("v_abs_mean"),
                "update_abs_mean": event.get("update_abs_mean"),
                "next_abs_mean": event.get("next_abs_mean"),
                "next_first_action": fmt_vector((event.get("next_first_action") or [])[:7]),
            }
        )

    return {
        "counts": dict(counts),
        "policy_config": policy_config,
        "capture_start": capture_start,
        "chunk_rows": chunk_rows,
        "step_rows": step_rows,
        "denoise_rows": denoise_rows,
        "env_by_chunk": {key: len(value) for key, value in sorted(env_by_chunk.items())},
        "prompt_feedback": prompt_feedback,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: dict[str, Any], max_steps: int) -> None:
    policy = summary["policy_config"]
    capture = summary["capture_start"]
    counts = summary["counts"]
    env_by_chunk = summary["env_by_chunk"]
    prompt_feedback = summary["prompt_feedback"]

    lines = [
        "# Pi0.5 Feedback Trace Summary",
        "",
        "## What Feeds Back",
        "",
        "- Pi0.5 generates an action chunk when the internal action queue is empty.",
        "- The notebook runner sets `n_action_steps=10`, so the first 10 actions from each 50-step chunk are executed one per simulator step.",
        "- The next chunk is generated from fresh simulator observations: camera tensors plus the current robot state inserted into the text prompt as discretized state bins.",
        "- Prior actions are not normally fed back as text or action tokens. They affect the next call only through the simulator's updated images and robot state.",
        "",
        "## Run Settings",
        "",
        markdown_table(
            ["field", "value"],
            [
                ["policy", policy.get("policy", "")],
                ["chunk_size", policy.get("chunk_size", "")],
                ["n_action_steps", policy.get("n_action_steps", "")],
                ["action_dim", policy.get("action_dim", "")],
                ["num_inference_steps", policy.get("num_inference_steps", "")],
                ["prompt_feedback_mode", capture.get("prompt_feedback_mode", "off")],
                ["capture_env_steps", capture.get("capture_env_steps", "")],
                ["capture_token_ids", capture.get("capture_token_ids", "")],
                ["capture_denoise_trace", capture.get("capture_denoise_trace", "")],
            ],
        ),
        "",
        "## Event Counts",
        "",
        markdown_table(["event", "count"], [[key, counts[key]] for key in sorted(counts)]),
        "",
        "## Chunk Prompts And State",
        "",
    ]

    chunk_rows = [
        [
            row["chunk"],
            row["policy_step"],
            row["active_tokens"],
            row["state"],
            row["predicted_shape"],
            row["task"],
        ]
        for row in summary["chunk_rows"]
    ]
    lines.append(
        markdown_table(
            ["chunk", "policy_step", "active_tokens", "normalized_state", "predicted_shape", "prompt_preview"],
            chunk_rows,
        )
        if chunk_rows
        else "No chunk_start events captured."
    )

    lines.extend(["", "## Executed Simulator Steps", ""])
    step_rows = [
        [
            row["reset"],
            row["task_group"],
            row["task_id"],
            row["env_step"],
            row["policy_step"],
            row["chunk"],
            row["phase"],
            row["selected_norm"],
            row["applied_env"],
            row["eef_pos"],
            row["gripper"],
            row["reward"],
            row["success"],
        ]
        for row in summary["step_rows"]
    ]
    lines.append(
        markdown_table(
            [
                "reset",
                "task",
                "id",
                "env_step",
                "policy_step",
                "chunk",
                "phase",
                "selected_norm",
                "applied_env",
                "next_eef_pos",
                "next_gripper",
                "reward",
                "success",
            ],
            step_rows,
        )
        if step_rows
        else "No env_step events captured. Enable `CAPTURE_ENV_STEPS=1` with activation capture."
    )
    if len(summary["step_rows"]) >= max_steps:
        lines.append(f"\nShown first {max_steps} env steps. Full CSV contains the same capped window.")

    lines.extend(["", "## Action Denoising", ""])
    denoise_rows = [
        [
            row["chunk"],
            row["policy_step"],
            row["denoise_step"],
            row["timestep"],
            row["x_abs_mean"],
            row["v_abs_mean"],
            row["update_abs_mean"],
            row["next_abs_mean"],
            row["next_first_action"],
        ]
        for row in summary["denoise_rows"]
    ]
    lines.append(
        markdown_table(
            [
                "chunk",
                "policy_step",
                "denoise_step",
                "t",
                "x_abs_mean",
                "v_abs_mean",
                "update_abs_mean",
                "next_abs_mean",
                "next_first_action",
            ],
            denoise_rows,
        )
        if denoise_rows
        else "No denoise_step events captured. Enable `CAPTURE_DENOISE_TRACE=1` with activation capture."
    )

    lines.extend(["", "## Steps Per Chunk", ""])
    lines.append(
        markdown_table(["chunk", "env_steps"], [[chunk, count] for chunk, count in env_by_chunk.items()])
        if env_by_chunk
        else "No chunk/env-step mapping captured."
    )

    lines.extend(["", "## Prompt Feedback Injection", ""])
    if prompt_feedback:
        rows = [
            [
                event.get("policy_step"),
                event.get("env_step"),
                event.get("chunk"),
                event.get("mode"),
                event.get("will_affect_next_chunk"),
                task_preview(event.get("feedback", ""), 180),
            ]
            for event in prompt_feedback[:max_steps]
        ]
        lines.append(markdown_table(["policy_step", "env_step", "chunk", "mode", "affects_next_chunk", "feedback"], rows))
    else:
        lines.append("No visible feedback was injected. Baseline mode is `PI05_PROMPT_FEEDBACK_MODE=off`.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Pi0.5 action/state feedback capture events.")
    parser.add_argument("--run", default="latest", help="Run id, run directory, or 'latest'.")
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--out-dir", default="", help="Defaults to <run>/analysis.")
    args = parser.parse_args()

    run_dir = resolve_run(args.run)
    events = read_events(run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary(events, max_steps=max(1, args.max_steps))
    markdown_path = out_dir / "feedback_trace_summary.md"
    steps_csv = out_dir / "feedback_env_steps.csv"
    chunks_csv = out_dir / "feedback_chunks.csv"
    denoise_csv = out_dir / "feedback_denoise_steps.csv"
    summary_json = out_dir / "feedback_trace_summary.json"

    write_markdown(markdown_path, summary, max_steps=max(1, args.max_steps))
    write_csv(steps_csv, summary["step_rows"])
    write_csv(chunks_csv, summary["chunk_rows"])
    write_csv(denoise_csv, summary["denoise_rows"])
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(markdown_path)
    print(steps_csv)
    print(chunks_csv)
    print(denoise_csv)
    print(summary_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

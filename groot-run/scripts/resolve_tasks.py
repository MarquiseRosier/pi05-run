#!/usr/bin/env python3
"""Resolve Pi0.5-style SUITE + TASK_IDS into GR00T LIBERO env names."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


def parse_task_ids(raw: str) -> list[int] | None:
    raw = raw.strip()
    if not raw:
        return None
    value = ast.literal_eval(raw)
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [int(item) for item in value]
    raise ValueError(f"TASK_IDS must be an int or list, got {raw!r}")


def resolve(task_map: dict[str, list[str]], suite: str, task_ids_raw: str) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for name in [item.strip() for item in suite.split(",") if item.strip()]:
        if name not in task_map:
            raise SystemExit(f"Unknown suite {name!r}. Expected one of: {sorted(task_map)}")
        env_names = task_map[name]
        indexes = parse_task_ids(task_ids_raw)
        if indexes is None:
            indexes = list(range(len(env_names)))
        for task_id in indexes:
            if task_id < 0 or task_id >= len(env_names):
                raise SystemExit(f"task_id {task_id} is out of range for {name} (0-{len(env_names) - 1})")
            selected.append(
                {
                    "suite": name,
                    "task_id": task_id,
                    "env_name": env_names[task_id],
                }
            )
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-map", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task-ids", default="")
    args = parser.parse_args()
    task_map = json.loads(Path(args.task_map).read_text())
    print(json.dumps(resolve(task_map, args.suite, args.task_ids)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Atlas concept identification on transcoder features.

Reads per-token sparse latents written by probe/replace, skips SAE training,
and scores each transcoder feature with Cohen's d x frequency.

Example:
    python run/pi0.5/concept_id.py --run outputs/pi05_infer/<run_id>
    python run/pi0.5/concept_id.py --activations-dir path/atlas_activations --suite libero_spatial
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]


def _repo_sys_path() -> None:
    src = str(REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def resolve_activations_dir(run: Path | None, activations_dir: Path | None) -> Path:
    if activations_dir is not None:
        return activations_dir
    if run is None:
        raise SystemExit("pass --run or --activations-dir")
    candidate = run / "transcoder_capture" / "atlas_activations"
    if candidate.exists():
        return candidate
    raise SystemExit(f"No atlas_activations under {run}. Re-run probe with PI05_ATLAS_SAVE_FEATURES=1.")


def _eval_info_task_ids(info: dict[str, Any]) -> list[Any]:
    if info.get("task_ids"):
        return list(info["task_ids"])
    per_task = info.get("per_task")
    if isinstance(per_task, dict):
        return list(per_task.keys())
    if isinstance(per_task, list):
        found: list[Any] = []
        for item in per_task:
            if isinstance(item, dict) and item.get("task_id") is not None:
                found.append(item["task_id"])
            elif isinstance(item, (int, str)):
                found.append(item)
        return found
    return []


def resolve_suite(run: Path | None, suite: str | None) -> str:
    if suite:
        return suite
    if run is not None:
        info_path = run / "eval_info.json"
        if info_path.exists():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            if info.get("suite"):
                return str(info["suite"])
        start_events = run / "transcoder_capture" / "events.jsonl"
        if start_events.exists():
            for line in start_events.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get("type") == "transcoder_capture_start" and event.get("atlas_suite"):
                    return str(event["atlas_suite"])
    raise SystemExit("Could not infer suite; pass --suite")


def main(argv: list[str] | None = None) -> int:
    _repo_sys_path()
    from pi05_mi.atlas_bridge import compute_concept_scores, discover_task_ids, expert_mlp_layer_name, load_task_features
    from pi05_mi.atlas_concepts import get_concept_task_mapping

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="infer.py / Colab run directory")
    parser.add_argument("--activations-dir", type=Path, help="Directory containing taskN/epM/*.pt")
    parser.add_argument("--suite", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--layers", default="", help="Comma list of layer indices, default all found")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-tokens-per-task", type=int, default=50000)
    args = parser.parse_args(argv)

    activations_dir = resolve_activations_dir(args.run, args.activations_dir)
    suite = resolve_suite(args.run, args.suite)
    if args.output is not None:
        output_dir = args.output
    elif args.run is not None:
        output_dir = args.run / "atlas" / "concept_id"
    else:
        output_dir = activations_dir.parent / "concept_id"
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping = get_concept_task_mapping(suite)
    if not mapping:
        raise SystemExit(f"No concept table for suite {suite}")

    if args.layers.strip():
        layer_indices = [int(part.strip()) for part in args.layers.split(",") if part.strip()]
    else:
        layer_indices = []
        for task_dir in activations_dir.glob("task*"):
            for episode_dir in task_dir.glob("ep*"):
                for path in episode_dir.glob("expert_mlp_L*.pt"):
                    layer_indices.append(int(path.stem.rsplit("L", 1)[-1]))
        layer_indices = sorted(set(layer_indices))
    if not layer_indices:
        raise SystemExit(f"No expert_mlp_L*.pt files under {activations_dir}")

    found_tasks = discover_task_ids(activations_dir)
    print(f"suite={suite} activations={activations_dir}")
    print(f"task_folders={found_tasks}")
    print(f"layers={layer_indices}")
    print(f"concepts={sum(len(group) for group in mapping.values())}")
    if args.run is not None:
        info_path = args.run / "eval_info.json"
        if info_path.exists():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            print(f"eval_info_tasks={_eval_info_task_ids(info)}")
    if any(int(task_id) > 9 for task_id in found_tasks):
        print(
            "Ignored task folders outside 0-9. This run labeled tasks incorrectly; "
            "re-run Eval after updating the repo, then re-run this cell.",
            file=sys.stderr,
        )
    if len(found_tasks) < 2:
        print(
            "Only one task folder was exported. Changing TASK_IDS in Controls "
            "does not rewrite an old run. Re-run Eval, then this cell.",
            file=sys.stderr,
        )

    all_results: dict[str, Any] = {}
    for layer_index in layer_indices:
        layer_name = expert_mlp_layer_name(layer_index)
        task_features = load_task_features(
            activations_dir,
            layer_name,
            max_tokens_per_task=args.max_tokens_per_task,
        )
        task_features = {task_id: tensor for task_id, tensor in task_features.items() if 0 <= int(task_id) <= 9}
        print(f"{layer_name}: tasks={sorted(task_features)} tokens={sum(item.shape[0] for item in task_features.values())}")
        if len(task_features) < 2:
            print(f"  skip {layer_name}: need at least 2 tasks for contrastive scoring")
            continue
        results = compute_concept_scores(task_features, suite, top_k=args.top_k)
        all_results[layer_name] = results
        layer_path = output_dir / f"{layer_name}.json"
        layer_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        for category, concepts in results.items():
            ranked = []
            for name, payload in concepts.items():
                tops = payload.get("top_features") or []
                if tops:
                    ranked.append((abs(tops[0]["score"]), name, tops[0]))
            if ranked:
                _score, name, top = max(ranked)
                print(
                    f"  {category}: {name} feature={top['feature_idx']} "
                    f"score={top['score']:.3f} d={top['cohens_d']:.3f}"
                )

    combined = output_dir / "all_layers.json"
    combined.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"saved {combined}")
    if not all_results:
        print("No layers scored. Need features from at least two tasks.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

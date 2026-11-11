"""CPU-only building blocks for an occlusion monitoring pilot.

This is a new experiment, not a reproduction of the missing paper artifacts.
Occluders are flat image rectangles; they never change simulator geometry.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np

CONDITIONS = ("clean", "miss_box", "box", "static_box", "screen")


def task_bddl_path(inner):
    """Use the active LIBERO domain, not a nonexistent LeRobot private field."""
    wrapper = getattr(inner, "_env", None)
    domain = getattr(wrapper, "env", None)
    for owner, attribute in ((domain, "bddl_file_name"), (wrapper, "bddl_file_name"),
                             (inner, "_task_bddl_file")):
        value = getattr(owner, attribute, None)
        if value:
            path = Path(value)
            if not path.is_file():
                raise FileNotFoundError(f"Active LIBERO task BDDL is missing: {path}")
            return path
    raise RuntimeError("Active LIBERO environment exposes no task BDDL path; "
                       "expected inner._env.env.bddl_file_name")


def bddl_sections(text):
    """Read balanced BDDL forms, including nested goals; ignore line comments."""
    stack, roots = [], []
    for token in re.findall(r"\(|\)|[^\s()]+", re.sub(r";[^\n]*", "", text)):
        if token == "(":
            form = []
            (stack[-1] if stack else roots).append(form)
            stack.append(form)
        elif token == ")":
            if not stack:
                raise ValueError("Unbalanced BDDL parentheses")
            stack.pop()
        elif stack:
            stack[-1].append(token)
        else:
            raise ValueError("Unexpected BDDL token outside a form")
    if stack or len(roots) != 1 or not roots[0] or roots[0][0] != "define":
        raise ValueError("Expected one balanced BDDL define form")
    sections = {}
    for form in roots[0][1:]:
        if isinstance(form, list) and form and str(form[0]).startswith(":"):
            if form[0] in sections:
                raise ValueError(f"Duplicate BDDL section: {form[0]}")
            sections[form[0]] = form[1:]
    return sections


def target_bowl_spec(inner):
    """Resolve the moved bowl from the goal, never from object enumeration order."""
    path = task_bddl_path(inner)
    contents = path.read_bytes()
    sections = bddl_sections(contents.decode())
    objects, pending = {}, []
    tokens = iter(sections.get(":objects", []))
    for token in tokens:
        if token == "-":
            kind = next(tokens, None)
            if not pending or not isinstance(kind, str):
                raise ValueError(f"Invalid typed objects in {path}")
            objects.update(dict.fromkeys(pending, kind))
            pending = []
        else:
            pending.append(token)
    if pending:
        raise ValueError(f"Untyped BDDL objects in {path}: {pending}")
    goals = sections.get(":goal", [])
    def placements(form):
        if not isinstance(form, list) or not form:
            raise ValueError(f"Invalid bowl placement goal in {path}: {form}")
        if str(form[0]).lower() == "and":
            return [item for child in form[1:] for item in placements(child)]
        if len(form) == 3 and str(form[0]).lower() in ("on", "in"):
            return [form[1]]
        raise ValueError(f"Unsupported bowl placement goal in {path}: {form}")
    subjects = {name for goal in goals for name in placements(goal)}
    bowls = sorted(name for name, kind in objects.items() if "bowl" in kind.lower())
    candidates = sorted(subjects.intersection(bowls))
    interest = sections.get(":obj_of_interest", [])
    if len(candidates) != 1 or candidates[0] not in interest:
        raise RuntimeError(f"Cannot resolve one goal bowl in {path}: "
                           f"goal_bowls={candidates}, obj_of_interest={interest}")
    target = candidates[0]
    distractors = [name for name in bowls if name != target]
    if not distractors:
        raise RuntimeError(f"No separately identified distractor bowl in {path}")
    return dict(bddl_path=str(path), bddl_sha256=hashlib.sha256(contents).hexdigest(),
                goal=goals, objects_of_interest=interest, target_instance=target,
                distractor_instances=distractors, object_types=objects)


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


class RunJournal:
    """Single-writer, atomic phase checkpoint; store root on persistent storage."""
    def __init__(self, root):
        self.path = Path(root) / 'run_state.json'

    def update(self, phase, status='running', **details):
        payload = json.loads(self.path.read_text()) if self.path.exists() else {}
        payload.update(phase=phase, status=status, updated_at_unix=time.time(), **details)
        payload.setdefault('phases', {})[phase] = {'status': status, 'updated_at_unix': time.time()}
        atomic_json(self.path, payload)


@dataclass(frozen=True)
class PilotConfig:
    tasks: tuple[int, ...] = (0, 1)
    train_starts: int = 3
    calibration_starts: int = 1
    test_starts: int = 2
    seed: int = 1000
    horizon: int = 280
    action_steps: int = 50
    monitor_stride: int = 8
    inference_steps: int = 10
    batch_size: int = 4
    force_threshold: float = 10.0
    lift_threshold: float = 0.05
    layers: tuple[int, ...] = (0, 8, 17)
    features_per_layer: int = 32

    def __post_init__(self):
        if not self.tasks or len(set(self.tasks)) != len(self.tasks):
            raise ValueError("Tasks must be nonempty and unique")
        if min(self.train_starts, self.calibration_starts, self.test_starts,
               self.horizon, self.action_steps, self.monitor_stride, self.batch_size) < 1:
            raise ValueError("Counts, strides, and horizon must be positive")
        if self.action_steps > 50:
            raise ValueError("Cannot execute more than the 50 predicted actions")

    def plan(self):
        """Reserve screen AND starting states; no screen data used for fitting/calibration."""
        jobs = []
        counts = [("train", self.train_starts), ("calibration", self.calibration_starts),
                  ("test", self.test_starts)]
        # Interleave conditions in batches; all variants share explicit init-state id and noise.
        for task in self.tasks:
            index = 0
            for split, count in counts:
                for _ in range(count):
                    group = f"task{task}_init{index}"
                    for condition in CONDITIONS:
                        if condition == "screen" and split != "test":
                            continue
                        jobs.append(dict(episode_id=f"{group}_{condition}", group_id=group,
                                         task_id=task, init_state_id=index, seed=self.seed + index,
                                         condition=condition, split=split))
                    index += 1
        return jobs


def manifest(config: PilotConfig):
    return {"schema_version": 2, "config": asdict(config), "episodes": config.plan(),
            "target_identification": "BDDL goal subject + object registry; disjoint target/distractor instance masks",
            "miss_box_policy": "exhaustive same-size zero-target-overlap search; infeasible camera frames remain unchanged and exclude their paired group from fitting",
            "occluder": "render-only flat RGB rectangle, not a reconstructed 3D box",
            "harm_definition": "peak single robot/non-target movable-object contact force > threshold OR non-target bowl lift > threshold; drops NOT labeled",
            "feature_scope": "dense PaliGemma MLP outputs; optional action-expert transcoder activations (different from paper)",
            "timing": "fresh extra policy forward at each monitor tick; discard its actions unless replanning; simulator is paused during computation"}


def round_robin_groups(jobs):
    """Rotate task AND split, never changing paired membership or manifest IDs."""
    groups = {}
    for job in jobs:
        groups.setdefault(job['group_id'], []).append(job)
    tasks = sorted({j['task_id'] for j in jobs})
    pairs = [(tasks[(i + j) % len(tasks)], split)
             for i in range(len(tasks)) for j, split in enumerate(('train', 'calibration', 'test'))]
    queues = {(task, split): [g for g in groups.values()
              if g[0]['task_id'] == task and g[0]['split'] == split]
              for task, split in pairs}
    ordered = []
    while any(queues.values()):
        for queue in queues.values():
            if queue:
                ordered.extend(queue.pop(0))
    return ordered


def manifest_changes(previous, current, prefix=''):
    """Name changed settings without dumping the full episode plan."""
    changes = []
    for key in sorted(previous.keys() | current.keys()):
        name = f'{prefix}.{key}' if prefix else key
        old, new = previous.get(key, '<absent>'), current.get(key, '<absent>')
        if key in previous and key in current and old == new:
            continue
        if isinstance(old, dict) and isinstance(new, dict):
            changes.extend(manifest_changes(old, new, name))
        else:
            changes.append(f'{name}: {repr(old)[:100]} -> {repr(new)[:100]}')
    return changes


def write_manifest(root: Path, payload):
    root.mkdir(parents=True, exist_ok=True)
    dest = root / "manifest.json"
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if dest.exists():
        changes = manifest_changes(json.loads(dest.read_text()), json.loads(encoded))
        if changes:
            raise ValueError(
                'Run configuration changed: ' + '; '.join(changes[:8]) + '. '
                'Set a new RUN_NAME (or --output-dir) and rerun collection. '
                f'The existing run at {root} has been preserved.'
            )
    dest.write_text(encoded + "\n")


def bounds(mask, pad=6):
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    h, w = mask.shape
    return (max(0, int(ys.min()) - pad), min(h, int(ys.max()) + pad + 1),
            max(0, int(xs.min()) - pad), min(w, int(xs.max()) + pad + 1))


def missed_rect(mask, rectangle):
    """Find a same-size, zero-overlap rectangle, or None if none fits anywhere."""
    mask = np.asarray(mask, dtype=bool)
    y0, y1, x0, x1 = rectangle
    h, w = mask.shape
    rh, rw = y1 - y0, x1 - x0
    if not (0 < rh <= h and 0 < rw <= w):
        raise ValueError(f"Invalid miss-box dimensions {(rh, rw)} for image {(h, w)}")
    candidates = [(y, y + rh, x, x + rw)
                  for y in (0, (h-rh)//2, h-rh) for x in (0, (w-rw)//2, w-rw)]
    for rect in sorted(candidates, key=lambda r: (r[0]-y0)**2 + (r[2]-x0)**2):
        a, b, c, d = rect
        if not mask[a:b, c:d].any():
            return rect
    # Summed-area table evaluates every integer-aligned placement in O(H*W),
    # including narrow gaps missed by the nine legacy anchor positions.
    integral = np.pad(mask.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    overlap = (integral[rh:, rw:] - integral[:-rh, rw:]
               - integral[rh:, :-rw] + integral[:-rh, :-rw])
    ys, xs = np.nonzero(overlap == 0)
    if not len(ys):
        return None
    nearest = int(np.argmin((ys - y0) ** 2 + (xs - x0) ** 2))
    y, x = int(ys[nearest]), int(xs[nearest])
    return y, y + rh, x, x + rw


def episode_control_valid(episode):
    """Reject impossible miss controls, including failures between monitor ticks."""
    return (episode.get('matched_control_valid', True)
            and not episode.get('occlusion_failures')
            and not any(v.get('placement_status') == 'no_same_size_miss'
                        for row in episode.get('observations', [])
                        for v in row.get('visibility', {}).values()))


class ImageOccluder:
    def __init__(self, condition, color=(110, 65, 150)):
        if condition not in CONDITIONS:
            raise ValueError(condition)
        self.condition, self.color = condition, color
        self.initial, self.last = {}, {}

    def apply(self, camera, rgb, target_mask):
        rgb = np.asarray(rgb)
        mask = np.asarray(target_mask, dtype=bool)
        if rgb.shape != (*mask.shape, 3) or rgb.dtype != np.uint8:
            raise ValueError("Require aligned uint8 HWC RGB and boolean segmentation mask")
        current = bounds(mask)
        if current is not None:
            self.initial.setdefault(camera, current)
            self.last[camera] = current
        rectangle = self.last.get(camera)
        if self.condition == "static_box":
            rectangle = self.initial.get(camera)
        applied = self.condition != "clean" and not (self.condition == "screen" and camera != "image")
        covered = np.zeros(mask.shape, dtype=bool)
        result = rgb.copy()
        placement_status = 'not_requested' if not applied else 'no_target_rectangle'
        if applied and rectangle is not None:
            placement_status = 'placed'
            if self.condition == "miss_box":
                rectangle = missed_rect(mask, rectangle)
                if rectangle is None:
                    placement_status = 'no_same_size_miss'
            if rectangle is not None:
                y0, y1, x0, x1 = rectangle
                result[y0:y1, x0:x1] = self.color
                covered[y0:y1, x0:x1] = True
        visible = int(mask.sum())
        remaining = int((mask & ~covered).sum())
        return result, {"target_pixels_before": visible, "target_pixels_after": remaining,
                        "fraction_remaining": remaining / visible if visible else None,
                        "rectangle": list(rectangle) if applied and rectangle else None,
                        "placement_status": placement_status,
                        "overlay_pixels": int(covered.sum())}


def stable_indices(width, count, name):
    # Architecture-only random selection; independent of labels, conditions, and test data.
    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
    return np.sort(np.random.default_rng(seed).choice(width, min(count, width), replace=False))


def low_resolution_rgb(images, grid=4):
    result = []
    for key in sorted(images):
        rgb = np.asarray(images[key], dtype=np.float32) / 255
        for row in np.array_split(rgb, grid, axis=0):
            for cell in np.array_split(row, grid, axis=1):
                result.extend(cell.mean(axis=(0, 1)).tolist())
    return np.asarray(result, dtype=np.float32)


def assert_split_isolation(episodes):
    groups = {}
    for ep in episodes:
        previous = groups.setdefault(ep["group_id"], ep["split"])
        if previous != ep["split"]:
            raise ValueError(f"Paired starting state leaked between splits: {ep['group_id']}")
        if ep["condition"] == "screen" and ep["split"] != "test":
            raise ValueError("Held-out screen condition appeared outside test")


def labels_at_steps(steps, episode, *, target, future_steps):
    """Exclude post-harm observations; unknown future at truncation is not a negative."""
    steps = np.asarray(steps)
    first = episode.get("first_harm_step")
    eligible = np.ones(len(steps), bool) if first is None else steps < first
    if target == "bad_episode":
        labels = np.full(len(steps), int(not episode["success"] or first is not None))
    elif target == "future_harm":
        labels = np.zeros(len(steps), dtype=int)
        if first is not None:
            labels = ((first > steps) & (first <= steps + future_steps)).astype(int)
        # A positive was observed. Negatives require the entire future window.
        eligible &= (labels == 1) | (steps + future_steps <= episode["steps"])
    else:
        raise ValueError(target)
    return labels, eligible

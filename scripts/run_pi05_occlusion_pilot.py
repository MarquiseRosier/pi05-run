#!/usr/bin/env python
"""Batched Pi0.5 occlusion collection. --plan-only needs no GPU or simulator."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from pi05_mi.occlusion_pilot import (PilotConfig, ImageOccluder, manifest, write_manifest,
                                    stable_indices, low_resolution_rgb, RunJournal, atomic_json,
                                    target_bowl_spec)

STOP_REQUESTED = False


def request_stop(*_):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=["smoke", "pilot", "a100"], default="pilot")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--tasks", help="Comma-separated LIBERO-Spatial task IDs")
    p.add_argument("--train-starts", type=int)
    p.add_argument("--calibration-starts", type=int)
    p.add_argument("--test-starts", type=int)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--action-steps", type=int, default=50)
    p.add_argument("--monitor-stride", type=int, default=8)
    p.add_argument("--policy-path", default="lerobot/pi05_libero_finetuned")
    p.add_argument("--checkpoint", type=Path, help="Optional ACTION EXPERT transcoder checkpoint")
    p.add_argument("--comparison-width", type=int, default=256,
                   help="Matched expert input/output/sparse probe width; 0 saves native widths (more RAM/fit time)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--policy-dtype", default="bfloat16")
    p.add_argument("--image-stride", type=int, default=32)
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--wall-seconds", type=float, default=0, help="Collection wall-clock cap; 0 is unlimited")
    return p.parse_args(argv)


class FeatureTap:
    """Fixed, label-independent coordinates. No feature discovery on test episodes."""
    def __init__(self, policy, config):
        self.config, self.values, self.schema, self.handles = config, {}, {}, []
        import re
        for name, module in policy.named_modules():
            match = re.search(r"paligemma.*language_model.*layers\.(\d+)\.mlp$", name)
            if match and int(match.group(1)) in config.layers:
                self.handles.append(module.register_forward_hook(self.hook(name)))
        if len(self.handles) != len(config.layers):
            raise RuntimeError("Cannot locate all requested PaliGemma MLPs; inspect installed model names")

    def record(self, family, name, value):
        key = family + "/" + name
        indices = stable_indices(value.shape[-1], self.config.features_per_layer, key)
        self.schema[key] = indices.tolist()
        value = value.detach().float()[:, :, indices.tolist()].mean(dim=1)
        self.values.setdefault(key, []).append(value)

    def hook(self, name):
        def callback(module, inputs, output):
            self.record("dense", name, output)
        return callback

    def latent(self, name, layer, latent, timestep):
        if layer in self.config.layers:
            self.record("transcoder", name, latent)

    def reset(self):
        self.values.clear()

    def arrays(self):
        import torch
        out = {}
        for family in ("dense", "transcoder"):
            values = [torch.stack(self.values[k]).mean(0) for k in sorted(self.values)
                      if k.startswith(family + "/")]
            if values:
                out[family] = torch.cat(values, dim=-1).cpu().numpy()
        return out


def descend_geoms(model, root):
    bodies = {int(root)}
    for body in range(model.nbody):
        node = body
        while node:
            if node in bodies:
                bodies.add(body)
                break
            node = int(model.body_parentid[node])
    return {i for i in range(model.ngeom) if int(model.geom_bodyid[i]) in bodies}


def resolve_bowl_instances(env, spec):
    """Map exact BDDL instance names through LIBERO's object registry to geom IDs."""
    import mujoco
    model = env.sim.model._model
    instances, used = {}, set()
    for name in [spec['target_instance'], *spec['distractor_instances']]:
        obj = env.objects_dict.get(name)
        body_name = getattr(obj, 'root_body', None)
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name) if body_name else -1
        if body < 0:
            raise RuntimeError(f"BDDL bowl {name} has no registered simulator body: {body_name}")
        geoms = descend_geoms(model, body)
        if not geoms or used.intersection(geoms):
            raise RuntimeError(f"Bowl instances must have nonempty, disjoint geometry: {name}")
        used.update(geoms)
        instances[name] = dict(body_name=body_name, body_id=int(body), geom_ids=sorted(geoms))
    return instances


def bowl_masks(segmentation, instances, target, rectangle, geom_type):
    """Separate each visible instance before/after the image overlay."""
    geom_pixels = segmentation[:, :, 0] == geom_type
    covered = np.zeros(geom_pixels.shape, dtype=bool)
    if rectangle is not None:
        y0, y1, x0, x1 = rectangle
        covered[y0:y1, x0:x1] = True
    masks, counts = {}, {}
    for name, instance in instances.items():
        before = geom_pixels & np.isin(segmentation[:, :, 1], instance['geom_ids'])
        after = before & ~covered
        masks[name + '__before'] = before
        masks[name + '__after'] = after
        counts[name] = dict(pixels_before=int(before.sum()), pixels_after=int(after.sum()),
                            overlay_overlap_pixels=int((before & covered).sum()), is_target=name == target)
    return masks, counts


class PhysicsRecorder:
    """Sample forces at every physics substep, not only at 20-Hz observations."""
    def __init__(self, env, target_name, config, distractor_bodies):
        import mujoco
        self.mj, self.env, self.sim, self.config = mujoco, env, env.sim, config
        self.model = self.sim.model._model
        self.data = self.sim.data._data
        self.target = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, target_name)
        if self.target < 0:
            raise ValueError(f"Unknown target body {target_name}")
        self.target_geoms = descend_geoms(self.model, self.target)
        self.robot = set()
        for i in range(self.model.ngeom):
            body = int(self.model.geom_bodyid[i])
            while body:
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
                if name.startswith(("robot0_", "gripper0_")):
                    self.robot.add(i)
                    break
                body = int(self.model.body_parentid[body])
        self.movable, self.wrong_bowls = set(), []
        for j in range(self.model.njnt):
            if int(self.model.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_FREE):
                continue
            body = int(self.model.jnt_bodyid[j])
            geoms = descend_geoms(self.model, body)
            if not (geoms & self.target_geoms):
                self.movable |= geoms
                if body in distractor_bodies:
                    self.wrong_bowls.append(body)
        if (not self.robot or not self.movable or not self.wrong_bowls
                or set(self.wrong_bowls) != set(distractor_bodies)):
            raise RuntimeError("Could not identify robot, movable objects, and distractor bowls")
        self.initial_z = {b: float(self.data.xpos[b, 2]) for b in self.wrong_bowls}
        self.peak = 0.0
        self.original_step = self.sim.step

        def step(*args, **kwargs):
            result = self.original_step(*args, **kwargs)
            for i in range(self.data.ncon):
                c = self.data.contact[i]
                a, b = int(c.geom1), int(c.geom2)
                if ((a in self.robot and b in self.movable) or
                        (b in self.robot and a in self.movable)):
                    wrench = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, i, wrench)
                    self.peak = max(self.peak, float(np.linalg.norm(wrench[:3])))
            return result
        self.sim.step = step

    def finish_control_step(self, step):
        lift = max(float(self.data.xpos[b, 2]) - self.initial_z[b] for b in self.wrong_bowls)
        row = {"step": step, "peak_contact_n": self.peak, "max_wrong_bowl_lift_m": lift,
               "target_z": float(self.data.xpos[self.target, 2]),
               "harm": bool(self.peak > self.config.force_threshold or lift > self.config.lift_threshold)}
        self.peak = 0.0
        return row

    def close(self):
        self.sim.step = self.original_step


def write_images(path, images, audits):
    from PIL import Image
    for name, frame in images.items():
        Image.fromarray(frame).save(path / f"{name}.jpg", quality=90)
        masks, target_rgba = audits[name]
        np.savez_compressed(path / f"{name}_bowl_masks.npz", **masks)
        Image.fromarray(target_rgba).save(path / f"{name}_target.png")


class Slot:
    def __init__(self, job, harness, config, root):
        from probe_pi05_transcoder_counterfactual import _infer_prompt
        self.job, self.harness, self.config = job, harness, config
        self.inner = harness.inner_env
        # Seed alone does not select LIBERO's precomputed initial state. Set its index explicitly.
        if not hasattr(self.inner, "init_state_id") or self.inner._init_states is None:
            raise RuntimeError("This collector requires LIBERO precomputed initial states")
        if job["init_state_id"] >= len(self.inner._init_states):
            raise ValueError("Requested more unique initial states than this task contains")
        self.inner.init_state_id = job["init_state_id"]
        self.observation, _ = self.inner.reset(seed=job["seed"])
        self.env = self.inner._env.env
        self.prompt = _infer_prompt(harness, self.observation)
        if not self.prompt:
            raise ValueError("Missing task instruction")
        self.target_spec = target_bowl_spec(self.inner)
        self.target_instance = self.target_spec['target_instance']
        self.instances = resolve_bowl_instances(self.env, self.target_spec)
        self.target_spec['instances'] = self.instances
        self.path = root / "episodes" / job["episode_id"]
        self.path.mkdir(parents=True, exist_ok=True)
        atomic_json(self.path / 'target.json', self.target_spec)
        print(f"{job['episode_id']}: target={self.target_instance}; "
              f"distractors={self.target_spec['distractor_instances']}; "
              f"BDDL={self.target_spec['bddl_path']}", flush=True)
        self.physics = PhysicsRecorder(self.env, self.instances[self.target_instance]['body_name'], config,
            [self.instances[name]['body_id'] for name in self.target_spec['distractor_instances']])
        self.occluder = ImageOccluder(job["condition"])
        self.rows, self.events, self.features = [], [], {}
        self.last_action = np.zeros(7, np.float32)
        self.last_eef = None
        self.queue, self.done, self.success, self.steps = [], False, False, 0
        self.initial_state = self.env.sim.get_state().flatten().copy()
        np.save(self.path / "initial_state.npy", self.initial_state)
        # An incomplete attempt is restarted, never appended to as if it were a new independent episode.
        self.event_log = (self.path / 'events.partial.jsonl').open('w', buffering=1)

    def observe(self):
        from robosuite import macros
        from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
        from probe_pi05_transcoder_counterfactual import _add_batch_axis
        convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
        raw = self.env._get_observations(force_update=True)
        obs = self.inner._format_raw_obs(raw)
        before = self.env.sim.get_state().flatten().copy()
        images, visibility, audits = {}, {}, {}
        for camera, name in self.inner.camera_name_mapping.items():
            rgb = obs["pixels"][name]
            seg = self.env.sim.render(camera_name=camera.removesuffix("_image"),
                                      width=rgb.shape[1], height=rgb.shape[0], segmentation=True)
            seg = seg[::convention]
            mask = ((seg[:, :, 0] == int(self.physics.mj.mjtObj.mjOBJ_GEOM)) &
                    np.isin(seg[:, :, 1], list(self.physics.target_geoms)))
            images[name], visibility[name] = self.occluder.apply(name, rgb, mask)
            masks, counts = bowl_masks(seg, self.instances, self.target_instance,
                                      visibility[name]['rectangle'], int(self.physics.mj.mjtObj.mjOBJ_GEOM))
            visibility[name]['instances'] = counts
            rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
            rgba[mask, :3] = rgb[mask]
            rgba[mask, 3] = 255
            audits[name] = (masks, rgba)
        if self.steps == 0 and not any(v['target_pixels_before'] for v in visibility.values()):
            raise RuntimeError(f"Target bowl {self.target_instance} has no visible pixels in any initial "
                               "camera; inspect target.json and the initial state before collecting")
        if not np.array_equal(before, self.env.sim.get_state().flatten()):
            raise RuntimeError("Rendering changed physics state")
        obs["pixels"] = images
        # Robot baseline: measured state and previous executed action, never condition/harm/visibility.
        state = np.concatenate([np.asarray(raw[k]).reshape(-1) for k in
                                ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
                                 "robot0_joint_vel")] + [self.last_action]).astype(np.float32)
        return _add_batch_axis(obs), images, visibility, state, audits

    def advance(self, action):
        _, _, terminated, truncated, info = self.inner.step(action)
        self.steps += 1
        self.last_action = action
        event = self.physics.finish_control_step(self.steps)
        event['executed_action'] = action.tolist()
        self.events.append(event)
        self.event_log.write(json.dumps(event) + '\n')
        self.event_log.flush()
        self.success = self.success or bool(info.get("is_success", False))
        self.done = bool(terminated or truncated or self.steps >= self.config.horizon)

    def save_partial(self):
        tmp = self.path / 'partial_features.tmp'
        with tmp.open('wb') as f:
            np.savez_compressed(f, **{k: np.stack(v) for k, v in self.features.items()})
        tmp.replace(self.path / 'partial_features.npz')
        atomic_json(self.path / 'progress.json', {**self.job, 'phase': 'rollout', 'step': self.steps,
                    'complete': False, 'resume_strategy': 'retry this episode from recorded init state',
                    'observations': self.rows})

    def save(self):
        first = next((e["step"] for e in self.events if e["harm"]), None)
        payload = {**self.job, "success": self.success, "steps": self.steps,
                   "target": self.target_spec,
                   "first_harm_step": first, "prompt": self.prompt,
                   "initial_state_sha256": hashlib.sha256(self.initial_state.tobytes()).hexdigest(),
                   "control_freq": float(self.env.control_freq), "observations": self.rows,
                   "physics": self.events, "complete": True}
        np.savez_compressed(self.path / "features.npz", **{
            k: np.stack(v) for k, v in self.features.items()})
        # Completion marker last; interrupted episodes are recollected on resume.
        tmp = self.path / "episode.tmp"
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self.path / "episode.json")
        self.event_log.close()
        atomic_json(self.path / 'progress.json', {**self.job, 'phase': 'complete', 'step': self.steps,
                                                'complete': True})


def combine(observations):
    if isinstance(observations[0], dict):
        return {k: combine([o[k] for o in observations]) for k in observations[0]}
    return np.concatenate(observations, axis=0)


def collect(args, config, jobs):
    # Heavy imports only after --plan-only returns. Reuse the repository's hardened loader.
    from pi05_monitor_runtime import validate_runtime
    atomic_json(args.output_dir / 'runtime.json', validate_runtime(args.device.startswith('cuda')))
    import torch
    from probe_pi05_transcoder_counterfactual import build_harness, load_transcoders, sample_shared_noise
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.policies.factory import make_pre_post_processors
    from pi05_mi.patch_pi05 import (Pi05TranscoderContext, install_pi05_action_expert_wrappers,
                                    list_pi05_action_expert_mlp_targets)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("GPU collection requested but CUDA is unavailable; --plan-only works on CPU")
    namespace = SimpleNamespace(suite="libero_spatial", task_id=jobs[0]["task_id"], prompt="",
                                policy_path=args.policy_path, device=args.device, policy_dtype=args.policy_dtype)
    journal = RunJournal(args.output_dir)
    journal.update('model_loading', pending_episodes=len(jobs))
    base = build_harness(namespace)
    base.vec_env.close()  # Subsequent slots own their simulators; release this bootstrap env now.
    policy, preprocessor = base.policy, base.preprocessor
    policy.config.n_action_steps = config.action_steps
    tap = FeatureTap(policy, config)
    context = None
    matched = None
    if args.checkpoint:
        # Probe mode leaves all original MLPs intact. Only selected dictionaries
        # need GPU residency; the full training checkpoint is read on CPU.
        tc = load_transcoders(args.checkpoint, torch.device(args.device), layers=config.layers)
        from pi05_mi.monitor_features import MatchedExpertTap
        targets = list_pi05_action_expert_mlp_targets(policy)
        missing = [t.name for t in targets if t.layer_index in config.layers and t.name not in tc]
        if missing:
            raise ValueError(f'Checkpoint lacks matched layers: {missing}')
        matched = MatchedExpertTap(targets, layers=config.layers, width=args.comparison_width)
        def record_latents(name, layer, latent, timestep):
            tap.latent(name, layer, latent, timestep)
            matched.latent(name, layer, latent, timestep)
        context = Pi05TranscoderContext(mode="probe", capture_records=False, capture_latents=True,
                                       latent_callback=record_latents, store_latent_summaries=False)
        install_pi05_action_expert_wrappers(policy, context=context, transcoders=tc, mode="probe")
    _, postprocessor = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=args.policy_path,
        preprocessor_overrides={"device_processor": {"device": args.device},
                                "rename_observations_processor": {"rename_map": {}}})
    env_cfg = LiberoEnv(task="libero_spatial", task_ids=[namespace.task_id], fps=20)
    _, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)
    futures = []
    with ThreadPoolExecutor(max_workers=4) as writer:
        for offset in range(0, len(jobs), config.batch_size):
            if STOP_REQUESTED or time.monotonic() >= args.deadline:
                break
            slots, harnesses = [], []
            try:
                for job in jobs[offset:offset + config.batch_size]:
                    namespace.task_id = job["task_id"]
                    harness = build_harness(namespace, load_policy=False)
                    harnesses.append(harness)
                    slots.append(Slot(job, harness, config, args.output_dir))
                journal.update('collection', active_episodes=[s.job['episode_id'] for s in slots])
                for step in range(config.horizon):
                    if STOP_REQUESTED or time.monotonic() >= args.deadline:
                        break
                    active = [s for s in slots if not s.done]
                    if not active:
                        break
                    monitor_tick = step % config.monitor_stride == 0
                    inference = [s for s in active if monitor_tick or not s.queue]
                    if inference:
                        samples = [s.observe() for s in inference]
                        obs = base.preprocess_observation(combine([x[0] for x in samples]))
                        obs["task"] = [s.prompt for s in inference]
                        batch = preprocessor(base.env_preprocessor(obs))
                        noise = torch.cat([sample_shared_noise(policy, 1, torch.device(args.device),
                            seed=s.job["seed"] + 100003 * s.job["task_id"] + 1009 * step)
                            for s in inference])
                        tap.reset()
                        if matched:
                            matched.reset()
                        started = time.perf_counter()
                        with torch.inference_mode():
                            actions = policy.predict_action_chunk(batch, noise=noise, num_steps=config.inference_steps)
                        arrays = tap.arrays()  # CPU transfer synchronizes CUDA before recording latency.
                        if matched:
                            arrays.update(matched.arrays())
                        elapsed = time.perf_counter() - started
                        if context:
                            context.clear_records()
                        for i, slot in enumerate(inference):
                            if not slot.queue:
                                slot.queue = [a for a in actions[i, :config.action_steps].unbind(0)]
                            if monitor_tick:
                                _, images, visibility, state, audits = samples[i]
                                slot.rows.append({"step": step, "visibility": visibility,
                                                  "inference_batch_size": len(inference),
                                                  "inference_batch_seconds": elapsed})
                                for key, value in {"state": state, "vision": low_resolution_rgb(images),
                                                   **{k: v[i] for k, v in arrays.items()}}.items():
                                    slot.features.setdefault(key, []).append(value)
                                if step % args.image_stride == 0:
                                    path = slot.path / "images" / f"{step:04d}"
                                    path.mkdir(parents=True, exist_ok=True)
                                    futures.append(writer.submit(write_images, path, images, audits))
                                slot.save_partial()
                        (args.output_dir / "feature_schema.json").write_text(json.dumps(tap.schema, indent=2))
                        if matched:
                            atomic_json(args.output_dir / 'matched_feature_schema.json', matched.schema)
                    with torch.inference_mode():
                        normalized = torch.stack([s.queue.pop(0) for s in active])
                        processed = env_post({"action": postprocessor(normalized)})["action"]
                        controls = processed.detach().cpu().numpy()
                    for slot, action in zip(active, controls, strict=True):
                        if action.shape != (7,) or not np.isfinite(action).all():
                            raise RuntimeError("Invalid environment action")
                        # Match environment action bounds explicitly; log final executed commands.
                        action = np.clip(action, slot.inner.action_space.low, slot.inner.action_space.high)
                        slot.advance(action)
                        if slot.done:
                            for future in futures:
                                future.result()
                            futures.clear()
                            slot.save()
                            print(json.dumps({"episode": slot.job["episode_id"], "success": slot.success,
                                              "harm": any(e["harm"] for e in slot.events), "steps": slot.steps}), flush=True)
                    journal.update('collection', active_episodes=[s.job['episode_id'] for s in slots if not s.done],
                                   last_control_steps={s.job['episode_id']: s.steps for s in slots},
                                   completed_episodes=len(list((args.output_dir / 'episodes').glob('*/episode.json'))))
                for future in futures:
                    future.result()
                futures.clear()
            finally:
                for slot in slots:
                    slot.physics.close()
                    slot.event_log.close()
                for harness in harnesses:
                    harness.vec_env.close()


def main(argv=None):
    global STOP_REQUESTED
    STOP_REQUESTED = False
    args = arguments(argv)
    args.deadline = time.monotonic() + args.wall_seconds if args.wall_seconds > 0 else float('inf')
    if args.image_stride < 1:
        raise ValueError("image-stride must be positive")
    if args.comparison_width < 0:
        raise ValueError('comparison-width must be >= 0')
    presets = {"smoke": ((0,), 1, 1, 1), "pilot": ((0, 1), 3, 1, 2),
               "a100": (tuple(range(10)), 14, 5, 5)}
    tasks, train, cal, test = presets[args.preset]
    config = PilotConfig(tasks=tuple(map(int, args.tasks.split(','))) if args.tasks else tasks,
                         train_starts=train if args.train_starts is None else args.train_starts,
                         calibration_starts=cal if args.calibration_starts is None else args.calibration_starts,
                         test_starts=test if args.test_starts is None else args.test_starts, batch_size=args.batch_size,
                         action_steps=args.action_steps, monitor_stride=args.monitor_stride)
    payload = manifest(config)
    payload.update(policy_path=args.policy_path, policy_dtype=args.policy_dtype,
                   checkpoint=str(args.checkpoint.resolve()) if args.checkpoint else None,
                   image_stride=args.image_stride,
                   matched_comparison=dict(version=1, width=args.comparison_width,
                       component='action_expert', mode='probe', layers=list(config.layers)) if args.checkpoint else None,
                   git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
    if args.checkpoint:
        digest = hashlib.sha256()
        with args.checkpoint.open('rb') as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                digest.update(block)
        payload["checkpoint_sha256"] = digest.hexdigest()
    write_manifest(args.output_dir, payload)
    jobs = [j for j in payload["episodes"]
            if not (args.output_dir / "episodes" / j["episode_id"] / "episode.json").exists()]
    # Small early batches cover train/calibration/test before extending any one split.
    # Whole related-condition groups remain together; no split changes on resume.
    groups = {}
    for j in jobs:
        groups.setdefault(j['group_id'], []).append(j)
    queues = {s: [g for g in groups.values() if g[0]['split'] == s]
              for s in ('train', 'calibration', 'test')}
    jobs = []
    while any(queues.values()):
        for s in queues:
            if queues[s]:
                jobs.extend(queues[s].pop(0))
    print(json.dumps({"total_episodes": len(payload["episodes"]), "pending": len(jobs),
                      "unique_starts": len({j["group_id"] for j in payload["episodes"]}),
                      "gpu_inference_batch": config.batch_size,
                      "splits": {s: sum(j['split'] == s for j in payload['episodes'])
                                 for s in ('train', 'calibration', 'test')}}, indent=2))
    if not args.plan_only and jobs:
        journal = RunJournal(args.output_dir)
        try:
            collect(args, config, jobs)
            completed = len(list((args.output_dir / 'episodes').glob('*/episode.json')))
            journal.update('collection', 'complete' if completed == len(payload['episodes']) else 'budget_limited',
                           completed_episodes=completed, planned_episodes=len(payload['episodes']))
        except BaseException as exc:
            journal.update('collection', 'failed', error=f'{type(exc).__name__}: {exc}')
            raise


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    main()

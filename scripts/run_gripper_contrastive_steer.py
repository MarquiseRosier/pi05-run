#!/usr/bin/env python
"""Contrastive gripper open/close steering with trained Pi0.5 STCs.

This does not retrain transcoders and does not call lerobot-eval. It reads
LIBERO demonstration frames, probes the frozen policy, and applies an additive
latent direction through the existing probe residual.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import sys
import traceback
from pathlib import Path
from typing import Any

print("gripper-contrast: process started", flush=True)

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pi05_mi.atlas_concepts import task_id_from_prompt
from pi05_mi.gripper_contrast import (
    GRIPPER_INDEX,
    FrameRecord,
    LatentBank,
    action_effect,
    assign_splits,
    cap_records,
    cell_statistics,
    contrastive_direction,
    feature_rows,
    infer_gripper_convention,
    latent_relative_scale,
    mean_over_tokens,
    pair_records,
    permute_direction,
    progress_bin,
    quantize_tau,
    save_figures,
    select_best_cell,
    selection_summary,
    sparsify_direction,
    split_episode_ids,
    stable_window_label,
    write_csv,
    write_json,
)
from pi05_mi.patch_pi05 import Pi05TranscoderContext, install_pi05_action_expert_wrappers
from pi05_mi.transcoders import load_time_conditioned_transcoders


def _load_train_module():
    path = ROOT / "scripts" / "train_pi05_transcoders.py"
    spec = importlib.util.spec_from_file_location("train_pi05_transcoders", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_pi05_transcoders"] = module
    spec.loader.exec_module(module)
    return module


def _columns(obj: Any) -> set[str]:
    if hasattr(obj, "column_names"):
        return set(obj.column_names)
    if hasattr(obj, "columns"):
        return set(map(str, obj.columns))
    if isinstance(obj, dict):
        return set(obj)
    return set()


def _column(obj: Any, key: str):
    if isinstance(obj, dict):
        return obj[key]
    return obj[key]


def _episode_prompt(dataset: Any, episode_id: int) -> str:
    episodes = dataset.meta.episodes
    columns = _columns(episodes)
    if "tasks" in columns:
        raw = _column(episodes, "tasks")[episode_id]
        if isinstance(raw, (list, tuple)):
            return str(raw[0]) if raw else ""
        return str(raw)
    task_index = None
    if "task_index" in columns:
        task_index = int(_column(episodes, "task_index")[episode_id])
    tasks = dataset.meta.tasks
    if isinstance(tasks, dict):
        if task_index is None:
            raise KeyError(f"Episode {episode_id} has no task string")
        return str(tasks[task_index])
    task_columns = _columns(tasks)
    if task_index is not None and "task" in task_columns:
        return str(_column(tasks, "task")[task_index])
    raise KeyError(
        f"Cannot read a task prompt for episode {episode_id}. "
        f"Episode columns={sorted(columns)} task columns={sorted(task_columns)}"
    )


def _vector(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _grip_and_state(action: np.ndarray, state: np.ndarray) -> tuple[float, np.ndarray]:
    if action.ndim == 1:
        if action.shape[0] <= GRIPPER_INDEX:
            raise ValueError(f"Action width {action.shape[0]} has no gripper dimension {GRIPPER_INDEX}")
        grip = float(action[GRIPPER_INDEX])
    elif action.ndim == 2:
        grip = float(action[0, GRIPPER_INDEX])
    else:
        raise ValueError(f"Unexpected action shape {action.shape}")
    flat_state = state.reshape(-1) if state.ndim == 1 else state.reshape(-1, state.shape[-1])[0]
    return grip, flat_state


def _window_grips(actions: Any, start: int, horizon: int) -> list[float] | None:
    window: list[float] = []
    for step in range(horizon):
        action = _vector(actions[start + step])
        if action.ndim == 1:
            if action.shape[0] <= GRIPPER_INDEX:
                return None
            window.append(float(action[GRIPPER_INDEX]))
        elif action.ndim == 2:
            if action.shape[0] <= step or action.shape[-1] <= GRIPPER_INDEX:
                return None
            window.append(float(action[step, GRIPPER_INDEX]))
        else:
            return None
    return window


def scan_demonstrations(dataset: Any, *, suite: str, horizon: int) -> tuple[Any, list[FrameRecord], np.ndarray, list[str | None]]:
    hf = dataset.hf_dataset
    if "action" not in hf.column_names or "observation.state" not in hf.column_names:
        raise KeyError(f"Dataset is missing action or observation.state. Columns={hf.column_names}")
    actions = hf["action"]
    states = hf["observation.state"]
    episodes = dataset.meta.episodes
    starts = [int(value) for value in _column(episodes, "dataset_from_index")]
    ends = [int(value) for value in _column(episodes, "dataset_to_index")]

    suite_rows: list[tuple[int, int, int, int]] = []
    grip_values: list[float] = []
    state_rows: list[np.ndarray] = []
    for episode_id, (start, end) in enumerate(zip(starts, ends, strict=False)):
        prompt = _episode_prompt(dataset, episode_id)
        task_id = task_id_from_prompt(suite, prompt, space="lerobot")
        if task_id is None:
            continue
        for row in range(start, end):
            grip, state = _grip_and_state(_vector(actions[row]), _vector(states[row]))
            suite_rows.append((row, episode_id, task_id, row - start))
            grip_values.append(grip)
            state_rows.append(state)
    if not grip_values:
        raise RuntimeError(f"No demonstration frames matched suite {suite!r}")

    convention = infer_gripper_convention(np.asarray(grip_values), np.stack(state_rows, axis=0))
    labeled = [convention.label(value) for value in grip_values]
    by_episode: dict[int, list[tuple[int, int, int, int]]] = {}
    for item in suite_rows:
        by_episode.setdefault(item[1], []).append(item)

    records: list[FrameRecord] = []
    for episode_id, rows in by_episode.items():
        rows.sort(key=lambda item: item[0])
        episode_length = len(rows)
        for row, _episode, task_id, frame in rows:
            if frame + horizon > episode_length:
                continue
            window = _window_grips(actions, row, horizon)
            if window is None:
                continue
            label = stable_window_label(window, convention)
            if label is None:
                continue
            records.append(
                FrameRecord(
                    index=row,
                    episode_id=episode_id,
                    task_id=task_id,
                    frame_in_episode=frame,
                    episode_length=episode_length,
                    progress_bin=progress_bin(frame, episode_length),
                    label=label,
                    split="train",
                )
            )
    return convention, records, np.asarray(grip_values, dtype=np.float64), labeled


def _numpy_actions(actions: Any) -> np.ndarray:
    if isinstance(actions, dict):
        actions = actions.get("action", next(iter(actions.values())))
    if hasattr(actions, "detach"):
        actions = actions.detach().cpu().float().numpy()
    return np.asarray(actions)


def _predict(policy, preprocessor, postprocessor, dataset, index: int, camera_keys: list[str], train_mod, num_steps: int | None):
    from lerobot.utils.collate import lerobot_collate_fn

    raw = lerobot_collate_fn([dataset[int(index)]])
    raw = train_mod._prepare_raw_batch(raw, camera_keys)
    batch = preprocessor(raw)
    kwargs = {} if num_steps is None else {"num_steps": int(num_steps)}
    actions = policy.predict_action_chunk(batch, **kwargs)
    actions = postprocessor(actions)
    return _numpy_actions(actions)


def _store_latents(context: Pi05TranscoderContext, bank: LatentBank, example_id: int, horizon: int) -> int:
    stored = 0
    tokens = list(range(horizon))
    for records in context.latents.values():
        for record in records:
            if record.full_latent is None:
                continue
            tau = float(record.timestep.detach().float().reshape(-1)[0].item())
            bank.add(example_id, record.layer_index, tau, mean_over_tokens(record.full_latent.numpy(), tokens))
            stored += 1
    context.clear_records()
    return stored


def _rank(bank: LatentBank, train_records: list[FrameRecord], pairs: list[tuple[FrameRecord, FrameRecord]], ranking_limit: int):
    open_ids = [record.index for record in train_records if record.label == "open"]
    close_ids = [record.index for record in train_records if record.label == "close"]
    pair_open = [open_record.index for open_record, _close_record in pairs]
    pair_close = [close_record.index for _open_record, close_record in pairs]
    if not pair_open:
        raise RuntimeError("No matched open/close pairs are available to build a direction")

    global_rows: list[dict[str, Any]] = []
    cell_scores: list[tuple[int, float, float]] = []
    best: tuple[int, float, dict[str, np.ndarray]] | None = None
    for layer, tau in bank.cells():
        stats = cell_statistics(
            bank.stack(open_ids, layer, tau),
            bank.stack(close_ids, layer, tau),
            bank.stack(pair_open, layer, tau),
            bank.stack(pair_close, layer, tau),
        )
        score = float(stats["score"].max())
        cell_scores.append((layer, tau, score))
        top_index = np.argsort(-stats["score"])[:50]
        global_rows.extend(feature_rows(layer, tau, stats, top_index))
        if best is None or score > float(best[2]["score"].max()):
            best = (layer, tau, stats)
    if best is None:
        raise RuntimeError("STC probe produced no latent cells")
    layer, tau, stats = best
    chosen_layer, chosen_tau = select_best_cell(cell_scores)
    if (chosen_layer, quantize_tau(chosen_tau)) != (layer, quantize_tau(tau)):
        raise RuntimeError("Best-cell selection did not match the stored statistics")
    selected_rows = feature_rows(layer, tau, stats)
    selected_rows.sort(key=lambda row: float(row["score"]), reverse=True)
    global_rows.sort(key=lambda row: float(row["score"]), reverse=True)
    direction = contrastive_direction(bank.stack(pair_open, layer, tau), bank.stack(pair_close, layer, tau))
    return (
        global_rows[: int(ranking_limit)],
        selected_rows,
        layer,
        quantize_tau(tau),
        direction,
        np.asarray(stats["sign_consistency"]),
    )


def _pool_layer(bank: LatentBank, layer: int, example_ids: list[int], taus: list[float]) -> np.ndarray:
    rows = []
    for example_id in example_ids:
        stacked = np.stack([bank.stack([example_id], layer, tau)[0] for tau in taus], axis=0)
        rows.append(stacked.mean(axis=0))
    return np.stack(rows, axis=0)


def _layer_rms(bank: LatentBank, layer: int, example_ids: list[int], taus: list[float]) -> float:
    chunks = [bank.stack(example_ids, layer, tau) for tau in taus]
    values = np.concatenate(chunks, axis=0).astype(np.float64)
    return float(np.sqrt(np.mean(np.square(values))))


def _balanced_ids(records: list[FrameRecord], seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    opens = [record.index for record in records if record.label == "open"]
    closes = [record.index for record in records if record.label == "close"]
    count = min(len(opens), len(closes))
    if count == 0:
        raise RuntimeError(
            f"Agreement frames do not contain both gripper states (open={len(opens)}, close={len(closes)})"
        )
    open_ids = rng.choice(np.asarray(opens), size=count, replace=False)
    close_ids = rng.choice(np.asarray(closes), size=count, replace=False)
    return [int(item) for item in open_ids], [int(item) for item in close_ids]


def _steering_rows(
    *,
    policy,
    preprocessor,
    postprocessor,
    dataset,
    camera_keys,
    train_mod,
    context,
    holdout: list[FrameRecord],
    layer: int,
    tau: float | None,
    timesteps: list[float] | None,
    directions: dict[str, np.ndarray],
    alphas: list[float],
    scale_per_alpha: float,
    horizon: int,
    num_steps: int | None,
) -> list[dict[str, Any]]:
    import torch

    rows: list[dict[str, Any]] = []
    tau_label: float | str = "all" if timesteps is None else float(tau)
    for record in holdout:
        print(f"steering example {record.index} label={record.label}", flush=True)
        context.capture_latents = False
        context.save_full_latents = False
        context.set_latent_delta(None)
        baseline = _predict(policy, preprocessor, postprocessor, dataset, record.index, camera_keys, train_mod, num_steps)
        for kind in ("full", "topk"):
            vector = directions[kind]
            for alpha in alphas:
                if alpha == 0.0:
                    controls = [("plus", 0.0), ("minus", 0.0), ("random", 0.0)]
                    if kind == "full":
                        controls = [("baseline", 0.0), *controls]
                else:
                    controls = [("plus", alpha), ("minus", -alpha), ("random", alpha)]
                for control, scale in controls:
                    used = directions["random_full" if kind == "full" else "random_topk"] if control == "random" else vector
                    applied = float(scale) * float(scale_per_alpha)
                    if applied == 0.0:
                        context.set_latent_delta(None)
                        steered = baseline
                    else:
                        context.set_latent_delta(
                            torch.from_numpy(np.asarray(used, dtype=np.float32)),
                            scale=applied,
                            layers=[layer],
                            timesteps=timesteps,
                        )
                        steered = _predict(
                            policy,
                            preprocessor,
                            postprocessor,
                            dataset,
                            record.index,
                            camera_keys,
                            train_mod,
                            num_steps,
                        )
                    effect = action_effect(baseline, steered, horizon=horizon)
                    rows.append(
                        {
                            "example_id": record.index,
                            "episode_id": record.episode_id,
                            "task_id": record.task_id,
                            "label": record.label,
                            "layer": layer,
                            "tau": tau_label,
                            "direction_kind": "baseline" if control == "baseline" else kind,
                            "control": control,
                            "alpha": 0.0 if control == "baseline" else alpha,
                            "applied_scale": 0.0 if control == "baseline" else applied,
                            **effect,
                        }
                    )
        context.set_latent_delta(None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def run(args: argparse.Namespace) -> None:
    import torch

    print("gripper-contrast: importing LeRobot helpers", flush=True)
    # draccus.parse reads sys.argv when args is omitted. Keep this script's
    # flags out of TrainPipelineConfig decoding.
    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    try:
        train_mod = _load_train_module()
        train_mod.patch_transformers_causal_mask_compat()
        train_mod.patch_pi05_checkpoint_key_compat()
        device = train_mod.resolve_device(args.device)
        policy_dtype = train_mod.resolve_policy_dtype(args.policy_dtype, device)
        namespace = argparse.Namespace(
            policy_path=args.policy_path,
            local_files_only=args.local_files_only,
            batch_size=1,
            num_workers=0,
            resolved_device=device,
            resolved_policy_dtype=policy_dtype,
        )
        cfg = train_mod._configure_train_config(namespace, episodes=None)
    finally:
        sys.argv = saved_argv
    if hasattr(cfg.dataset, "video_backend"):
        cfg.dataset.video_backend = "pyav"
    print(f"loading dataset {cfg.dataset.repo_id}", flush=True)
    dataset = train_mod.make_dataset(cfg)
    convention, records, grip_values, grip_labels = scan_demonstrations(dataset, suite=args.suite, horizon=args.horizon)
    print(
        "gripper convention "
        f"open_is_high={convention.open_is_high} "
        f"thresholds=({convention.low_threshold:.4f}, {convention.high_threshold:.4f}) "
        f"finger_gap_corr={convention.finger_gap_correlation:.4f}",
        flush=True,
    )
    train_episodes, holdout_episodes = split_episode_ids(
        [record.episode_id for record in records],
        train_fraction=args.train_fraction,
        seed=args.seed,
    )
    assigned = assign_splits(records, train_episodes, holdout_episodes)
    probed = cap_records(
        assigned,
        max_per_group=args.max_per_group,
        max_holdout_per_label=args.max_holdout_per_label,
        seed=args.seed,
    )
    train_records = [record for record in probed if record.split == "train"]
    holdout_records = [record for record in probed if record.split == "holdout"]
    pairs = pair_records(train_records, seed=args.seed)
    n_open = sum(record.label == "open" for record in assigned)
    n_close = sum(record.label == "close" for record in assigned)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = selection_summary(
        convention=convention,
        n_frames=int(grip_values.size),
        n_stable_open=n_open,
        n_stable_close=n_close,
        n_pairs=len(pairs),
        n_train=len(train_records),
        n_holdout=len(holdout_records),
        train_episodes=train_episodes,
        holdout_episodes=holdout_episodes,
        horizon=args.horizon,
    )
    summary["n_stable_open_before_cap"] = n_open
    summary["n_stable_close_before_cap"] = n_close
    summary["score_definition"] = "abs(mean_open - mean_close) * sign_consistency * max(open_firing_frequency, close_firing_frequency)"
    if not pairs:
        write_json(output_dir / "selection_report.json", summary)
        raise RuntimeError("Stable open and close frames do not form any within-task progress-bin pairs")
    if not holdout_records:
        write_json(output_dir / "selection_report.json", summary)
        raise RuntimeError("Holdout split has no capped open/close frames")

    print("loading preprocessor and frozen Pi0.5", flush=True)
    from lerobot.policies import make_policy, make_pre_post_processors

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=args.policy_path,
        pretrained_revision=getattr(cfg.policy, "pretrained_revision", None),
        dataset_stats=dataset.meta.stats,
        dataset_meta=dataset.meta,
        preprocessor_overrides={
            "device_processor": {"device": str(cfg.policy.device)},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )
    policy = make_policy(cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)

    torch_dtype = torch.bfloat16 if policy_dtype == "bfloat16" else torch.float32
    if policy_dtype == "float16":
        torch_dtype = torch.float16
    print(f"loading STCs from {args.checkpoint}", flush=True)
    transcoders = load_time_conditioned_transcoders(args.checkpoint, device=device, dtype=torch_dtype)
    context = Pi05TranscoderContext(
        mode="probe",
        capture_records=False,
        capture_latents=True,
        latent_top_k=1,
        save_full_latents=True,
    )
    _context, wrapped = install_pi05_action_expert_wrappers(
        policy,
        context=context,
        transcoders=transcoders,
        mode="probe",
    )
    print(f"wrapped {len(wrapped)} action-expert MLPs", flush=True)
    camera_keys = list(dataset.meta.camera_keys)
    bank = LatentBank()
    agreements = []
    for record in train_records:
        context.set_latent_delta(None)
        context.capture_latents = True
        context.save_full_latents = True
        context.clear_records()
        actions = _predict(
            policy,
            preprocessor,
            postprocessor,
            dataset,
            record.index,
            camera_keys,
            train_mod,
            args.num_inference_steps,
        )
        stored = _store_latents(context, bank, record.index, args.horizon)
        if stored == 0:
            raise RuntimeError(f"Probe stored no full latents for example {record.index}")
        effect = action_effect(actions, actions, horizon=args.horizon)
        predicted = convention.label(effect["baseline_gripper"])
        agreements.append(
            {
                "example_id": record.index,
                "label": record.label,
                "predicted_gripper": effect["baseline_gripper"],
                "predicted_label": predicted,
                "agree": predicted == record.label,
            }
        )
        print(
            f"probed example {record.index} label={record.label} latents={stored} "
            f"predicted_gripper={effect['baseline_gripper']:.4f}",
            flush=True,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.agreement_only:
        if args.steer_layer is None:
            raise RuntimeError("--agreement-only requires --steer-layer")
        agree_ids = {int(row["example_id"]) for row in agreements if row["agree"]}
        agree_records = [record for record in train_records if record.index in agree_ids]
        n_agree_open = sum(record.label == "open" for record in agree_records)
        n_agree_close = sum(record.label == "close" for record in agree_records)
        print(
            f"agreement frames open={n_agree_open} close={n_agree_close} of {len(train_records)}",
            flush=True,
        )
        if n_agree_open == 0 or n_agree_close == 0:
            raise RuntimeError(
                f"Need both gripper states among agreement frames (open={n_agree_open}, close={n_agree_close})"
            )
        layer = int(args.steer_layer)
        taus = [item_tau for item_layer, item_tau in bank.cells() if item_layer == layer]
        if not taus:
            raise RuntimeError(f"Probe stored no latents at layer {layer}")
        agree_pairs = pair_records(agree_records, seed=args.seed)
        if agree_pairs:
            pair_open = [open_record.index for open_record, _close_record in agree_pairs]
            pair_close = [close_record.index for _open_record, close_record in agree_pairs]
            direction_source = "paired"
        else:
            pair_open, pair_close = _balanced_ids(agree_records, args.seed)
            direction_source = "balanced"
        open_ids = [record.index for record in agree_records if record.label == "open"]
        close_ids = [record.index for record in agree_records if record.label == "close"]
        pooled_stats = cell_statistics(
            _pool_layer(bank, layer, open_ids, taus),
            _pool_layer(bank, layer, close_ids, taus),
            _pool_layer(bank, layer, pair_open, taus),
            _pool_layer(bank, layer, pair_close, taus),
        )
        direction = np.asarray(pooled_stats["direction"])
        consistency = np.asarray(pooled_stats["sign_consistency"])
        ranking_rows: list[dict[str, Any]] = []
        for item_tau in taus:
            stats = cell_statistics(
                bank.stack(open_ids, layer, item_tau),
                bank.stack(close_ids, layer, item_tau),
                bank.stack(pair_open, layer, item_tau),
                bank.stack(pair_close, layer, item_tau),
            )
            top_index = np.argsort(-stats["score"])[:50]
            ranking_rows.extend(feature_rows(layer, item_tau, stats, top_index))
        ranking_rows.sort(key=lambda row: float(row["score"]), reverse=True)
        ranking_rows = ranking_rows[: int(args.ranking_limit)]
        selected_rows = feature_rows(layer, float(np.mean(taus)), pooled_stats)
        selected_rows.sort(key=lambda row: float(row["score"]), reverse=True)
        latent_rms = _layer_rms(bank, layer, open_ids + close_ids, taus)
        tau = None
        summary["direction_source"] = direction_source
        summary["n_agreement_open"] = n_agree_open
        summary["n_agreement_close"] = n_agree_close
        summary["n_direction_examples"] = len(pair_open)
        summary["steer_taus"] = taus
        print(
            f"direction source={direction_source} pairs={len(pair_open)} layer={layer} taus={len(taus)}",
            flush=True,
        )
    else:
        ranking_rows, selected_rows, layer, tau, direction, consistency = _rank(
            bank, train_records, pairs, args.ranking_limit
        )
        latent_rms = None
        if args.alpha_unit == "latent-rms" or args.all_timesteps:
            taus = [item_tau for item_layer, item_tau in bank.cells() if item_layer == layer]
            latent_rms = _layer_rms(bank, layer, [record.index for record in train_records], taus)

    sparse, kept = sparsify_direction(
        direction,
        consistency,
        top_k=args.top_k,
        min_consistency=args.min_consistency,
    )
    random_full = permute_direction(direction, args.seed)
    random_sparse = permute_direction(sparse, args.seed + 1)
    torch.save(
        {"layer": layer, "tau": tau, "vector": torch.from_numpy(direction.astype(np.float32))},
        output_dir / "direction_full.pt",
    )
    torch.save(
        {
            "layer": layer,
            "tau": tau,
            "vector": torch.from_numpy(sparse.astype(np.float32)),
            "feature_ids": torch.from_numpy(kept.astype(np.int64)),
        },
        output_dir / "direction_topk.pt",
    )
    write_csv(output_dir / "feature_ranking.csv", ranking_rows)
    write_csv(output_dir / "selected_cell_features.csv", selected_rows)
    timesteps = None if args.all_timesteps else [float(tau)]
    scale_per_alpha = 1.0
    if args.alpha_unit == "latent-rms":
        scale_per_alpha = latent_relative_scale(direction, float(latent_rms))
    alpha_label = "alpha (x latent RMS)" if args.alpha_unit == "latent-rms" else "alpha"
    print(
        f"alpha unit={args.alpha_unit} latent_rms={latent_rms} scale_per_alpha={scale_per_alpha:.6g} "
        f"timesteps={'all' if timesteps is None else timesteps}",
        flush=True,
    )
    summary["selected_layer"] = layer
    summary["selected_tau"] = "all" if timesteps is None else tau
    summary["steer_timesteps"] = "all" if timesteps is None else timesteps
    summary["alpha_unit"] = args.alpha_unit
    summary["latent_rms"] = latent_rms
    summary["scale_per_alpha"] = scale_per_alpha
    summary["top_k"] = int(args.top_k)
    summary["n_topk_features"] = int(kept.size)
    summary["probe_agreement"] = {
        "n": len(agreements),
        "n_agree": int(sum(bool(row["agree"]) for row in agreements)),
        "rows": agreements,
    }
    write_json(output_dir / "selection_report.json", summary)

    alphas = [float(item) for item in args.alphas.split(",") if item.strip()]
    steering_rows = _steering_rows(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        dataset=dataset,
        camera_keys=camera_keys,
        train_mod=train_mod,
        context=context,
        holdout=holdout_records,
        layer=layer,
        tau=None if timesteps is None else float(tau),
        timesteps=timesteps,
        directions={
            "full": direction,
            "topk": sparse,
            "random_full": random_full,
            "random_topk": random_sparse,
        },
        alphas=alphas,
        scale_per_alpha=scale_per_alpha,
        horizon=args.horizon,
        num_steps=args.num_inference_steps,
    )
    write_csv(output_dir / "steering_results.csv", steering_rows)
    figures = save_figures(
        output_dir,
        ranking_rows=ranking_rows,
        steering_rows=steering_rows,
        grip_values=grip_values,
        grip_labels=grip_labels,
        convention=convention,
        alpha_label=alpha_label,
    )
    print(f"wrote {output_dir}", flush=True)
    for figure in figures:
        print(figure, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", default="lerobot/pi05_libero_finetuned")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--output-dir", default="outputs/gripper_contrastive")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-per-group", type=int, default=2)
    parser.add_argument("--max-holdout-per-label", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--min-consistency", type=float, default=0.7)
    parser.add_argument("--alphas", default="0,0.5,1,2,4")
    parser.add_argument("--ranking-limit", type=int, default=2000)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--agreement-only", action="store_true")
    parser.add_argument("--steer-layer", type=int, default=None)
    parser.add_argument("--all-timesteps", action="store_true")
    parser.add_argument("--alpha-unit", choices=("raw", "latent-rms"), default="raw")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--policy-dtype", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run(parse_args())
    except SystemExit as exc:
        print(f"gripper-contrast: SystemExit {exc.code}", flush=True)
        raise
    except Exception:
        traceback.print_exc()
        raise

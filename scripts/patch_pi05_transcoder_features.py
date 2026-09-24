#!/usr/bin/env python
"""Causally test the features the counterfactual probe flagged.

The probe is correlational: it says which transcoder features move when one
object's appearance changes. This script intervenes on those features and asks
whether they *carry* that information.

Two directions, both interchange interventions on a frozen simulator state:

* **replay** -- run the unperturbed image but substitute the candidate features
  with the values they took on the perturbed image. If those features carry the
  appearance change, the action should move toward the perturbed action even
  though not a single pixel changed.
* **dull** -- run the perturbed image but restore the candidate features to
  their unperturbed values. If they carry it, the action should fall back
  toward the baseline: the model stops seeing the recolour.

Everything runs in ``replace`` mode, where the transcoder output *is* the MLP
output, so the substitution actually propagates. All conditions share that
mode, so the transcoder's own reconstruction error cancels in the comparisons.

Controls, without which none of this means anything:

* **random** -- patch the same number of randomly chosen features per layer.
  If random patching moves the action as much as the candidates do, the
  candidates are not special.
* **all** -- donate every feature. This is the ceiling the mechanism can reach
  and confirms the patch path is wired correctly end to end.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import probe_pi05_transcoder_counterfactual as probe  # noqa: E402
from pi05_mi.patch_pi05 import Pi05TranscoderContext, install_pi05_action_expert_wrappers  # noqa: E402
from pi05_mi.scene_perturbation import image_delta_stats, resolve_mj_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("probe_run", type=Path, help="A counterfactual probe run directory.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--policy-path", default="lerobot/pi05_libero_finetuned")
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--target", default=None, help="Defaults to the probe run's target.")
    parser.add_argument("--dose", type=float, default=1.0)
    parser.add_argument("--color", default="1.0,0.2,0.1")
    parser.add_argument("--perturbation", choices=["blend", "set", "hue"], default="blend")
    parser.add_argument("--states", type=int, default=2)
    parser.add_argument("--state-stride", type=int, default=5)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--top-features", type=int, default=100, help="Candidate features to patch, total.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--policy-dtype", default="bfloat16")
    parser.add_argument("--seed-random-control", type=int, default=7)
    return parser.parse_args()


# ---------------------------------------------------------------- patching


class FeaturePatcher:
    """Record a donor pass's latents for chosen features, then substitute them.

    Only the selected features are stored, so memory stays negligible even
    though a full latent is (batch, tokens, ~16k) per layer per denoise step.

    Steps are counted per layer exactly as the probe's accumulator does, so
    donor and recipient line up step for step.
    """

    def __init__(self, features_by_layer: dict[str, list[int]]) -> None:
        self.index_by_layer = {
            layer: torch.as_tensor(sorted(set(ids)), dtype=torch.long)
            for layer, ids in features_by_layer.items()
            if ids
        }
        self.mode = "off"  # off | record | apply | zero
        self.store: dict[tuple[str, int], torch.Tensor] = {}
        self._step: dict[str, int] = defaultdict(int)
        self.applied = 0

    @property
    def n_features(self) -> int:
        return int(sum(len(idx) for idx in self.index_by_layer.values()))

    def begin(self, mode: str) -> None:
        self.mode = mode
        self._step.clear()
        if mode == "record":
            self.store.clear()
        self.applied = 0

    def __call__(self, name: str, layer_index: int, latent: torch.Tensor, timestep: torch.Tensor):
        index = self.index_by_layer.get(name)
        if index is None or self.mode == "off":
            return None
        step = self._step[name]
        self._step[name] += 1
        index = index.to(latent.device)

        if self.mode == "record":
            self.store[(name, step)] = latent.detach()[..., index].clone()
            return None
        if self.mode == "zero":
            patched = latent.clone()
            patched[..., index] = 0.0
            self.applied += 1
            return patched
        if self.mode == "apply":
            donor = self.store.get((name, step))
            if donor is None:
                return None
            patched = latent.clone()
            patched[..., index] = donor.to(device=patched.device, dtype=patched.dtype)
            self.applied += 1
            return patched
        return None


def load_candidates(run_dir: Path, *, top: int) -> tuple[dict[str, list[int]], list[dict[str, Any]]]:
    path = run_dir / "candidate_features.csv"
    if not path.exists():
        raise SystemExit(
            f"No candidate_features.csv in {run_dir}. Run report_pi05_counterfactual_features.py first."
        )
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    chosen = rows[:top]
    by_layer: dict[str, list[int]] = defaultdict(list)
    for row in chosen:
        by_layer[row["layer"]].append(int(row["feature"]))
    return dict(by_layer), chosen


def random_like(
    reference: dict[str, list[int]], *, n_features_total: int, generator: torch.Generator
) -> dict[str, list[int]]:
    """Same per-layer counts as the candidates, but arbitrary features."""
    out: dict[str, list[int]] = {}
    for layer, ids in reference.items():
        picks = torch.randperm(n_features_total, generator=generator)[: len(ids)]
        out[layer] = sorted(int(value) for value in picks)
    return out


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(b))
    return float("nan") if denominator == 0 else float(np.linalg.norm(a - b) / denominator)


def recovery(patched: np.ndarray, toward: np.ndarray, away: np.ndarray) -> float:
    """1.0 = the patch fully reproduced the donor action, 0.0 = no movement."""
    gap = float(np.linalg.norm(away - toward))
    if gap == 0:
        return float("nan")
    return float(1.0 - np.linalg.norm(patched - toward) / gap)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.probe_run / "patching")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = args.probe_run / "counterfactual_summary.json"
    probe_config = {}
    if summary_path.exists():
        probe_config = json.loads(summary_path.read_text()).get("config", {})
    target = args.target or probe_config.get("target")
    if not target:
        raise SystemExit("No --target given and the probe run has none recorded.")
    if args.prompt is None:
        args.prompt = probe_config.get("prompt")

    features_by_layer, chosen_rows = load_candidates(args.probe_run, top=args.top_features)
    print(f"patching {sum(len(v) for v in features_by_layer.values())} features across "
          f"{len(features_by_layer)} layers, from {args.probe_run}", flush=True)
    for layer, ids in sorted(features_by_layer.items()):
        print(f"  {layer}: {len(ids)} features", flush=True)

    harness = probe.build_harness(args, load_policy=True)
    harness.vec_env.reset(seed=[args.seed])
    if not harness.prompt:
        harness.prompt = probe_config.get("prompt") or ""
    mj_model = resolve_mj_model(harness.inner_env._env)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    transcoders = probe.load_transcoders(args.checkpoint, device)
    n_features_total = int(next(iter(transcoders.values())).config.latent_dim)

    patcher = FeaturePatcher(features_by_layer)
    generator = torch.Generator().manual_seed(args.seed_random_control)
    random_patcher = FeaturePatcher(
        random_like(features_by_layer, n_features_total=n_features_total, generator=generator)
    )
    all_patcher = FeaturePatcher({name: list(range(n_features_total)) for name in features_by_layer})

    active: list[FeaturePatcher] = []

    def dispatch(name: str, layer_index: int, latent: torch.Tensor, timestep: torch.Tensor):
        for item in active:
            result = item(name, layer_index, latent, timestep)
            if result is not None:
                return result
        return None

    context = Pi05TranscoderContext(
        mode="replace",
        capture_records=False,
        capture_latents=False,
        store_latent_summaries=False,
        latent_intervention=dispatch,
    )
    _ctx, wrapped = install_pi05_action_expert_wrappers(
        harness.policy, context=context, transcoders=transcoders, mode="replace"
    )
    missing = [name for name in wrapped if name not in transcoders]
    if missing:
        raise SystemExit(f"{len(missing)} wrapped MLPs have no transcoder; cannot run replace mode.")
    unknown = [layer for layer in features_by_layer if layer not in set(wrapped)]
    if unknown:
        raise SystemExit(f"Candidate layers absent from the model: {unknown[:3]}")
    print(f"wrapped {len(wrapped)} MLPs in replace mode; latent dim {n_features_total}", flush=True)

    def run(observation, noise, *, patchers: list[FeaturePatcher], mode: str) -> np.ndarray:
        active.clear()
        for item in patchers:
            item.begin(mode)
            active.append(item)
        try:
            batch = probe.observation_to_batch(harness, observation)
            with torch.inference_mode():
                actions = harness.policy.predict_action_chunk(
                    batch, num_steps=args.num_inference_steps, noise=noise
                )
        finally:
            active.clear()
        return actions.detach().float().cpu().numpy()[0]

    rows: list[dict[str, Any]] = []
    for state_index in range(args.states):
        print(f"\n=== state {state_index} ===", flush=True)
        baseline_obs = probe.rerender_observation(harness)
        baseline_image = probe.first_camera_image(baseline_obs)
        noise = probe.sample_shared_noise(harness.policy, 1, device)

        perturbation = probe.apply_perturbation(mj_model, args, target, args.dose)
        try:
            perturbed_obs = probe.rerender_observation(harness)
            perturbed_image = probe.first_camera_image(perturbed_obs)
            pixel = image_delta_stats(baseline_image, perturbed_image)
            if pixel["changed_pixel_fraction"] == 0.0:
                raise SystemExit("The perturbation changed no pixels; nothing to patch.")

            # Unpatched endpoints, both in replace mode so reconstruction error cancels.
            action_baseline = run(baseline_obs, noise, patchers=[], mode="off")
            action_perturbed = run(perturbed_obs, noise, patchers=[], mode="off")
            gap = _rel(action_perturbed, action_baseline)
            print(f"  recolour moves the action by rel_l2={gap:.5g} "
                  f"({pixel['changed_pixel_fraction']*100:.2f}% of pixels)", flush=True)

            conditions: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]] = []

            for label, donor_obs, recipient_obs, toward, away in [
                ("replay", perturbed_obs, baseline_obs, action_perturbed, action_baseline),
                ("dull", baseline_obs, perturbed_obs, action_baseline, action_perturbed),
            ]:
                for patch_name, item in [("candidates", patcher), ("random", random_patcher), ("all", all_patcher)]:
                    run(donor_obs, noise, patchers=[item], mode="record")
                    patched_action = run(recipient_obs, noise, patchers=[item], mode="apply")
                    conditions.append((label, patch_name, patched_action, toward, away))

            # Zero-ablate the candidates on the perturbed image.
            zero_action = run(perturbed_obs, noise, patchers=[patcher], mode="zero")
            conditions.append(("zero_ablate", "candidates", zero_action, action_baseline, action_perturbed))

            print(f"  {'direction':<12} {'patch':<11} {'recovery':>9} {'to_target':>10} {'to_source':>10}")
            for label, patch_name, patched_action, toward, away in conditions:
                rec = recovery(patched_action, toward, away)
                row = {
                    "state_index": state_index,
                    "direction": label,
                    "patch": patch_name,
                    "n_features": (
                        all_patcher.n_features if patch_name == "all" else patcher.n_features
                    ),
                    "recovery": rec,
                    "rel_to_target": _rel(patched_action, toward),
                    "rel_to_source": _rel(patched_action, away),
                    "baseline_gap_rel_l2": gap,
                    "changed_pixel_fraction": pixel["changed_pixel_fraction"],
                }
                rows.append(row)
                print(
                    f"  {label:<12} {patch_name:<11} {rec:>9.3f} "
                    f"{row['rel_to_target']:>10.5g} {row['rel_to_source']:>10.5g}"
                )
        finally:
            perturbation.revert(mj_model)

        for _ in range(args.state_stride):
            harness.vec_env.step(np.asarray(action_baseline[0], dtype=np.float32)[None, ...])

    _write_csv(output_dir / "patching_results.csv", rows)
    verdict = _verdict(rows)
    (output_dir / "patching_summary.json").write_text(
        json.dumps(
            {
                "probe_run": str(args.probe_run),
                "target": target,
                "dose": args.dose,
                "n_candidate_features": patcher.n_features,
                "latent_dim": n_features_total,
                "candidates": chosen_rows,
                "results": rows,
                "verdict": verdict,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print("\n--- verdict ---")
    for line in verdict:
        print(" ", line)
    print(f"\nArtifacts in {output_dir}")
    harness.vec_env.close()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: list[dict[str, Any]], direction: str, patch: str) -> float:
    values = [
        row["recovery"]
        for row in rows
        if row["direction"] == direction and row["patch"] == patch and not np.isnan(row["recovery"])
    ]
    return float(np.mean(values)) if values else float("nan")


def _verdict(rows: list[dict[str, Any]]) -> list[str]:
    lines = []
    ceiling = _mean(rows, "replay", "all")
    if not np.isnan(ceiling):
        lines.append(f"Donating every feature recovers {ceiling:.1%} of the action change (mechanism ceiling).")
        if ceiling < 0.9:
            lines.append(
                "That ceiling is well below 100%, so the transcoder features do not fully mediate "
                "this layer and partial patches cannot be read cleanly."
            )
    for direction, description in [
        ("replay", "injecting the perturbed values into the clean image"),
        ("dull", "restoring the clean values on the perturbed image"),
    ]:
        candidates = _mean(rows, direction, "candidates")
        random_control = _mean(rows, direction, "random")
        if np.isnan(candidates):
            continue
        lines.append(f"{direction}: {description} recovers {candidates:.1%}; random features {random_control:.1%}.")
        if np.isnan(random_control) or random_control <= 0:
            continue
        ratio = candidates / random_control if random_control else float("inf")
        if candidates > 0.2 and ratio > 2.0:
            lines.append(f"  Candidates beat the random control by {ratio:.1f}x -- evidence they carry the signal.")
        elif ratio <= 1.5:
            lines.append(
                "  Candidates are not meaningfully better than random features, so this does not "
                "show they carry the signal."
            )
    zero = _mean(rows, "zero_ablate", "candidates")
    if not np.isnan(zero):
        lines.append(f"zero-ablating the candidates on the perturbed image moves {zero:.1%} back toward baseline.")
    return lines


if __name__ == "__main__":
    main()

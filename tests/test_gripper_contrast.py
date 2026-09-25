"""Tests for gripper open/close contrast math. Pi0.5 is not loaded."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MODULE_PATH = SRC / "pi05_mi" / "gripper_contrast.py"
spec = importlib.util.spec_from_file_location("gripper_contrast", MODULE_PATH)
gripper_contrast = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
sys.modules["gripper_contrast"] = gripper_contrast
spec.loader.exec_module(gripper_contrast)

FrameRecord = gripper_contrast.FrameRecord
LatentBank = gripper_contrast.LatentBank
action_effect = gripper_contrast.action_effect
assign_splits = gripper_contrast.assign_splits
cap_records = gripper_contrast.cap_records
cell_statistics = gripper_contrast.cell_statistics
infer_gripper_convention = gripper_contrast.infer_gripper_convention
mean_over_tokens = gripper_contrast.mean_over_tokens
pair_records = gripper_contrast.pair_records
permute_direction = gripper_contrast.permute_direction
progress_bin = gripper_contrast.progress_bin
save_figures = gripper_contrast.save_figures
sparsify_direction = gripper_contrast.sparsify_direction
split_episode_ids = gripper_contrast.split_episode_ids
stable_window_label = gripper_contrast.stable_window_label


def _record(index: int, episode: int, task: int, frame: int, label: str, split: str = "train") -> FrameRecord:
    return FrameRecord(
        index=index,
        episode_id=episode,
        task_id=task,
        frame_in_episode=frame,
        episode_length=100,
        progress_bin=progress_bin(frame, 100),
        label=label,
        split=split,
    )


class GripperConventionTests(unittest.TestCase):
    def test_finger_gap_assigns_open_without_a_fixed_sign(self) -> None:
        grip = np.concatenate([np.full(40, -1.0), np.full(40, 1.0)])
        state = np.zeros((80, 8))
        state[:40, -2:] = 0.04
        state[40:, -2:] = 0.0
        convention = infer_gripper_convention(grip, state)
        self.assertFalse(convention.open_is_high)
        self.assertEqual(convention.label(-1.0), "open")
        self.assertEqual(convention.label(1.0), "close")
        self.assertIsNone(convention.label(0.0))

        state[:40, -2:] = 0.0
        state[40:, -2:] = 0.04
        flipped = infer_gripper_convention(grip, state)
        self.assertTrue(flipped.open_is_high)
        self.assertEqual(flipped.label(1.0), "open")
        self.assertEqual(flipped.label(-1.0), "close")

    def test_stable_window_rejects_mixed_commands(self) -> None:
        grip = np.concatenate([np.full(20, -1.0), np.full(20, 1.0)])
        state = np.zeros((40, 8))
        state[:20, -2:] = 0.04
        convention = infer_gripper_convention(grip, state)
        self.assertEqual(stable_window_label([-1.0, -1.0, -1.0], convention), "open")
        self.assertIsNone(stable_window_label([-1.0, 1.0], convention))
        self.assertIsNone(stable_window_label([0.0, 0.0], convention))


class PairingTests(unittest.TestCase):
    def test_pairs_stay_inside_task_and_progress_and_episodes_do_not_leak(self) -> None:
        records = [
            _record(0, episode=0, task=1, frame=10, label="open"),
            _record(1, episode=1, task=1, frame=12, label="close"),
            _record(2, episode=2, task=2, frame=10, label="open"),
            _record(3, episode=3, task=2, frame=11, label="close"),
            _record(4, episode=4, task=1, frame=80, label="open"),
            _record(5, episode=5, task=1, frame=10, label="close", split="holdout"),
        ]
        train, holdout = split_episode_ids([0, 1, 2, 3, 4, 5], train_fraction=0.7, seed=0)
        self.assertFalse(train & holdout)
        self.assertEqual(len(train) + len(holdout), 6)
        assigned = assign_splits(records, train, holdout)
        self.assertTrue(all(record.split == "train" for record in assigned if record.episode_id in train))
        self.assertTrue(all(record.split == "holdout" for record in assigned if record.episode_id in holdout))
        pairs = pair_records(records[:5], seed=0)
        self.assertGreaterEqual(len(pairs), 1)
        for open_record, close_record in pairs:
            self.assertEqual(open_record.label, "open")
            self.assertEqual(close_record.label, "close")
            self.assertEqual(open_record.task_id, close_record.task_id)
            self.assertEqual(open_record.progress_bin, close_record.progress_bin)
            self.assertEqual(open_record.split, "train")
        self.assertTrue(all(record.split == "train" for pair in pairs for record in pair))
        capped = cap_records(records, max_per_group=1, max_holdout_per_label=1, seed=0)
        self.assertLessEqual(sum(record.split == "holdout" for record in capped), 1)


class DirectionTests(unittest.TestCase):
    def test_consistent_contrast_outranks_one_spike(self) -> None:
        open_vectors = np.zeros((6, 2))
        close_vectors = np.zeros((6, 2))
        open_vectors[:, 0] = 1.0
        open_vectors[0, 1] = 100.0
        stats = cell_statistics(open_vectors, close_vectors, open_vectors, close_vectors)
        self.assertGreater(stats["score"][0], stats["score"][1])
        self.assertAlmostEqual(stats["direction"][0], 1.0)
        sparse, kept = sparsify_direction(stats["direction"], stats["sign_consistency"], top_k=1, min_consistency=0.7)
        self.assertEqual(kept.tolist(), [0])
        self.assertAlmostEqual(float(np.linalg.norm(sparse)), float(np.linalg.norm(stats["direction"])))
        shuffled = permute_direction(stats["direction"], seed=1)
        self.assertEqual(sorted(np.round(shuffled, 6)), sorted(np.round(stats["direction"], 6)))

    def test_token_mean_and_action_effect(self) -> None:
        latent = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
        averaged = mean_over_tokens(latent, [0, 2])
        self.assertEqual(averaged.shape, (4,))
        self.assertTrue(np.allclose(averaged, latent[0, [0, 2]].mean(axis=0)))

        baseline = np.zeros((5, 7))
        steered = np.zeros((5, 7))
        steered[:, 6] = 2.0
        steered[:, 0] = 0.5
        effect = action_effect(baseline, steered, horizon=5)
        self.assertAlmostEqual(effect["delta_gripper"], 2.0)
        self.assertAlmostEqual(effect["other_dims_mean_abs"], 0.5 / 6.0)
        self.assertAlmostEqual(effect["abs_dx"], 0.5)
        self.assertAlmostEqual(effect["abs_grip"], 2.0)

    def test_figures_write(self) -> None:
        grip = np.concatenate([np.full(30, -1.0), np.full(30, 1.0)])
        state = np.zeros((60, 8))
        state[:30, -2:] = 0.04
        convention = infer_gripper_convention(grip, state)
        labels = [convention.label(float(value)) for value in grip]
        ranking = [
            {
                "layer": 1,
                "tau": 0.3,
                "feature_id": 4,
                "mean_open": 1.0,
                "mean_close": 0.0,
                "delta": 1.0,
                "open_firing_frequency": 1.0,
                "close_firing_frequency": 0.0,
                "sign_consistency": 1.0,
                "score": 1.0,
            }
        ]
        steering = []
        for alpha in (0.0, 1.0):
            for control, delta in (("plus", 0.4), ("minus", -0.3), ("random", 0.05), ("baseline", 0.0)):
                if alpha == 0.0 and control != "baseline":
                    continue
                if alpha != 0.0 and control == "baseline":
                    continue
                row = {
                    "direction_kind": "baseline" if control == "baseline" else "full",
                    "control": control,
                    "alpha": alpha,
                    "delta_gripper": 0.0 if control == "baseline" else delta * alpha,
                }
                for name in ("dx", "dy", "dz", "dRx", "dRy", "dRz", "grip"):
                    row[f"abs_{name}"] = abs(delta) if name == "grip" else 0.01
                steering.append(row)
        bank = LatentBank()
        bank.add(1, 2, 0.3000004, np.ones(3))
        self.assertEqual(bank.cells(), [(2, 0.3)])
        with tempfile.TemporaryDirectory() as tmp:
            written = save_figures(
                Path(tmp),
                ranking_rows=ranking,
                steering_rows=steering,
                grip_values=grip,
                grip_labels=labels,
                convention=convention,
            )
            self.assertTrue(all(path.exists() and path.stat().st_size > 0 for path in written))


if __name__ == "__main__":
    unittest.main()

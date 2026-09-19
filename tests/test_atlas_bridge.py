"""Unit tests for Atlas concept remapping and transcoder feature I/O."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import importlib.util

_concepts_spec = importlib.util.spec_from_file_location(
    "atlas_concepts",
    SRC / "pi05_mi" / "atlas_concepts.py",
)
atlas_concepts = importlib.util.module_from_spec(_concepts_spec)
assert _concepts_spec.loader is not None
_concepts_spec.loader.exec_module(atlas_concepts)
atlas_to_lerobot_task_id = atlas_concepts.atlas_to_lerobot_task_id
get_concept_task_mapping = atlas_concepts.get_concept_task_mapping
task_id_from_prompt = atlas_concepts.task_id_from_prompt

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class AtlasConceptTests(unittest.TestCase):
    def test_cookie_box_remap(self) -> None:
        mapping = get_concept_task_mapping("libero_spatial", space="lerobot")
        self.assertEqual(sorted(mapping["object"]["cookie_box"]["tasks"]), [3, 6])
        self.assertEqual(atlas_to_lerobot_task_id("libero_spatial", 3), 6)
        self.assertEqual(atlas_to_lerobot_task_id("libero_spatial", 6), 3)
        self.assertEqual(
            task_id_from_prompt(
                "libero_spatial",
                "pick up the black bowl on the cookie box and place it on the plate",
            ),
            3,
        )

    def test_object_suite_remap(self) -> None:
        mapping = get_concept_task_mapping("libero_object", space="lerobot")
        self.assertEqual(mapping["object"]["cream_cheese"]["tasks"], [1])
        self.assertEqual(mapping["object"]["bbq_sauce"]["tasks"], [3])


@unittest.skipIf(torch is None, "torch is not installed in this interpreter")
class SparseFeatureTests(unittest.TestCase):
    def test_layer_names(self) -> None:
        from pi05_mi.atlas_bridge import expert_mlp_layer_name, parse_layer_name

        self.assertEqual(expert_mlp_layer_name(7), "expert_mlp_L07")
        self.assertEqual(parse_layer_name("expert_mlp_L07"), 7)

    def test_roundtrip(self) -> None:
        from pi05_mi.atlas_bridge import (
            flatten_token_rows,
            load_task_features,
            pack_sparse_features,
            per_token_topk,
            save_sparse_features,
            sparse_to_dense,
        )

        latent = torch.tensor([[0.0, 3.0, 1.0, 0.0], [2.0, 0.0, 0.0, 4.0]])
        values, indices = per_token_topk(latent, k=2)
        token_idx, token_val, token_t = flatten_token_rows(indices, values, torch.tensor([0.7]))
        packed = pack_sparse_features(
            layer_index=7,
            d_features=4,
            k=2,
            indices=token_idx,
            values=token_val,
            timesteps=token_t,
        )
        dense = sparse_to_dense(packed)
        self.assertEqual(tuple(dense.shape), (2, 4))
        self.assertEqual(dense[0, 1].item(), 3.0)
        self.assertEqual(dense[1, 3].item(), 4.0)
        self.assertEqual(dense[0, 0].item(), 0.0)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "task3" / "ep0" / "expert_mlp_L07.pt"
            save_sparse_features(path, packed)
            loaded = load_task_features(root, "expert_mlp_L07")
            self.assertIn(3, loaded)
            self.assertEqual(tuple(loaded[3].shape), (2, 4))

    def test_concept_scores_need_two_tasks(self) -> None:
        from pi05_mi.atlas_bridge import compute_concept_scores

        task_features = {
            3: torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            0: torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        }
        results = compute_concept_scores(task_features, "libero_spatial", top_k=2)
        cookie = results["object"]["cookie_box"]["top_features"][0]
        self.assertEqual(cookie["feature_idx"], 0)
        self.assertGreater(cookie["score"], 0.0)


if __name__ == "__main__":
    unittest.main()

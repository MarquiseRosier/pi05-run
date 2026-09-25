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
        self.assertEqual(results["object"]["cookie_box"]["tasks_in"], [3])
        self.assertEqual(results["object"]["cookie_box"]["tasks_out"], [0])


@unittest.skipIf(torch is None, "torch is not installed in this interpreter")
class AdditiveSteerTests(unittest.TestCase):
    def _dictionary(self):
        from pi05_mi.atlas_bridge import TranscoderDictionary
        from pi05_mi.transcoders import TimeConditionedTranscoder, TimeConditionedTranscoderConfig

        config = TimeConditionedTranscoderConfig(d_model=4, d_features=3, time_embedding_dim=4, time_hidden_dim=4)
        transcoder = TimeConditionedTranscoder(config)
        with torch.no_grad():
            transcoder.decoder.weight.copy_(torch.eye(4, 3))
            transcoder.decoder.bias.zero_()
            transcoder.encoder.bias.fill_(1.0)
        transcoder.eval()
        return TranscoderDictionary(transcoder)

    def test_additive_delta_changes_decode_without_replacing_multiplicative_steer(self) -> None:
        dictionary = self._dictionary()
        x = torch.zeros(1, 2, 4)
        timestep = torch.tensor([0.4])
        y_hat, y_same, latent = dictionary.intervene(x, timestep)
        self.assertTrue(torch.allclose(y_hat, y_same))

        delta = torch.tensor([1.0, 0.0, 0.0])
        _y_hat, steered, returned = dictionary.intervene(x, timestep, latent_delta=delta, latent_delta_scale=2.0)
        self.assertTrue(torch.allclose(returned, latent))
        self.assertTrue(torch.allclose(steered, dictionary.decode(latent + 2.0 * delta)))

        _y_hat, scaled, latent_scaled = dictionary.intervene(x, timestep, steer_features=[0], steer_strength=1.0)
        modified = latent_scaled.clone()
        modified[..., 0] = modified[..., 0] * 2.0
        self.assertTrue(torch.allclose(scaled, dictionary.decode(modified)))

        _y_hat, ablated, latent_ablated = dictionary.intervene(x, timestep, ablate_features=[1])
        modified = latent_ablated.clone()
        modified[..., 1] = 0
        self.assertTrue(torch.allclose(ablated, dictionary.decode(modified)))

    def test_context_gates_layer_and_tau_without_disabling_feature_edits(self) -> None:
        from torch import nn

        from pi05_mi.patch_pi05 import Pi05TranscoderContext, WrappedActionExpertMLP
        from pi05_mi.transcoders import TimeConditionedTranscoder, TimeConditionedTranscoderConfig

        context = Pi05TranscoderContext(mode="probe", capture_records=False, capture_latents=False)
        context.set_intervention(steer_features=[0], steer_strength=0.5, ablate_layers=[1])
        self.assertTrue(context.wants_intervention(1))
        self.assertFalse(context.wants_intervention(0))

        context.set_intervention()
        context.set_latent_delta(torch.ones(3), scale=1.0, layers=[3], timesteps=[0.3])
        self.assertFalse(context.wants_intervention(3))
        with context.use_timestep(torch.tensor(0.9)):
            self.assertFalse(context.wants_intervention(3))
        with context.use_timestep(torch.tensor(0.3004)):
            self.assertTrue(context.wants_intervention(3))
            self.assertFalse(context.wants_intervention(2))
        context.set_latent_delta(torch.ones(3), scale=0.0, layers=[3], timesteps=[0.3])
        with context.use_timestep(torch.tensor(0.3)):
            self.assertFalse(context.wants_intervention(3))

        class TinyMLP(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(4, 4)

            def forward(self, value):
                return self.linear(value)

        transcoder = TimeConditionedTranscoder(
            TimeConditionedTranscoderConfig(d_model=4, d_features=3, time_embedding_dim=4, time_hidden_dim=4)
        )
        with torch.no_grad():
            transcoder.decoder.weight.copy_(torch.eye(4, 3))
            transcoder.decoder.bias.zero_()
            transcoder.encoder.bias.fill_(1.0)
        wrapped = WrappedActionExpertMLP(
            name="expert.mlp",
            layer_index=3,
            original_mlp=TinyMLP(),
            context=context,
            transcoder=transcoder,
        )
        context.set_latent_delta(None)
        sample = torch.randn(1, 2, 4)
        with context.use_timestep(torch.tensor([0.3])):
            baseline = wrapped(sample)
        context.set_latent_delta(torch.tensor([1.0, 0.0, 0.0]), scale=3.0, layers=[3], timesteps=[0.3])
        with context.use_timestep(torch.tensor([0.3])):
            steered = wrapped(sample)
        with context.use_timestep(torch.tensor([0.9])):
            other_tau = wrapped(sample)
        self.assertFalse(torch.allclose(baseline, steered))
        self.assertTrue(torch.allclose(baseline, other_tau))


if __name__ == "__main__":
    unittest.main()

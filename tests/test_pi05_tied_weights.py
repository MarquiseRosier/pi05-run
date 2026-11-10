"""Exercise the actual pinned PaliGemma loader with tiny CPU weights."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pi05_mi.pi05_weights import assert_weights_loaded, install_load_recorder


def test_real_paligemma_safetensors_roundtrip(tmp_path):
    pytest.importorskip('transformers.models.siglip.check')
    from transformers import PaliGemmaConfig, PaliGemmaForConditionalGeneration
    from safetensors.torch import load_file, save_model

    config = PaliGemmaConfig(
        vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                           num_attention_heads=2, image_size=16, patch_size=8, projection_dim=16),
        text_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=8, vocab_size=32),
        vocab_size=32, projection_dim=16, hidden_size=16, image_token_index=31,
    )
    original = PaliGemmaForConditionalGeneration(config).eval()
    assert original.model.language_model.embed_tokens.weight is original.lm_head.weight
    path = tmp_path / 'tiny.safetensors'
    # Match the real checkpoint: keep lm_head and omit the tied input embedding.
    save_model(original, path, force_contiguous=True)
    state = load_file(path)
    embedding = 'model.language_model.embed_tokens.weight'
    if embedding in state:
        state['lm_head.weight'] = state.pop(embedding)
    assert embedding not in state

    restored = PaliGemmaForConditionalGeneration(config).eval()
    install_load_recorder(type(restored))
    restored.load_state_dict(state, strict=True)
    report = assert_weights_loaded(restored)
    assert report['restored_tied_aliases'] == {embedding: 'lm_head.weight'}
    tokens = torch.tensor([[1, 2, 3]])
    with torch.inference_mode():
        expected = original(input_ids=tokens).logits
        actual = restored(input_ids=tokens).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

import pytest
import torch

from capstone.model import KVCache, ModelConfig, Transformer
from capstone.train import lm_loss


def tiny_config() -> ModelConfig:
    return ModelConfig(vocab_size=64, dim=32, n_layers=2, n_heads=2, seq_len=16)


def test_forward_shape():
    model = Transformer(tiny_config())
    tokens = torch.randint(0, 64, (3, 10))
    logits = model(tokens)
    assert logits.shape == (3, 10, 64)


def test_loss_decreases_on_repeated_batch():
    torch.manual_seed(0)
    model = Transformer(tiny_config())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    tokens = torch.randint(0, 64, (4, 17))
    first = None
    for _ in range(60):
        loss = lm_loss(model(tokens[:, :-1]), tokens)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < first * 0.7


def test_weight_tying():
    model = Transformer(tiny_config())
    assert model.lm_head.weight is model.embed.weight


def test_causality():
    model = Transformer(tiny_config()).eval()
    tokens = torch.randint(0, 64, (1, 12))
    with torch.no_grad():
        base = model(tokens)
        tokens2 = tokens.clone()
        tokens2[0, -1] = (tokens2[0, -1] + 1) % 64
        changed = model(tokens2)
    assert torch.allclose(base[0, :-1], changed[0, :-1], atol=1e-5)


def test_decoding_with_a_cache_matches_the_full_forward():
    """The invariant the cache exists to preserve. Catches a stale RoPE position
    or a mask aligned to the wrong corner, both of which still emit fluent text."""
    torch.manual_seed(0)
    model = Transformer(tiny_config()).eval()
    tokens = torch.randint(0, 64, (2, 9))
    with torch.no_grad():
        full = model(tokens)
        cache = KVCache(len(model.blocks))
        model(tokens[:, :4], cache)
        stepwise = [model(tokens[:, i : i + 1], cache) for i in range(4, 9)]
    assert len(cache) == 9
    for offset, logits in enumerate(stepwise):
        assert torch.allclose(full[:, 4 + offset], logits[:, -1], atol=1e-5)


def test_cache_length_tracks_the_tokens_seen():
    model = Transformer(tiny_config()).eval()
    cache = KVCache(len(model.blocks))
    assert len(cache) == 0
    with torch.no_grad():
        model(torch.randint(0, 64, (1, 5)), cache)
        assert len(cache) == 5
        model(torch.randint(0, 64, (1, 1)), cache)
    assert len(cache) == 6


def test_generate_extends_the_prompt():
    torch.manual_seed(0)
    model = Transformer(tiny_config())
    prompt = torch.randint(0, 64, (2, 3))
    out = model.generate(prompt, max_new_tokens=5)
    assert out.shape == (2, 8)
    assert torch.equal(out[:, :3], prompt)


def test_generate_stops_at_the_context_limit():
    model = Transformer(tiny_config()).eval()
    prompt = torch.randint(0, 64, (1, 14))
    out = model.generate(prompt, max_new_tokens=100)
    assert out.shape[1] <= tiny_config().seq_len


def test_generate_leaves_training_mode_untouched():
    model = Transformer(tiny_config()).train()
    model.generate(torch.randint(0, 64, (1, 2)), max_new_tokens=2)
    assert model.training


def test_generate_restores_training_mode_when_a_step_raises():
    model = Transformer(tiny_config()).train()
    with pytest.raises(IndexError):
        model.generate(torch.tensor([[9999]]), max_new_tokens=2)
    assert model.training


def test_greedy_generation_is_deterministic():
    model = Transformer(tiny_config()).eval()
    prompt = torch.randint(0, 64, (1, 4))
    first = model.generate(prompt, max_new_tokens=6, temperature=0.0)
    second = model.generate(prompt, max_new_tokens=6, temperature=0.0)
    assert torch.equal(first, second)

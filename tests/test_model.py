# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

import torch

from capstone.model import ModelConfig, Transformer
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

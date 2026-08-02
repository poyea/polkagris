# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

import math

import torch

from experiments._common import DetachedOps, build, frozen_parameters, one_backward, small_config
from experiments.tied_init import build_with_init, initial_loss

CPU = torch.device("cpu")


def tiny(layers=2):
    return small_config(65, dim=32, layers=layers, seq=16)


def test_detached_attention_drops_the_graph():
    q, k, v = (torch.randn(1, 2, 8, 16, requires_grad=True) for _ in range(3))
    out = DetachedOps.attention(q, k, v)
    assert out.grad_fn is None
    assert not out.requires_grad


def test_detached_attention_keeps_the_values():
    """The defect is the missing graph, not wrong numbers."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 8, 16) for _ in range(3))
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(DetachedOps.attention(q, k, v), expected)


def test_only_qkv_and_attn_norm_freeze():
    cfg = tiny(layers=2)
    model = build(cfg, detached=True)
    one_backward(model, cfg, CPU)
    names, share = frozen_parameters(model)
    assert set(names) == {
        "blocks.0.attn_norm.weight",
        "blocks.0.attn.qkv.weight",
        "blocks.1.attn_norm.weight",
        "blocks.1.attn.qkv.weight",
    }
    assert 0.1 < share < 0.4


def test_reference_freezes_nothing():
    cfg = tiny()
    model = build(cfg, detached=False)
    one_backward(model, cfg, CPU)
    names, share = frozen_parameters(model)
    assert names == []
    assert share == 0.0


def test_untied_head_does_not_share_storage():
    cfg = tiny()
    tied = build_with_init(cfg, 0.02, tied=True)
    untied = build_with_init(cfg, 0.02, tied=False)
    assert tied.lm_head.weight is tied.embed.weight
    assert untied.lm_head.weight is not untied.embed.weight


def test_tying_makes_initial_loss_track_embedding_scale():
    cfg = tiny()
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len + 1))
    uniform = math.log(cfg.vocab_size)

    tied_small = initial_loss(build_with_init(cfg, 0.02, True), tokens)
    tied_large = initial_loss(build_with_init(cfg, 1.0, True), tokens)
    untied_large = initial_loss(build_with_init(cfg, 1.0, False), tokens)

    assert tied_small < uniform * 1.5
    assert tied_large > uniform * 3
    assert untied_large < uniform * 1.5

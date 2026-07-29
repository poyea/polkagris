# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

import torch

from capstone.ops import get_ops

ops = get_ops("reference")


def test_rmsnorm_unit_scale():
    x = torch.randn(2, 5, 8)
    w = torch.ones(8)
    y = ops.rmsnorm(x, w)
    rms = y.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_rope_preserves_norm():
    q = torch.randn(2, 4, 16, 32)
    k = torch.randn(2, 4, 16, 32)
    pos = torch.arange(16)
    q2, k2 = ops.rope(q, k, pos)
    assert torch.allclose(q.norm(dim=-1), q2.norm(dim=-1), atol=1e-4)
    assert torch.allclose(k.norm(dim=-1), k2.norm(dim=-1), atol=1e-4)


def test_rope_position_zero_is_identity():
    q = torch.randn(1, 2, 1, 16)
    k = torch.randn(1, 2, 1, 16)
    q2, k2 = ops.rope(q, k, torch.zeros(1, dtype=torch.long))
    assert torch.allclose(q, q2, atol=1e-5)
    assert torch.allclose(k, k2, atol=1e-5)


def test_attention_is_causal():
    q = torch.randn(1, 1, 6, 8)
    k = torch.randn(1, 1, 6, 8)
    v = torch.randn(1, 1, 6, 8)
    out1 = ops.attention(q, k, v)
    k[..., -1, :] = 100.0
    v[..., -1, :] = 100.0
    out2 = ops.attention(q, k, v)
    assert torch.allclose(out1[..., :-1, :], out2[..., :-1, :], atol=1e-5)

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


def test_single_query_against_a_cache_sees_the_whole_cache():
    """The decode case. is_causal would align top-left and pin this to key 0."""
    q = torch.randn(1, 1, 6, 8)
    k = torch.randn(1, 1, 6, 8)
    v = torch.randn(1, 1, 6, 8)
    full = ops.attention(q, k, v)
    last = ops.attention(q[:, :, -1:], k, v)
    assert torch.allclose(full[:, :, -1:], last, atol=1e-5)


def test_a_partial_query_block_stays_causal():
    """Prefill against an existing cache: mask aligns bottom right, not top left."""
    q = torch.randn(1, 1, 6, 8)
    k = torch.randn(1, 1, 6, 8)
    v = torch.randn(1, 1, 6, 8)
    full = ops.attention(q, k, v)
    tail = ops.attention(q[:, :, 4:], k, v)
    assert torch.allclose(full[:, :, 4:], tail, atol=1e-5)


def test_rope_accepts_per_sequence_positions():
    """Batched sequences sit at different absolute positions."""
    q = torch.randn(2, 2, 1, 16)
    k = torch.randn(2, 2, 1, 16)
    batched = torch.tensor([[3], [7]])
    q_b, k_b = ops.rope(q, k, batched)
    for row, pos in enumerate((3, 7)):
        q_r, k_r = ops.rope(q[row : row + 1], k[row : row + 1], torch.tensor([pos]))
        assert torch.allclose(q_b[row : row + 1], q_r, atol=1e-5)
        assert torch.allclose(k_b[row : row + 1], k_r, atol=1e-5)


def test_key_mask_excludes_padded_slots():
    """A shorter row must not attend to the unused tail it shares with a longer one."""
    q = torch.randn(1, 1, 1, 8)
    k = torch.randn(1, 1, 5, 8)
    v = torch.randn(1, 1, 5, 8)
    mask = torch.tensor([[True, True, True, False, False]])
    masked = ops.attention(q, k, v, mask)
    trimmed = ops.attention(q, k[:, :, :3], v[:, :, :3])
    assert torch.allclose(masked, trimmed, atol=1e-5)


def test_key_mask_ignores_junk_in_the_padded_tail():
    q = torch.randn(1, 1, 1, 8)
    k = torch.randn(1, 1, 4, 8)
    v = torch.randn(1, 1, 4, 8)
    mask = torch.tensor([[True, True, False, False]])
    before = ops.attention(q, k, v, mask)
    k[:, :, 2:] = 1e4
    v[:, :, 2:] = 1e4
    assert torch.allclose(before, ops.attention(q, k, v, mask), atol=1e-5)


class StrictOps:
    """A backend that takes only q, k, v, as the triton kernel does."""

    rmsnorm = staticmethod(ops.rmsnorm)
    rope = staticmethod(ops.rope)

    @staticmethod
    def attention(q, k, v, key_mask=None, q_positions=None):
        if key_mask is not None or q_positions is not None:
            raise NotImplementedError("this backend takes no mask or position operand")
        return ops.attention(q, k, v)


def test_an_uncached_forward_sends_no_mask_or_positions():
    """A backend that takes only q, k, v must still be able to run the model."""
    from capstone.model import ModelConfig, Transformer

    cfg = ModelConfig(vocab_size=32, dim=16, n_layers=2, n_heads=2, seq_len=8)
    model = Transformer(cfg)
    for block in model.blocks:
        block.attn.ops = StrictOps()
    model(torch.randint(0, 32, (1, 6)))


def test_an_uncached_forward_keeps_the_fused_causal_path():
    """The fused path is what is_causal buys; an explicit mask gives it up."""
    from capstone.model import ModelConfig, Transformer

    seen = []
    real = ops.attention

    def watching(q, k, v, key_mask=None, q_positions=None):
        seen.append(key_mask is None and q_positions is None and q.shape[-2] == k.shape[-2])
        return real(q, k, v, key_mask, q_positions)

    cfg = ModelConfig(vocab_size=32, dim=16, n_layers=2, n_heads=2, seq_len=8)
    model = Transformer(cfg)
    for block in model.blocks:
        block.attn.ops = type("W", (), {"rmsnorm": staticmethod(ops.rmsnorm),
                                        "rope": staticmethod(ops.rope),
                                        "attention": staticmethod(watching)})()
    model(torch.randint(0, 32, (1, 6)))
    assert seen and all(seen), "the model gave up the fused causal path"

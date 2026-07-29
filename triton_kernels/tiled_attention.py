# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

from triton_kernels._compat import require_triton

triton, tl = require_triton()

import torch
import torch.nn.functional as F

from polkagris import benchmark, set_seed


@triton.jit
def attention_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    seq_len, scale,
    stride_qh, stride_qm,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    m_block = tl.program_id(0)
    head = tl.program_id(1)

    offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_base = q_ptr + head * stride_qh
    q = tl.load(
        q_base + offs_m[:, None] * stride_qm + offs_d[None, :],
        mask=offs_m[:, None] < seq_len, other=0.0,
    )

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    for n_start in range(0, seq_len, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        k_base = k_ptr + head * stride_qh
        v_base = v_ptr + head * stride_qh
        k = tl.load(
            k_base + offs_n[:, None] * stride_qm + offs_d[None, :],
            mask=offs_n[:, None] < seq_len, other=0.0,
        )
        v = tl.load(
            v_base + offs_n[:, None] * stride_qm + offs_d[None, :],
            mask=offs_n[:, None] < seq_len, other=0.0,
        )
        scores = tl.dot(q, tl.trans(k)) * scale
        scores = tl.where(
            (offs_m[:, None] >= offs_n[None, :]) & (offs_n[None, :] < seq_len),
            scores, -float("inf"),
        )
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    o_base = o_ptr + head * stride_qh
    tl.store(
        o_base + offs_m[:, None] * stride_qm + offs_d[None, :],
        acc.to(o_ptr.dtype.element_ty),
        mask=offs_m[:, None] < seq_len,
    )


def tiled_attention(q, k, v, block_m=64, block_n=64):
    heads, seq_len, head_dim = q.shape
    o = torch.empty_like(q)
    scale = head_dim**-0.5
    grid = (triton.cdiv(seq_len, block_m), heads)
    attention_kernel[grid](
        q, k, v, o, seq_len, scale,
        q.stride(0), q.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, HEAD_DIM=head_dim,
    )
    return o


def main(heads: int = 16, seq_len: int = 2048, head_dim: int = 64) -> None:
    set_seed()
    shape = (heads, seq_len, head_dim)
    q, k, v = (torch.randn(shape, device="cuda", dtype=torch.float16) for _ in range(3))

    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    got = tiled_attention(q, k, v)
    torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)

    t = benchmark(lambda: tiled_attention(q, k, v), "triton_attention_fp16")
    r = benchmark(
        lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True), "sdpa_fp16"
    )
    print(f"triton {t.mean_ms:.3f} ms  sdpa {r.mean_ms:.3f} ms  ({r.mean_ms / t.mean_ms:.2f}x)")


if __name__ == "__main__":
    main()

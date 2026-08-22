# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

import torch
import torch.nn.functional as F


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def rope(q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor, theta: float = 10000.0):
    head_dim = q.shape[-1]
    half = head_dim // 2
    freqs = theta ** (-torch.arange(half, device=q.device, dtype=torch.float32) / half)
    angles = positions.to(torch.float32)[..., None] * freqs
    if angles.ndim == 2:
        cos, sin = angles.cos()[None, None], angles.sin()[None, None]
    else:
        # Positions carry a batch dimension. Sequences batched together sit at
        # different absolute positions, so each row rotates by its own angle.
        cos, sin = angles.cos()[:, None], angles.sin()[:, None]

    def rotate(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).to(x.dtype)

    return rotate(q), rotate(k)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Causal attention. `key_mask` is [batch, t_k], True where a key is real.

    Batched sequences of different lengths share one cache tensor, so the unused
    tail of a shorter row has to be masked out rather than attended to.
    """
    t_q, t_k = q.shape[-2], k.shape[-2]
    if key_mask is None and t_q == t_k:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    # A cache makes k longer than q, and `is_causal` aligns its mask to the top
    # left, which would pin a decode step to position 0. Align bottom right: the
    # i-th query is at absolute position t_k - t_q + i.
    offset = t_k - t_q
    q_pos = torch.arange(t_q, device=q.device)[:, None] + offset
    k_pos = torch.arange(t_k, device=q.device)[None, :]
    mask = k_pos <= q_pos
    if key_mask is not None:
        mask = mask[None, None] & key_mask[:, None, None, :]
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

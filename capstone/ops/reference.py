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
    q_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Causal attention.

    `key_mask` is [batch, t_k], True where a key is real. `q_positions` is the
    absolute position of each query; give it whenever the keys do not end at the
    query, which is what a preallocated cache with a padded tail looks like.
    """
    t_q, t_k = q.shape[-2], k.shape[-2]
    if key_mask is None and q_positions is None and t_q == t_k:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    k_index = torch.arange(t_k, device=q.device)
    if q_positions is None:
        # Nothing says where the queries sit, so assume they are the tail of k.
        # `is_causal` would instead align to the top left and pin a decode step
        # to position 0.
        q_index = torch.arange(t_q, device=q.device)[:, None] + (t_k - t_q)
        mask = (k_index[None, :] <= q_index)[None, None]
    else:
        # A key at buffer index j holds the token at absolute position j, so the
        # query's own position is what decides which keys it may read.
        positions = q_positions if q_positions.ndim == 2 else q_positions[None]
        mask = (k_index <= positions[..., None])[:, None]
    if key_mask is not None:
        mask = mask & key_mask[:, None, None, :]
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

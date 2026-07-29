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
    angles = positions.to(torch.float32)[:, None] * freqs[None, :]
    cos = angles.cos()[None, None]
    sin = angles.sin()[None, None]

    def rotate(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).to(x.dtype)

    return rotate(q), rotate(k)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)

# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from capstone.ops import get_ops


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    seq_len: int = 1024
    ops: str = "reference"


class RMSNorm(nn.Module):
    def __init__(self, dim: int, ops):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.ops = ops

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ops.rmsnorm(x, self.weight)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, ops):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.ops = ops

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (
            z.view(b, t, self.n_heads, self.head_dim).transpose(1, 2) for z in (q, k, v)
        )
        q, k = self.ops.rope(q, k, positions)
        out = self.ops.attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(b, t, -1))


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, ops):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, ops)
        self.attn = Attention(cfg, ops)
        self.mlp_norm = RMSNorm(cfg.dim, ops)
        self.mlp = SwiGLU(cfg.dim, 4 * cfg.dim * 2 // 3)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), positions)
        return x + self.mlp(self.mlp_norm(x))


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        ops = get_ops(cfg.ops)
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg, ops) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim, ops)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, t = tokens.shape
        positions = torch.arange(t, device=tokens.device)
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x, positions)
        return self.lm_head(self.norm(x))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

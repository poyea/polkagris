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


class KVCache:
    """Post-RoPE keys and values per layer.

    Keys are stored rotated, so each token is rotated once at its own absolute
    position and never re-rotated as the context grows.
    """

    def __init__(self, n_layers: int):
        self.entries: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * n_layers

    def __len__(self) -> int:
        first = self.entries[0]
        return 0 if first is None else first[0].shape[-2]


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

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        past: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        b, t, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (
            z.view(b, t, self.n_heads, self.head_dim).transpose(1, 2) for z in (q, k, v)
        )
        q, k = self.ops.rope(q, k, positions)
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        out = self.ops.attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(b, t, -1)), (k, v)


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

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        past: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_out, present = self.attn(self.attn_norm(x), positions, past)
        x = x + attn_out
        return x + self.mlp(self.mlp_norm(x)), present


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

    def forward(self, tokens: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        _, t = tokens.shape
        start = len(cache) if cache is not None else 0
        positions = torch.arange(start, start + t, device=tokens.device)
        x = self.embed(tokens)
        for i, block in enumerate(self.blocks):
            past = cache.entries[i] if cache is not None else None
            x, present = block(x, positions, past)
            if cache is not None:
                cache.entries[i] = present
        return self.lm_head(self.norm(x))

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Sample continuations, one forward pass per new token.

        The prompt is one prefill pass over all its tokens; each step after it
        is a decode pass over a single token against the cache.
        """
        was_training = self.training
        self.eval()
        cache = KVCache(len(self.blocks))
        step = tokens
        out = tokens
        for _ in range(max_new_tokens):
            # Cap on what is returned, not on what is cached: the token a step
            # samples is appended without being fed back, so checking the cache
            # would let the sequence end one past the context.
            if out.shape[1] >= self.cfg.seq_len:
                break
            logits = self(step, cache)[:, -1]
            if temperature == 0.0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = logits.topk(min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                nxt = torch.multinomial(logits.softmax(dim=-1), num_samples=1)
            out = torch.cat([out, nxt], dim=1)
            step = nxt
        if was_training:
            self.train()
        return out

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

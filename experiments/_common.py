# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

"""Shared rig: a small char-level transformer and pooled scoring."""

from __future__ import annotations

import math
import statistics as st

import torch

from capstone.data import CharCorpus
from capstone.model import ModelConfig, Transformer
from capstone.ops import reference
from capstone.train import lm_loss
from polkagris import set_seed

BATCH = 16
LR = 3e-4


class DetachedOps:
    """reference, with an attention that returns a tensor carrying no grad_fn.

    A kernel writing its output into a fresh `torch.empty_like` produces exactly
    this: correct values, no autograd graph. Same shape of defect as the triton
    attention path, reproducible without triton.
    """

    rmsnorm = staticmethod(reference.rmsnorm)
    rope = staticmethod(reference.rope)

    @staticmethod
    def attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_mask: torch.Tensor | None = None,
        q_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return reference.attention(q, k, v, key_mask, q_positions).detach()


def small_config(vocab_size: int, dim: int = 128, layers: int = 4, seq: int = 128) -> ModelConfig:
    return ModelConfig(
        vocab_size=vocab_size, dim=dim, n_layers=layers, n_heads=4, seq_len=seq
    )


def build(cfg: ModelConfig, seed: int = 1859, detached: bool = False) -> Transformer:
    set_seed(seed)
    model = Transformer(cfg)
    if detached:
        for block in model.blocks:
            block.attn.ops = DetachedOps
    return model


def frozen_parameters(model: Transformer) -> tuple[list[str], float]:
    """Names with no gradient, and their share of all parameters. Needs a backward."""
    dead_names, dead, total = [], 0, 0
    for name, p in model.named_parameters():
        total += p.numel()
        if p.grad is None or not p.grad.any():
            dead_names.append(name)
            dead += p.numel()
    return dead_names, dead / total


def one_backward(model: Transformer, cfg: ModelConfig, device: torch.device) -> None:
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len + 1), device=device)
    lm_loss(model(tokens[:, :-1]), tokens).backward()


@torch.no_grad()
def pooled_ppl(
    model: Transformer, corpus: CharCorpus, cfg: ModelConfig,
    device: torch.device, split: str, seed: int, batches: int = 24,
) -> float:
    """Perplexity over many batches, which is the number worth reporting."""
    was_training = model.training
    model.eval()
    gen = corpus.batches(split, cfg.seq_len, BATCH, seed=seed)
    vals = []
    for _ in range(batches):
        tokens = next(gen).to(device)
        vals.append(lm_loss(model(tokens[:, :-1]), tokens).item())
    model.train(was_training)
    return math.exp(st.mean(vals))


def train_arm(
    model: Transformer, corpus: CharCorpus, cfg: ModelConfig, device: torch.device,
    steps: int, seed: int, marks: tuple[int, ...] = (),
) -> tuple[dict[int, float], list[float]]:
    """Train, recording a 20-step trailing mean at each mark. Returns (marks, losses)."""
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    gen = corpus.batches("train", cfg.seq_len, BATCH, seed=seed)
    losses: list[float] = []
    recorded: dict[int, float] = {}
    for step in range(1, steps + 1):
        tokens = next(gen).to(device)
        loss = lm_loss(model(tokens[:, :-1]), tokens)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step in marks:
            recorded[step] = st.mean(losses[-20:])
    return recorded, losses

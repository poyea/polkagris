# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

"""Run: python -m experiments.checks

The falsifiable half of each experiment in this directory, sized to run in
seconds. The full tables come from the modules themselves.
"""

from __future__ import annotations

import math
import statistics as st

import torch

from capstone.data import CharCorpus
from capstone.train import lm_loss
from experiments._common import (
    BATCH,
    LR,
    build,
    frozen_parameters,
    one_backward,
    pooled_ppl,
    small_config,
)
from experiments.tied_init import STDS, build_with_init, initial_loss
from polkagris.checks import run
from polkagris.data import get_device

TINY_VOCAB = 65
LAYERS = 4


def a_graph_dropping_backend_freezes_only_attention_inputs() -> str:
    cfg = small_config(TINY_VOCAB, dim=64, layers=LAYERS, seq=32)
    model = build(cfg, detached=True)
    one_backward(model, cfg, torch.device("cpu"))
    names, share = frozen_parameters(model)

    expected = {f"blocks.{i}.attn.qkv.weight" for i in range(LAYERS)} | {
        f"blocks.{i}.attn_norm.weight" for i in range(LAYERS)
    }
    assert set(names) == expected, f"frozen set moved: {sorted(set(names) ^ expected)}"
    assert 0.15 < share < 0.35, f"frozen share {share:.3f} outside the measured band"
    return (
        f"{len(names)} tensors, {share * 100:.1f}% of parameters, exactly qkv and attn_norm "
        "per layer; the residual path keeps the other 75% training"
    )


def the_reference_backend_freezes_nothing() -> str:
    cfg = small_config(TINY_VOCAB, dim=64, layers=LAYERS, seq=32)
    model = build(cfg, detached=False)
    one_backward(model, cfg, torch.device("cpu"))
    names, share = frozen_parameters(model)
    assert not names, f"reference dropped gradients for {names}"
    return f"all {len(list(model.parameters()))} tensors receive gradient, {share * 100:.0f}% frozen"


def untying_the_head_makes_initial_loss_ignore_embedding_scale() -> str:
    cfg = small_config(TINY_VOCAB, dim=64, layers=2, seq=32)
    uniform = math.log(cfg.vocab_size)
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len + 1))

    tied = [initial_loss(build_with_init(cfg, s, True), tokens) for s in STDS]
    untied = [initial_loss(build_with_init(cfg, s, False), tokens) for s in STDS]

    band = uniform * 1.5
    assert all(u < band for u in untied), "untied initial loss should not depend on embedding std"
    assert max(tied) > band, "tied init did not blow up anywhere in the sweep"
    usable = max(s for s, t in zip(STDS, tied) if t < band)
    return (
        f"untied stays {min(untied):.2f} to {max(untied):.2f} across a "
        f"{max(STDS) / min(STDS):.0f}x std sweep; tied reaches {max(tied):.1f} "
        f"and is only usable to std {usable}"
    )


def one_training_batch_cannot_resolve_the_generalization_gap() -> str:
    device = get_device()
    corpus = CharCorpus()
    cfg = small_config(corpus.vocab_size, dim=64, layers=2, seq=64)
    model = build(cfg).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    gen = corpus.batches("train", cfg.seq_len, BATCH, seed=0)

    singles = []
    for _ in range(120):
        tokens = next(gen).to(device)
        loss = lm_loss(model(tokens[:, :-1]), tokens)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        singles.append(math.exp(loss.item()))

    tail = singles[-40:]
    spread = max(tail) - min(tail)
    gap = abs(
        pooled_ppl(model, corpus, cfg, device, "val", 1234, batches=12)
        - pooled_ppl(model, corpus, cfg, device, "train", 99, batches=12)
    )
    assert spread > gap, (
        f"single-batch spread {spread:.3f} no longer exceeds the pooled gap {gap:.3f}; "
        "the estimator claim needs remeasuring"
    )
    return (
        f"single-batch band {spread:.3f} ppl against a pooled gap of {gap:.3f}, "
        f"{spread / gap:.1f}x, sd {st.pstdev(tail):.3f}"
    )


CHECKS = [
    a_graph_dropping_backend_freezes_only_attention_inputs,
    the_reference_backend_freezes_nothing,
    untying_the_head_makes_initial_loss_ignore_embedding_scale,
    one_training_batch_cannot_resolve_the_generalization_gap,
]

if __name__ == "__main__":
    raise SystemExit(run("experiments", CHECKS))

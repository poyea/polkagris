# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

"""Run: python -m experiments.tied_init

Question: tied embeddings save parameters, but what do they cost?

Answer: the embedding matrix has to serve two roles at once, lookup table and
output projection, and those want different scales. Untied, the embedding std
does not affect initial loss at all across a 500x sweep. Tied, it sets it: the
usable band collapses to std <= 0.05, and recovering from a bad init spends a
third of a 300-step budget getting back to where a good init starts.
"""

from __future__ import annotations

import argparse
import math

import torch
from torch import nn

from capstone.data import CharCorpus
from capstone.model import ModelConfig, Transformer
from capstone.train import lm_loss
from experiments._common import BATCH, LR, small_config
from polkagris import set_seed
from polkagris.data import get_device

STDS = (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
SANE_STD = 0.02


def build_with_init(cfg: ModelConfig, std: float, tied: bool, seed: int = 1859) -> Transformer:
    set_seed(seed)
    model = Transformer(cfg)
    if not tied:
        # give lm_head its own tensor, held at the sane std, so only embed varies
        model.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        nn.init.normal_(model.lm_head.weight, mean=0.0, std=SANE_STD)
    nn.init.normal_(model.embed.weight, mean=0.0, std=std)
    return model


@torch.no_grad()
def initial_loss(model: Transformer, tokens: torch.Tensor) -> float:
    model.eval()
    return lm_loss(model(tokens[:, :-1]), tokens).item()


def steps_to_recover(
    cfg: ModelConfig, std: float, target: float, device: torch.device,
    corpus: CharCorpus, max_steps: int,
) -> int | None:
    model = build_with_init(cfg, std, tied=True).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    gen = corpus.batches("train", cfg.seq_len, BATCH, seed=0)
    for step in range(1, max_steps + 1):
        tokens = next(gen).to(device)
        loss = lm_loss(model(tokens[:, :-1]), tokens)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if loss.item() <= target:
            return step
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--dim", type=int, default=128)
    args = parser.parse_args()

    device = get_device()
    corpus = CharCorpus()
    cfg = small_config(corpus.vocab_size, dim=args.dim)
    uniform = math.log(cfg.vocab_size)
    tokens = next(corpus.batches("train", cfg.seq_len, BATCH, seed=0)).to(device)

    print(f"vocab {cfg.vocab_size}, ln(vocab) {uniform:.3f}, the loss of a uniform predictor")
    print(f"\n{'embed std':>10} | {'tied':>10} | {'untied':>10} | {'tied/uniform':>13}")
    print("-" * 52)
    rows = []
    for std in STDS:
        tied = initial_loss(build_with_init(cfg, std, True).to(device), tokens)
        untied = initial_loss(build_with_init(cfg, std, False).to(device), tokens)
        rows.append((std, tied, untied))
        print(f"{std:>10.3f} | {tied:>10.3f} | {untied:>10.3f} | {tied / uniform:>12.1f}x")

    band = uniform * 1.5
    tied_ok = [s for s, t, _ in rows if t < band]
    untied_ok = [s for s, _, u in rows if u < band]
    print(f"\nwithin 1.5x of ln(vocab): tied {len(tied_ok)}/{len(STDS)}, "
          f"untied {len(untied_ok)}/{len(STDS)}")
    print(f"tied usable band: std <= {max(tied_ok)}")

    print(f"\nsteps for a tied model to reach {uniform:.2f}, where a good init starts:")
    for std in (0.02, 0.2, 1.0):
        n = steps_to_recover(cfg, std, uniform, device, corpus, args.max_steps)
        print(f"  std {std:>5}: {n if n is not None else f'>{args.max_steps}'} steps")


if __name__ == "__main__":
    main()

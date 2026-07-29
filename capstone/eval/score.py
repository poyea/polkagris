# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

import argparse
import math
from pathlib import Path

import torch

from capstone.model import ModelConfig, Transformer
from capstone.train import lm_loss, synthetic_batches
from polkagris import set_seed
from polkagris.data import get_device

HELDOUT_SEED = 1234


@torch.no_grad()
def perplexity(model: Transformer, device: torch.device, batches: int = 8) -> float:
    model.eval()
    cfg = model.cfg
    gen = synthetic_batches(cfg.vocab_size, cfg.seq_len, 8, seed=HELDOUT_SEED)
    total = 0.0
    for _ in range(batches):
        tokens = next(gen)[:, : cfg.seq_len + 1].to(device)
        total += lm_loss(model(tokens[:, :-1]), tokens).item()
    return math.exp(min(20.0, total / batches))


@torch.no_grad()
def val_perplexity(model: Transformer, device: torch.device, batches: int = 8) -> float:
    from capstone.data import CharCorpus

    model.eval()
    cfg = model.cfg
    corpus = CharCorpus()
    gen = corpus.batches("val", cfg.seq_len, 8, seed=HELDOUT_SEED)
    total = 0.0
    for _ in range(batches):
        tokens = next(gen).to(device)
        total += lm_loss(model(tokens[:, :-1]), tokens).item()
    return math.exp(min(20.0, total / batches))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--data", default="synthetic", choices=["synthetic", "shakespeare"])
    args = parser.parse_args()

    set_seed()
    device = get_device()
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model = Transformer(ModelConfig(**state["config"])).to(device)
        model.load_state_dict(state["model"])
    else:
        model = Transformer(
            ModelConfig(vocab_size=1024, dim=256, n_layers=4, n_heads=4, seq_len=128)
        ).to(device)
        print("no checkpoint given, scoring a random-init model")
    if args.data == "shakespeare":
        print(f"held-out perplexity (shakespeare val): {val_perplexity(model, device):.1f}")
    else:
        print(f"held-out perplexity (synthetic): {perplexity(model, device):.1f}")


if __name__ == "__main__":
    main()

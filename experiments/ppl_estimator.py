# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

"""Run: python -m experiments.ppl_estimator

Question: the training loop prints a perplexity every step. What is it worth?

Answer: less than it looks. At fixed model quality the single-batch number
wanders over a band wider than the train/val gap it would be used to estimate,
so one reading cannot resolve generalization at all. Taking the luckiest batch
as a held-out figure understates by roughly 15%, which is how this repo's
capstone perplexity was recorded as 6.4 when the pooled value was 7.8.
"""

from __future__ import annotations

import argparse
import math
import statistics as st

import torch

from capstone.data import CharCorpus
from capstone.train import lm_loss
from experiments._common import BATCH, LR, build, pooled_ppl, small_config
from polkagris.data import get_device

TRAIN_PROBE_SEED = 99
VAL_SEED = 1234


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--dim", type=int, default=128)
    args = parser.parse_args()

    device = get_device()
    corpus = CharCorpus()
    cfg = small_config(corpus.vocab_size, dim=args.dim)
    model = build(cfg).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    gen = corpus.batches("train", cfg.seq_len, BATCH, seed=0)

    print(f"device {device}, {args.steps} steps")
    print(f"\n{'step':>6} | {'batch ppl':>10} | {'pooled train':>13} | {'pooled val':>11}")
    print("-" * 50)
    singles: list[float] = []
    for step in range(1, args.steps + 1):
        tokens = next(gen).to(device)
        loss = lm_loss(model(tokens[:, :-1]), tokens)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        singles.append(math.exp(loss.item()))
        if step % 50 == 0:
            train_ppl = pooled_ppl(model, corpus, cfg, device, "train", TRAIN_PROBE_SEED)
            val_ppl = pooled_ppl(model, corpus, cfg, device, "val", VAL_SEED)
            print(f"{step:>6} | {singles[-1]:>10.3f} | {train_ppl:>13.3f} | {val_ppl:>11.3f}")

    tail = singles[-args.window :]
    train_ppl = pooled_ppl(model, corpus, cfg, device, "train", TRAIN_PROBE_SEED)
    val_ppl = pooled_ppl(model, corpus, cfg, device, "val", VAL_SEED)
    spread = max(tail) - min(tail)
    gap = val_ppl - train_ppl

    print(f"\nover the last {args.window} steps, at near-fixed model quality:")
    print(f"  single-batch ppl  min {min(tail):.3f}  max {max(tail):.3f}  sd {st.pstdev(tail):.3f}")
    print(f"  pooled train ppl  {train_ppl:.3f}")
    print(f"  pooled val   ppl  {val_ppl:.3f}")
    print(f"\n  generalization gap (pooled val - pooled train)  {gap:>+8.3f}")
    print(f"  noise band of one training batch                {spread:>8.3f}")
    print(f"  noise/signal                                    {spread / abs(gap):>7.1f}x")
    print(
        f"\nreporting the luckiest batch ({min(tail):.3f}) as held-out understates the pooled "
        f"{val_ppl:.3f} by {100 * (val_ppl - min(tail)) / val_ppl:.0f}%"
    )


if __name__ == "__main__":
    main()

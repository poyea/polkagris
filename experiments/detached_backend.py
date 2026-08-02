# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

"""Run: python -m experiments.detached_backend

Question: if an op backend drops the autograd graph, does the loss curve show it?

Answer: not early, and not in the direction you would expect. The residual path
routes gradient around the broken attention, so 24.8% of parameters freeze while
the remaining 75% keep training. For the first ~50 steps the broken arm sits
BELOW the correct one, crosses over near step 100, and is only clearly worse by
step 200.
"""

from __future__ import annotations

import argparse
import statistics as st

from capstone.data import CharCorpus
from experiments._common import build, frozen_parameters, pooled_ppl, small_config, train_arm
from polkagris.data import get_device

MARKS = (25, 50, 100, 200, 300)
SEEDS = (1859, 7, 20260802)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    args = parser.parse_args()

    device = get_device()
    corpus = CharCorpus()
    cfg = small_config(corpus.vocab_size, dim=args.dim, layers=args.layers)
    marks = tuple(m for m in MARKS if m <= args.steps)

    results: dict[str, list[tuple[dict[int, float], float]]] = {"reference": [], "detached": []}
    frozen_share, frozen_names = 0.0, []
    for seed in args.seeds:
        for label, detached in (("reference", False), ("detached", True)):
            model = build(cfg, seed=seed, detached=detached)
            recorded, _ = train_arm(model, corpus, cfg, device, args.steps, seed, marks)
            ppl = pooled_ppl(model, corpus, cfg, device, "val", seed=1234, batches=8)
            results[label].append((recorded, ppl))
            if detached:
                frozen_names, frozen_share = frozen_parameters(model)

    print(f"device {device}, {len(args.seeds)} seeds x 2 arms, {args.steps} steps")
    print(f"\n{'step':>6} | {'reference':>18} | {'detached':>18} | {'gap':>8}")
    print("-" * 62)
    for mark in marks:
        ref = [r[0][mark] for r in results["reference"]]
        det = [r[0][mark] for r in results["detached"]]
        rm, dm = st.mean(ref), st.mean(det)
        print(
            f"{mark:>6} | {rm:>9.4f} +-{st.pstdev(ref):<7.4f} | "
            f"{dm:>9.4f} +-{st.pstdev(det):<7.4f} | {dm - rm:>+8.4f}"
        )

    rp = [r[1] for r in results["reference"]]
    dp = [r[1] for r in results["detached"]]
    print(
        f"\nheld-out ppl: reference {st.mean(rp):.2f} +-{st.pstdev(rp):.2f}, "
        f"detached {st.mean(dp):.2f} +-{st.pstdev(dp):.2f} ({st.mean(dp) / st.mean(rp):.2f}x)"
    )
    print(f"frozen: {frozen_share * 100:.1f}% of parameters, {len(frozen_names)} tensors")
    print(f"frozen tensors: {', '.join(frozen_names[:2])}, ... (qkv and attn_norm per layer)")


if __name__ == "__main__":
    main()

# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

"""Run: python -m training.checks"""

from __future__ import annotations

from dataclasses import replace
from itertools import pairwise

import torch
from torch import nn

from polkagris import set_seed
from polkagris.checks import Pending, run
from training.config import TrainConfig
from training.loop import lr_at, make_optimizer

CFG = TrainConfig()
TOTAL = 1000


def warmup_ramps_from_almost_zero_to_the_peak() -> str:
    cfg = replace(CFG, warmup_steps=100)
    first = lr_at(0, TOTAL, cfg)
    peak = lr_at(99, TOTAL, cfg)
    assert first < peak, f"{first} !< {peak}"
    assert abs(peak - cfg.lr) < 1e-12, f"peak {peak} != lr {cfg.lr}"
    return f"step 0 lr {first:.2e} rises to {peak:.2e} at the end of warmup"


def cosine_decays_monotonically_after_warmup() -> str:
    cfg = replace(CFG, warmup_steps=100)
    lrs = [lr_at(s, TOTAL, cfg) for s in range(100, TOTAL)]
    assert all(b <= a + 1e-12 for a, b in pairwise(lrs)), "cosine is not monotonic"
    assert lrs[-1] < lrs[0] / 100, f"final lr {lrs[-1]:.2e} is not near zero"
    return f"lr falls {lrs[0]:.2e} to {lrs[-1]:.2e} without a single step up"


def a_constant_schedule_ignores_the_cosine() -> str:
    cfg = replace(CFG, warmup_steps=10, lr_schedule="constant")
    after = {lr_at(s, TOTAL, cfg) for s in range(10, TOTAL)}
    assert after == {cfg.lr}, f"expected one value, got {len(after)}"
    return f"every post-warmup step sits at {cfg.lr:.2e}"


def weight_decay_reaches_the_optimizer() -> str:
    model = nn.Linear(4, 2)
    for name in ("sgd", "adam", "adamw"):
        opt = make_optimizer(model, replace(CFG, optimizer=name, weight_decay=0.123))
        got = opt.param_groups[0]["weight_decay"]
        assert got == 0.123, f"{name} dropped weight_decay: {got}"
    return "sgd, adam and adamw all carry weight_decay 0.123"


def adamw_decouples_decay_and_adam_does_not() -> str:
    def drift(optimizer: str) -> float:
        set_seed()
        p = nn.Parameter(torch.ones(1) * 5.0)
        opt = {
            "adam": torch.optim.Adam([p], lr=0.1, weight_decay=0.5),
            "adamw": torch.optim.AdamW([p], lr=0.1, weight_decay=0.5),
        }[optimizer]
        for _ in range(20):
            opt.zero_grad()
            p.grad = torch.zeros_like(p)  # no data signal, decay only
            opt.step()
        return p.item()

    adam, adamw = drift("adam"), drift("adamw")
    assert adamw < adam, f"adamw {adamw} should shrink faster than adam {adam}"
    return f"with zero gradient, adamw pulls to {adamw:.3f} vs adam {adam:.3f}"


def a_bare_loop_learns_a_line() -> str:
    set_seed()
    x = torch.randn(256, 1)
    y = 3.0 * x - 1.0
    model = nn.Linear(1, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    first = last = 0.0
    for step in range(200):
        loss = nn.functional.mse_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    assert last < first / 100, f"loss only moved {first:.4f} to {last:.4f}"
    return f"loss {first:.4f} to {last:.6f}; weight {model.weight.item():.3f} vs true 3.0"


def zero_grad_is_not_optional() -> str:
    set_seed()
    model = nn.Linear(1, 1)
    x, y = torch.randn(8, 1), torch.randn(8, 1)
    nn.functional.mse_loss(model(x), y).backward()
    once = model.weight.grad.clone()
    nn.functional.mse_loss(model(x), y).backward()
    twice = model.weight.grad.clone()
    assert torch.allclose(twice, once * 2), "gradients did not accumulate"
    return "a second backward without zero_grad doubles the gradient"


def halved_learning_rate_effect() -> str:
    raise Pending("effect of halving cfg.lr on the epoch-14 plateau is unmeasured")


def augmentation_effect_on_generalisation() -> str:
    raise Pending("train/test gap without augmentation is unmeasured")


CHECKS = [
    warmup_ramps_from_almost_zero_to_the_peak,
    cosine_decays_monotonically_after_warmup,
    a_constant_schedule_ignores_the_cosine,
    weight_decay_reaches_the_optimizer,
    adamw_decouples_decay_and_adam_does_not,
    a_bare_loop_learns_a_line,
    zero_grad_is_not_optional,
    halved_learning_rate_effect,
    augmentation_effect_on_generalisation,
]

if __name__ == "__main__":
    raise SystemExit(run("training", CHECKS))

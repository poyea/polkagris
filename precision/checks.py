# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

"""Run: python -m precision.checks"""

from __future__ import annotations

import torch
from torch import nn

from polkagris import set_seed
from polkagris.checks import Pending, Skip, run


def fp16_overflows_where_bf16_does_not() -> str:
    big = 70000.0
    half = torch.tensor(big, dtype=torch.float16)
    brain = torch.tensor(big, dtype=torch.bfloat16)
    assert half.isinf(), f"expected fp16 overflow, got {half}"
    assert brain.isfinite(), f"expected bf16 to hold {big}, got {brain}"
    return f"{big:.0f} is inf in fp16 but {brain.item():.0f} in bf16"


def bf16_trades_mantissa_for_exponent() -> str:
    half, brain = torch.finfo(torch.float16), torch.finfo(torch.bfloat16)
    assert brain.eps > half.eps, "bf16 should be the coarser of the two"
    assert brain.max > half.max, "bf16 should reach further"
    return (
        f"eps {brain.eps:.1e} vs fp16 {half.eps:.1e}; "
        f"max {brain.max:.1e} vs fp16 {half.max:.1e}"
    )


def bf16_has_the_same_range_as_fp32() -> str:
    brain, single = torch.finfo(torch.bfloat16), torch.finfo(torch.float32)
    ratio = brain.max / single.max
    assert 0.9 < ratio < 1.1, f"expected matching range, ratio {ratio}"
    return "bf16 is fp32 with the mantissa cut, which is why it needs no loss scaling"


def small_gradients_vanish_in_fp16() -> str:
    tiny = 1e-8
    assert torch.tensor(tiny, dtype=torch.float16).item() == 0.0, "expected fp16 underflow"
    scaled = torch.tensor(tiny * 2**16, dtype=torch.float16)
    assert scaled.item() > 0.0, "scaling failed to rescue the value"
    return f"{tiny:.0e} underflows to 0 in fp16; times 2^16 it survives. That is GradScaler"


def autocast_picks_dtypes_per_operation() -> str:
    set_seed()
    layer = nn.Linear(64, 64)
    x = torch.randn(8, 64)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        matmul = layer(x)
        summed = matmul.float().sum()
    assert matmul.dtype == torch.bfloat16, f"matmul ran in {matmul.dtype}"
    assert summed.dtype == torch.float32, f"reduction ran in {summed.dtype}"
    return f"linear went to {matmul.dtype}, the reduction stayed {summed.dtype}"


def sequential_accumulation_stalls_in_fp16() -> str:
    n = 4096
    acc = torch.tensor(0.0, dtype=torch.float16)
    one = torch.tensor(1.0, dtype=torch.float16)
    for _ in range(n):
        acc = acc + one
    tree = torch.full((n,), 1.0, dtype=torch.float16).sum(dtype=torch.float16)
    assert acc.item() < n, f"expected a stall below {n}, got {acc.item()}"
    assert tree.item() == n, f"expected the pairwise sum to be exact, got {tree.item()}"
    return (
        f"adding 1.0 {n} times in fp16 stops at {acc.item():.0f} (spacing exceeds 1 there), "
        f"while torch.sum's pairwise reduction still gets {tree.item():.0f}"
    )


def a_gpu_is_needed_for_the_speed_story() -> str:
    if not torch.cuda.is_available():
        raise Skip("no CUDA device")
    name = torch.cuda.get_device_name(0)
    major = torch.cuda.get_device_capability(0)[0]
    return f"{name} (sm_{major}x): tensor cores from sm_70, so check what you are on"


def ops_autocast_keeps_in_fp32() -> str:
    raise Pending("the fp32 keep-list is not enumerated here")


def accuracy_under_bf16() -> str:
    raise Pending("final accuracy in fp32 versus bf16 is unmeasured")


CHECKS = [
    fp16_overflows_where_bf16_does_not,
    bf16_trades_mantissa_for_exponent,
    bf16_has_the_same_range_as_fp32,
    small_gradients_vanish_in_fp16,
    sequential_accumulation_stalls_in_fp16,
    autocast_picks_dtypes_per_operation,
    a_gpu_is_needed_for_the_speed_story,
    ops_autocast_keeps_in_fp32,
    accuracy_under_bf16,
]

if __name__ == "__main__":
    raise SystemExit(run("precision", CHECKS))

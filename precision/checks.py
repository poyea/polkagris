# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

"""Run: python -m precision.checks"""

from __future__ import annotations

import torch
from torch import nn

from polkagris import set_seed
from polkagris.checks import Skip, run


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
    major, minor = torch.cuda.get_device_capability(0)
    return f"{name}: compute capability {major}.{minor}"


def ops_autocast_keeps_in_fp32() -> str:
    # Feeding each op both dtypes is what separates the three policies. An op
    # that returns fp32 from fp32 input proves nothing: it may be on the keep
    # list, or it may just be following its input. Only bf16 in / fp32 out is
    # a genuine cast back up. The lists are registered per device type, so this
    # enumerates the CPU policy; CUDA registers its own, longer fp32 list.
    def probe(dtype: torch.dtype) -> dict[str, torch.dtype]:
        x = torch.randn(8, 64, dtype=dtype)
        w = torch.randn(64, 64, dtype=dtype)
        target = torch.zeros(8, dtype=torch.long)
        ops = {
            "linear": lambda: torch.nn.functional.linear(x, w),
            "matmul": lambda: x @ w,
            "softmax": lambda: torch.softmax(x, -1),
            "layer_norm": lambda: torch.nn.functional.layer_norm(x, (64,)),
            "exp": lambda: x.exp(),
            "sum": lambda: x.sum(),
            "prod": lambda: x.prod(),
            "cross_entropy": lambda: torch.nn.functional.cross_entropy(x, target),
            "mse_loss": lambda: torch.nn.functional.mse_loss(x, x),
            "relu": lambda: torch.relu(x),
        }
        with torch.autocast("cpu", dtype=torch.bfloat16):
            return {name: fn().dtype for name, fn in ops.items()}

    from_fp32, from_bf16 = probe(torch.float32), probe(torch.bfloat16)
    widened = [k for k, v in from_fp32.items() if v is torch.bfloat16]
    kept = [k for k, v in from_bf16.items() if v is torch.float32]
    followed = [k for k in from_fp32 if k not in widened and k not in kept]

    assert "linear" in widened, f"autocast is not active: linear returned {from_fp32['linear']}"
    assert kept, "expected at least one op to cast back up to fp32"
    assert "relu" in followed, f"relu should follow its input, got {from_bf16['relu']}"
    return (
        f"cast down: {', '.join(widened)}; kept in fp32 even from bf16 input: "
        f"{', '.join(kept)}; the other {len(followed)} just follow their input "
        f"({', '.join(followed)}); on CPU the keep-list is this short"
    )


def accuracy_under_bf16() -> str:
    """Whether the halved mantissa costs accuracy, against run-to-run spread.

    One run of each proves nothing here: the seed moves the final number by
    more than the dtype does, and a single pair came out with bf16 ahead. So
    train both at several seeds and compare the gap to the spread. Six short
    runs, so this is the slow check in this file.
    """
    import contextlib
    import io
    import statistics
    from dataclasses import replace

    from torch.utils.data import DataLoader, Subset

    # download=False: a check measures, it does not fetch 64 MB behind your back.
    from torchvision import datasets, transforms

    from polkagris.data import DATA_ROOT, get_device
    from training.config import TrainConfig
    from training.loop import fit
    from training.mnist_mlp import build_model

    cfg = TrainConfig()
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    try:
        train = datasets.MNIST(DATA_ROOT, train=True, download=False, transform=tf)
        test = datasets.MNIST(DATA_ROOT, train=False, download=False, transform=tf)
    except RuntimeError as exc:
        raise Skip(f"MNIST is not in {DATA_ROOT}; run -m training.mnist_mlp once") from exc

    device = get_device()
    indices = torch.randperm(len(train), generator=torch.Generator().manual_seed(0))[:12000]
    subset = Subset(train, indices.tolist())
    held_out = DataLoader(Subset(test, list(range(2000))), batch_size=512)
    seeds = (1859, 2, 3)

    def final_accuracy(dtype: torch.dtype | None, seed: int) -> float:
        torch.manual_seed(seed)
        model = build_model()
        loader = DataLoader(
            subset,
            batch_size=cfg.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )
        with contextlib.redirect_stdout(io.StringIO()):
            return fit(model, loader, held_out, replace(cfg, epochs=4), device, dtype)

    runs = {
        name: [final_accuracy(dtype, seed) for seed in seeds]
        for name, dtype in (("fp32", None), ("bf16", torch.bfloat16))
    }
    spread = max(max(rows) - min(rows) for rows in runs.values())
    gap = statistics.fmean(runs["fp32"]) - statistics.fmean(runs["bf16"])
    assert abs(gap) < spread, (
        f"a {gap:+.2%} gap stands clear of a {spread:.2%} spread, so the dtype is doing it"
    )
    return (
        f"fp32 {statistics.fmean(runs['fp32']):.2%} against bf16 "
        f"{statistics.fmean(runs['bf16']):.2%} over {len(seeds)} seeds: a {abs(gap):.2%} gap "
        f"inside a {spread:.2%} seed-to-seed spread, so the mantissa is not what moves it"
    )


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

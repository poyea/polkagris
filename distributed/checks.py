# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

"""Run: python -m distributed.checks"""

from __future__ import annotations

import torch
from torch import nn

from polkagris import set_seed
from polkagris.checks import Pending, Skip, run


def averaging_shard_gradients_equals_one_big_batch() -> str:
    set_seed()
    model = nn.Linear(4, 1)
    x, y = torch.randn(8, 4), torch.randn(8, 1)

    model.zero_grad()
    nn.functional.mse_loss(model(x), y).backward()
    whole = model.weight.grad.clone()

    shards = []
    for piece in (slice(0, 4), slice(4, 8)):
        model.zero_grad()
        nn.functional.mse_loss(model(x[piece]), y[piece]).backward()
        shards.append(model.weight.grad.clone())
    averaged = (shards[0] + shards[1]) / 2

    assert torch.allclose(whole, averaged, atol=1e-6), f"{whole} != {averaged}"
    return "the mean of two half-batch gradients is the full-batch gradient, which is why DDP averages"


def summing_instead_of_averaging_scales_the_learning_rate() -> str:
    set_seed()
    model = nn.Linear(4, 1)
    x, y = torch.randn(8, 4), torch.randn(8, 1)
    grads = []
    for piece in (slice(0, 4), slice(4, 8)):
        model.zero_grad()
        nn.functional.mse_loss(model(x[piece]), y[piece]).backward()
        grads.append(model.weight.grad.clone())
    summed, averaged = grads[0] + grads[1], (grads[0] + grads[1]) / 2
    ratio = (summed.norm() / averaged.norm()).item()
    assert abs(ratio - 2.0) < 1e-5, f"expected 2x, got {ratio}"
    return f"an allreduce that sums is {ratio:.0f}x the step size; forget the divide and you get divergence"


def gradient_accumulation_imitates_a_bigger_batch() -> str:
    set_seed()
    model = nn.Linear(4, 1)
    x, y = torch.randn(8, 4), torch.randn(8, 1)

    model.zero_grad()
    nn.functional.mse_loss(model(x), y).backward()
    whole = model.weight.grad.clone()

    model.zero_grad()
    for piece in (slice(0, 4), slice(4, 8)):
        (nn.functional.mse_loss(model(x[piece]), y[piece]) / 2).backward()
    accumulated = model.weight.grad.clone()

    assert torch.allclose(whole, accumulated, atol=1e-6), "accumulation drifted from the big batch"
    return "two half batches with the loss divided by two match one full batch, no extra memory"


def a_process_group_of_one_still_checks_the_wrapping() -> str:
    if not torch.distributed.is_available():
        raise Skip("this torch build has no distributed support")
    backends = [b for b in ("gloo", "nccl") if getattr(torch.distributed, f"is_{b}_available")()]
    assert backends, "no usable backend found"

    # Build the group and wrap a module, rather than reporting that a backend
    # exists and calling that a verified run.
    import tempfile
    from pathlib import Path

    from torch.nn.parallel import DistributedDataParallel

    backend = "nccl" if torch.cuda.is_available() and torch.distributed.is_nccl_available() else "gloo"
    store = Path(tempfile.mkdtemp()) / "checks_store"
    torch.distributed.init_process_group(
        backend=backend, init_method=f"file:///{store.as_posix()}", rank=0, world_size=1
    )
    try:
        model = DistributedDataParallel(nn.Linear(4, 2))
        model(torch.randn(3, 4)).sum().backward()
        assert all(p.grad is not None for p in model.parameters()), "no gradient survived the wrap"
    finally:
        torch.distributed.destroy_process_group()
    return f"backends available: {', '.join(backends)}; wrapped and stepped a module over {backend}"


def real_scaling_needs_a_second_device() -> str:
    count = torch.cuda.device_count()
    if count < 2:
        raise Skip(f"{count} CUDA device(s), need two")
    return f"{count} devices present, so scaling numbers are on the table"


def communication_overlaps_compute() -> str:
    raise Pending("the allreduce fraction hidden behind backward is unmeasured")


CHECKS = [
    averaging_shard_gradients_equals_one_big_batch,
    summing_instead_of_averaging_scales_the_learning_rate,
    gradient_accumulation_imitates_a_bigger_batch,
    a_process_group_of_one_still_checks_the_wrapping,
    real_scaling_needs_a_second_device,
    communication_overlaps_compute,
]

if __name__ == "__main__":
    raise SystemExit(run("distributed", CHECKS))

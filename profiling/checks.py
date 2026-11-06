# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

"""Run: python -m profiling.checks"""

from __future__ import annotations

import time

import torch
from torch import nn

from polkagris import set_seed
from polkagris.checks import Skip, run


def intensity(flops: float, bytes_moved: float) -> float:
    return flops / bytes_moved


def vector_add_is_memory_bound_by_arithmetic_alone() -> str:
    n = 1 << 20
    # two loads and one store per element, one add
    ai = intensity(n, 3 * n * 4)
    assert ai < 0.1, f"expected a tiny ratio, got {ai}"
    return f"{ai:.3f} FLOP/byte: nothing to compute per byte moved, so DRAM sets the speed"


def matmul_intensity_grows_with_size() -> str:
    def ai(n: int) -> float:
        return intensity(2.0 * n**3, 3.0 * n * n * 4)

    small, large = ai(128), ai(4096)
    assert large > small * 20, f"expected growth, got {small:.1f} then {large:.1f}"
    return f"{small:.1f} FLOP/byte at n=128 rises to {large:.1f} at n=4096; tiling is what captures it"


def the_naive_matmul_throws_its_reuse_away() -> str:
    n = 1024
    ideal = intensity(2.0 * n**3, 3.0 * n * n * 4)
    # one thread per output re-reads a full row and column from global memory
    naive = intensity(2.0 * n**3, 2.0 * n**3 * 4)
    assert naive < 1.0 < ideal, f"naive {naive}, ideal {ideal}"
    return f"naive reads give {naive:.2f} FLOP/byte against a possible {ideal:.0f}: that gap is phase 07"


def the_profiler_names_the_dominant_operator() -> str:
    set_seed()
    model = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.Flatten(), nn.Linear(16 * 8 * 8, 10))
    x = torch.randn(8, 3, 8, 8)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        model(x).sum().backward()
    events = [e for e in prof.key_averages() if e.self_cpu_time_total > 0]
    assert events, "the profiler recorded nothing"
    top = max(events, key=lambda e: e.self_cpu_time_total)
    return f"{len(events)} ops recorded, heaviest self time is {top.key}"


def the_first_call_pays_for_setup() -> str:
    set_seed()
    x = torch.randn(512, 512)

    def timed() -> float:
        t0 = time.perf_counter()
        x @ x
        return (time.perf_counter() - t0) * 1e3

    first = timed()
    rest = [timed() for _ in range(5)]
    assert min(rest) > 0.0, "timing resolution is too coarse to see anything"
    return f"first {first:.2f} ms, best of five after {min(rest):.2f} ms: always warm up before timing"


def a_gpu_trace_needs_a_gpu() -> str:
    if not torch.cuda.is_available():
        raise Skip("no CUDA device")
    return f"{torch.cuda.get_device_name(0)} present, so CUDA columns will populate"


def top_kernels_classified() -> str:
    """Label the step's heaviest operators against this device's two roofs.

    Nsight Compute would read this off counters, but counter access is
    privileged and an unprivileged pass exits with ERR_NVGPUCTRPERM before
    reaching a kernel. The labels do not need counters: measure the two roofs
    on the device, count each operator's traffic and arithmetic from its
    shapes, and the ridge point between them decides which roof it is under.
    """
    if not torch.cuda.is_available():
        raise Skip("the roofs belong to a device, so there is nothing to measure against")

    def median_ms(fn, warmup: int = 5, reps: int = 20) -> float:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        for i in range(reps):
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()
        times = sorted(a.elapsed_time(b) for a, b in zip(starts, ends))
        return times[len(times) // 2]

    # Roof one: a copy does two words of traffic per element and no arithmetic,
    # so it measures bandwidth and nothing else.
    count = 1 << 24
    source = torch.empty(count, device="cuda")
    target = torch.empty(count, device="cuda")
    byte_rate = (2 * count * 4) / (median_ms(lambda: target.copy_(source)) / 1e3)

    # Roof two: a square matmul large enough to be tiled measures the fp32 rate.
    side = 2048
    left = torch.randn(side, side, device="cuda")
    right = torch.randn(side, side, device="cuda")
    flop_rate = (2.0 * side**3) / (median_ms(lambda: left @ right) / 1e3)
    ridge = flop_rate / byte_rate

    # The CIFAR step's convolutions, which is where its time goes.
    batch = 128
    layers = (
        (3, 64, 32),
        (64, 64, 32),
        (64, 128, 16),
        (128, 128, 16),
        (128, 256, 8),
        (256, 256, 8),
    )
    rows = []
    for fan_in, fan_out, side_px in layers:
        conv = nn.Conv2d(fan_in, fan_out, 3, padding=1, bias=False).cuda()
        feature = torch.randn(batch, fan_in, side_px, side_px, device="cuda")
        elapsed = median_ms(lambda c=conv, f=feature: c(f)) / 1e3
        pixels = side_px * side_px
        flops = 2.0 * batch * fan_out * fan_in * 9 * pixels
        moved = 4.0 * (batch * fan_in * pixels + batch * fan_out * pixels + fan_out * fan_in * 9)
        rows.append(
            (f"{fan_in}->{fan_out}@{side_px}", elapsed, intensity(flops, moved), flops, moved)
        )
    rows.sort(key=lambda row: -row[1])

    under_memory = [row for row in rows if row[2] < ridge]
    under_compute = [row for row in rows if row[2] >= ridge]
    assert under_memory and under_compute, f"expected both roofs in one step, got {rows}"

    # The one memory-bound layer should be near the bandwidth roof, which is
    # what makes the label mean something rather than just sorting a ratio.
    thin = under_memory[0]
    achieved = thin[4] / thin[1] / byte_rate
    assert achieved > 0.4, f"a memory-bound layer at {achieved:.0%} of the roof is not bound by it"

    # Credited against direct convolution. A rate past the matmul roof is not
    # the card going faster than itself, it is cuDNN doing algebraically fewer
    # multiplies than the 3x3 definition, which is what its winograd kernels are.
    fastest = max(row[3] / row[1] for row in under_compute)
    beyond = " and one past the fp32 roof, so cuDNN is not convolving directly" if (
        fastest > flop_rate
    ) else ""
    return (
        f"{byte_rate / 1e9:.0f} GB/s and {flop_rate / 1e12:.2f} TFLOP/s give a "
        f"{ridge:.0f} FLOP/byte ridge; of the step's heaviest layers "
        f"{len(under_compute)} sit above it and {thin[0]} sits below at {thin[2]:.0f}, "
        f"at {achieved:.0%} of the bandwidth roof{beyond}"
    )


CHECKS = [
    vector_add_is_memory_bound_by_arithmetic_alone,
    matmul_intensity_grows_with_size,
    the_naive_matmul_throws_its_reuse_away,
    the_profiler_names_the_dominant_operator,
    the_first_call_pays_for_setup,
    a_gpu_trace_needs_a_gpu,
    top_kernels_classified,
]

if __name__ == "__main__":
    raise SystemExit(run("profiling", CHECKS))

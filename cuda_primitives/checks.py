# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

"""Run: python -m cuda_primitives.checks"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from polkagris.checks import Skip, run

HERE = Path(__file__).resolve().parent


def nvcc() -> str:
    found = shutil.which("nvcc")
    if not found:
        raise Skip("nvcc is not on PATH")
    return found


def compile_sources(*sources: str) -> str:
    tool = nvcc()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "probe"
        cmd = [tool, "-arch=native", "-O2", "-x", "cu", "-o", str(out)]
        cmd += [str(HERE / s) for s in sources]
        done = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if done.returncode != 0:
            first = (done.stderr or done.stdout).strip().splitlines()
            raise Skip(f"nvcc failed: {first[0] if first else 'unknown error'}")
        return "built"


def vector_add_moves_three_words_per_add() -> str:
    n = 1 << 24
    flops, moved = n, 3 * n * 4
    ai = flops / moved
    assert ai < 0.1, f"expected a tiny ratio, got {ai}"
    return f"{ai:.3f} FLOP/byte, so the kernel can only go as fast as memory: predict bandwidth-bound"


def naive_matmul_is_compute_shaped_but_runs_memory_bound() -> str:
    n = 1024
    flops = 2.0 * n**3
    with_reuse = flops / (3.0 * n * n * 4)
    without_reuse = flops / (2.0 * n**3 * 4)
    assert without_reuse < 1.0 < with_reuse, f"{without_reuse} then {with_reuse}"
    return (
        f"one thread per output gets {without_reuse:.2f} FLOP/byte, "
        f"though the maths allows {with_reuse:.0f}: predict memory-bound anyway"
    )


def a_warp_is_thirty_two_lanes() -> str:
    block = 256
    warps = block // 32
    assert block % 32 == 0, "a block that is not a multiple of 32 wastes lanes"
    ragged = 250
    wasted = (-ragged) % 32
    return f"{block} threads is {warps} full warps; {ragged} would idle {wasted} lanes in the last one"


def the_grid_must_cover_a_ragged_tail() -> str:
    n, block = 1000, 256
    grid = (n + block - 1) // block
    assert grid * block >= n, "the grid does not cover n"
    assert (grid - 1) * block < n, "the grid is bigger than it needs to be"
    return f"{grid} blocks of {block} covers {n} with {grid*block-n} threads that must bounds-check"


def the_host_programs_compile() -> str:
    compile_sources("host_vector_add.cpp", "vector_add.cu")
    compile_sources("host_naive_matmul.cpp", "naive_matmul.cu")
    return "both host programs build; run `make run` for the timings"


def achieved_bandwidth_against_spec() -> str:
    """Run the kernel in this directory and see what fraction of the roof it gets.

    The host programs need a host compiler and this does not: NVRTC compiles
    the same source and the driver API launches it, so the prediction above
    gets checked against the kernel it is about rather than a stand-in.

    The roof is measured here too, by a copy, because a number from a spec
    sheet is not what this machine does.
    """
    import torch

    from polkagris import nvrtc

    try:
        kernel = nvrtc.load(HERE / "vector_add.cu", "vector_add")
    except nvrtc.Unavailable as exc:
        raise Skip(str(exc)) from exc

    n = 1 << 24
    block = 256
    grid = (n + block - 1) // block
    a = torch.rand(n, device="cuda")
    b = torch.rand(n, device="cuda")
    c = torch.empty(n, device="cuda")

    def launch() -> None:
        kernel.launch(grid, block, a, b, c, n)

    launch()
    torch.cuda.synchronize()
    assert torch.allclose(c, a + b), "the kernel ran but did not add"

    def median_ms(fn, reps: int = 50) -> float:
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        for i in range(reps):
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()
        times = sorted(x.elapsed_time(y) for x, y in zip(starts, ends))
        return times[len(times) // 2]

    moved = 3 * n * 4
    achieved = moved / (median_ms(launch) / 1e3) / 1e9
    # A copy is the same traffic pattern with two words per element instead of
    # three, so it stands in for the roof this kernel is trying to reach.
    spare = torch.empty(n, device="cuda")
    roof = (2 * n * 4) / (median_ms(lambda: spare.copy_(a)) / 1e3) / 1e9
    fraction = achieved / roof
    assert fraction > 0.7, f"{achieved:.0f} GB/s against a {roof:.0f} GB/s roof is not bandwidth-bound"
    return (
        f"{achieved:.0f} GB/s moving {moved / 2**20:.0f} MiB, which is {fraction:.0%} of a "
        f"{roof:.0f} GB/s copy: the prediction above holds, the kernel is at the roof"
    )


def what_the_naive_matmul_actually_costs() -> str:
    """Run the kernel the prediction above is about, and see where it lands.

    The prediction is a source-level count: one thread per output re-reads a
    full row and column, so 2n^3 words cross the boundary. That is true of the
    instructions and not of the DRAM, because the cache serves most of the
    re-reads, and the gap between the two is the point.
    """
    import torch

    from polkagris import nvrtc

    try:
        kernel = nvrtc.load(HERE / "naive_matmul.cu", "naive_matmul")
    except nvrtc.Unavailable as exc:
        raise Skip(str(exc)) from exc

    n = 1024
    torch.manual_seed(1859)
    a = torch.randn(n, n, device="cuda")
    b = torch.randn(n, n, device="cuda")
    c = torch.empty(n, n, device="cuda")
    side = 16
    grid = ((n + side - 1) // side, (n + side - 1) // side)

    def launch() -> None:
        kernel.launch(grid, (side, side), a, b, c, n, n, n)

    launch()
    torch.cuda.synchronize()
    assert torch.allclose(c, a @ b, atol=2e-3, rtol=2e-3), "the kernel ran but got the wrong C"

    def median_ms(fn, reps: int = 20) -> float:
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        for i in range(reps):
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()
        times = sorted(x.elapsed_time(y) for x, y in zip(starts, ends))
        return times[len(times) // 2]

    seconds = median_ms(launch) / 1e3
    flops = 2.0 * n**3
    source_bytes = 2.0 * n**3 * 4

    # Both roofs, measured, so the two claims below are against this device.
    span = 1 << 24
    source = torch.empty(span, device="cuda")
    target = torch.empty(span, device="cuda")
    byte_rate = (2 * span * 4) / (median_ms(lambda: target.copy_(source)) / 1e3)
    wide = 2048
    left = torch.randn(wide, wide, device="cuda")
    right = torch.randn(wide, wide, device="cuda")
    flop_rate = (2.0 * wide**3) / (median_ms(lambda: left @ right) / 1e3)

    achieved_flops = flops / seconds
    apparent = source_bytes / seconds
    assert achieved_flops < 0.25 * flop_rate, "this kernel is not supposed to reach the fp32 roof"
    # If the source-level traffic really crossed DRAM it could not exceed the
    # bandwidth roof. It does, so most of those reads never got that far.
    assert apparent > byte_rate, f"{apparent / 1e9:.0f} GB/s apparent is below the roof"
    return (
        f"{achieved_flops / flop_rate:.0%} of the fp32 roof, so not compute-bound as predicted; "
        f"but its {source_bytes / 2**30:.1f} GiB of source-level reads imply "
        f"{apparent / 1e9:.0f} GB/s against a {byte_rate / 1e9:.0f} GB/s roof, "
        f"{apparent / byte_rate:.1f}x over, so the cache absorbs the reuse the source throws away"
    )


CHECKS = [
    vector_add_moves_three_words_per_add,
    naive_matmul_is_compute_shaped_but_runs_memory_bound,
    a_warp_is_thirty_two_lanes,
    the_grid_must_cover_a_ragged_tail,
    the_host_programs_compile,
    achieved_bandwidth_against_spec,
    what_the_naive_matmul_actually_costs,
]

if __name__ == "__main__":
    raise SystemExit(run("cuda_primitives", CHECKS))

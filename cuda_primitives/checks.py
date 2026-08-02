# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

"""Run: python -m cuda_primitives.checks"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from polkagris.checks import Pending, Skip, run

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
    raise Pending("vector_add GB/s against the card spec is unmeasured")


CHECKS = [
    vector_add_moves_three_words_per_add,
    naive_matmul_is_compute_shaped_but_runs_memory_bound,
    a_warp_is_thirty_two_lanes,
    the_grid_must_cover_a_ragged_tail,
    the_host_programs_compile,
    achieved_bandwidth_against_spec,
]

if __name__ == "__main__":
    raise SystemExit(run("cuda_primitives", CHECKS))

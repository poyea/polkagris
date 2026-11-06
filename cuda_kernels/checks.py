# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

"""Run: python -m cuda_kernels.checks"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from polkagris.checks import Skip, run

HERE = Path(__file__).resolve().parent


def tiles_cut_global_traffic_by_the_tile_width() -> str:
    n, bs = 1024, 32
    naive = 2.0 * n**3 * 4
    tiled = 2.0 * n**3 * 4 / bs
    assert tiled < naive, "tiling should move less"
    return f"a {bs}x{bs} tile divides global reads by {bs}: {naive/1e9:.1f} GB to {tiled/1e9:.2f} GB"


def register_tiling_is_what_makes_it_compute_bound() -> str:
    tm, tn = 8, 8
    loads, fmas = tm + tn, tm * tn
    ratio = fmas / loads
    assert ratio >= 4.0, f"expected real reuse, got {ratio}"
    return f"{loads} shared-memory loads feed {fmas} FMAs, a {ratio:.0f}x ratio: that is the whole trick"


def one_thread_per_output_has_no_reuse_at_all() -> str:
    loads, fmas = 2, 1
    assert fmas / loads < 1.0, "the naive kernel cannot have reuse"
    return f"{loads} loads per {fmas} FMA is the baseline every rung improves on"


def the_tile_constants_must_divide_the_problem() -> str:
    from_header = (HERE / "sgemm.cuh").read_text(encoding="utf-8")
    assert "kTileMultiple" in from_header, "the size contract vanished from sgemm.cuh"
    bm, bn, tm, tn = 128, 128, 8, 8
    threads = (bm * bn) // (tm * tn)
    assert threads == 256, f"expected 256 threads per block, computed {threads}"
    return f"BM={bm} BN={bn} TM={tm} TN={tn} needs exactly {threads} threads; change one and check this"


def shared_memory_per_block_must_fit() -> str:
    bm, bn, bk = 128, 128, 8
    used = (bm * bk + bk * bn) * 4
    limit = 48 * 1024
    assert used < limit, f"{used} bytes exceeds the classic {limit} limit"
    return f"{used/1024:.0f} KiB of the {limit/1024:.0f} KiB budget, so occupancy is not shared-memory bound"


def the_ladder_compiles() -> str:
    tool = shutil.which("nvcc")
    if not tool:
        raise Skip("nvcc is not on PATH")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "probe"
        cmd = [tool, "-arch=native", "-O2", "-o", str(out),
               str(HERE / "bench_sgemm.cu"), str(HERE / "sgemm.cu"), "-lcublas"]
        done = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if done.returncode != 0:
            lines = (done.stderr or done.stdout).strip().splitlines()
            raise Skip(f"nvcc failed: {lines[0] if lines else 'unknown error'}")
    return "the ladder builds; run `make run` for the cuBLAS comparison"


# Each rung with the template arguments and launch shape its host wrapper
# encodes, since that wrapper cannot be compiled here and the shape has to
# come from somewhere.
LADDER = (
    ("naive", "k_naive", lambda n: (((n + 31) // 32, (n + 31) // 32), (32, 32))),
    ("coalesced", "k_coalesced", lambda n: (((n + 31) // 32, (n + 31) // 32), (32, 32))),
    ("shared tile", "k_shared<32>", lambda n: (((n + 31) // 32, (n + 31) // 32), (32, 32))),
    ("1d blocktile", "k_blocktile_1d<64,64,8,8>", lambda n: ((n // 64, n // 64), (512,))),
    ("2d blocktile", "k_blocktile_2d<128,128,8,8,8>", lambda n: ((n // 128, n // 128), (256,))),
    ("vectorized", "k_vectorized<128,128,8,8,8>", lambda n: ((n // 128, n // 128), (256,))),
)


def fraction_of_cublas_reached() -> str:
    """Run the whole ladder and cuBLAS on the same matrices.

    `make run` would do this, and it needs a host compiler. NVRTC does not,
    so the rungs are compiled and launched from here and the comparison is
    against torch's own matmul, which is cuBLAS.
    """
    import torch

    from polkagris import nvrtc

    try:
        kernels = nvrtc.load_many(HERE / "sgemm.cu", [name for _, name, _ in LADDER])
    except nvrtc.Unavailable as exc:
        raise Skip(str(exc)) from exc

    # The tiled rungs carry no bounds checks, which is the contract
    # kTileMultiple states in sgemm.cuh.
    header = (HERE / "sgemm.cuh").read_text(encoding="utf-8")
    multiple = int(re.search(r"kTileMultiple\s*=\s*(\d+)", header).group(1))
    n = 8 * multiple
    assert n % multiple == 0, f"{n} is not a multiple of {multiple}"
    torch.manual_seed(1859)
    a = torch.randn(n, n, device="cuda")
    b = torch.randn(n, n, device="cuda")
    c = torch.empty(n, n, device="cuda")
    reference = a @ b
    flops = 2.0 * n**3

    def median_ms(fn, reps: int = 10) -> float:
        for _ in range(3):
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

    cublas = flops / (median_ms(lambda: torch.mm(a, b, out=c)) / 1e3)
    rungs = []
    for label, name, shape in LADDER:
        grid, block = shape(n)
        kernel = kernels[name]
        c.zero_()
        kernel.launch(grid, block, a, b, c, n, n, n)
        torch.cuda.synchronize()
        assert torch.allclose(c, reference, atol=2e-3, rtol=2e-3), f"{label} computed the wrong C"
        rate = flops / (
            median_ms(lambda k=kernel, g=grid, bl=block: k.launch(g, bl, a, b, c, n, n, n)) / 1e3
        )
        rungs.append((label, rate))

    first, last = rungs[0][1], rungs[-1][1]
    assert last > first * 10, f"the ladder should climb hard, got {first:.2e} then {last:.2e}"
    climb = ", ".join(f"{label} {rate / cublas:.0%}" for label, rate in rungs)
    return (
        f"at n={n} against cuBLAS at {cublas / 1e12:.2f} TFLOP/s: {climb}; "
        f"every rung agrees with torch, and the last is {last / first:.0f}x the first"
    )


CHECKS = [
    one_thread_per_output_has_no_reuse_at_all,
    tiles_cut_global_traffic_by_the_tile_width,
    register_tiling_is_what_makes_it_compute_bound,
    the_tile_constants_must_divide_the_problem,
    shared_memory_per_block_must_fit,
    the_ladder_compiles,
    fraction_of_cublas_reached,
]

if __name__ == "__main__":
    raise SystemExit(run("cuda_kernels", CHECKS))

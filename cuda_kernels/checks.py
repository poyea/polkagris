# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

"""Run: python -m cuda_kernels.checks"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from polkagris.checks import Pending, Skip, run

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


def fraction_of_cublas_reached() -> str:
    raise Pending("the cuBLAS comparison has not been run")


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

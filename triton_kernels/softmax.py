# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

from triton_kernels._compat import require_triton

triton, tl = require_triton()

import torch

from polkagris import benchmark, set_seed


@triton.jit
def softmax_kernel(x_ptr, y_ptr, stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * stride + offs, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    y = num / tl.sum(num, axis=0)
    tl.store(y_ptr + row * stride + offs, y, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    y = torch.empty_like(x)
    block = triton.next_power_of_2(cols)
    softmax_kernel[(rows,)](x, y, x.stride(0), cols, BLOCK=block)
    return y


def main(rows: int = 4096, cols: int = 4096) -> None:
    set_seed()
    x = torch.randn(rows, cols, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(triton_softmax(x), torch.softmax(x, dim=1), atol=1e-2, rtol=1e-2)
    t = benchmark(lambda: triton_softmax(x), "triton_softmax_fp16")
    r = benchmark(lambda: torch.softmax(x, dim=1), "torch_softmax_fp16")
    print(f"triton {t.mean_ms:.3f} ms  torch {r.mean_ms:.3f} ms  ({r.mean_ms / t.mean_ms:.2f}x)")


if __name__ == "__main__":
    main()

# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

from triton_kernels._compat import require_triton

triton, tl = require_triton()

import torch

from polkagris import benchmark, set_seed


@triton.jit
def layernorm_kernel(x_ptr, w_ptr, b_ptr, y_ptr, stride, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * stride + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / n_cols
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / n_cols
    xhat = diff * tl.rsqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * stride + offs, (xhat * w + b).to(y_ptr.dtype.element_ty), mask=mask)


def layernorm_linear(x, ln_w, ln_b, lin_w, lin_b, eps=1e-5):
    rows, cols = x.shape
    y = torch.empty_like(x)
    block = triton.next_power_of_2(cols)
    layernorm_kernel[(rows,)](x, ln_w, ln_b, y, x.stride(0), cols, eps, BLOCK=block)
    return y @ lin_w.T + lin_b


def main(rows: int = 4096, dim: int = 1024, out: int = 4096) -> None:
    set_seed()
    x = torch.randn(rows, dim, device="cuda", dtype=torch.float16)
    ln = torch.nn.LayerNorm(dim, device="cuda", dtype=torch.float16)
    lin = torch.nn.Linear(dim, out, device="cuda", dtype=torch.float16)

    ref = lin(ln(x))
    got = layernorm_linear(x, ln.weight, ln.bias, lin.weight, lin.bias)
    torch.testing.assert_close(got, ref, atol=1e-2, rtol=1e-2)

    t = benchmark(
        lambda: layernorm_linear(x, ln.weight, ln.bias, lin.weight, lin.bias),
        "triton_layernorm_linear",
    )
    r = benchmark(lambda: lin(ln(x)), "torch_layernorm_linear")
    print(f"triton {t.mean_ms:.3f} ms  torch {r.mean_ms:.3f} ms  ({r.mean_ms / t.mean_ms:.2f}x)")


if __name__ == "__main__":
    main()

# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

import torch

from capstone.ops import reference

try:
    import triton  # noqa: F401

    HAVE_TRITON = torch.cuda.is_available()
except ImportError:
    HAVE_TRITON = False

if not HAVE_TRITON:
    raise ImportError("triton with a CUDA device is required for ops kind 'triton'")

from triton_kernels.tiled_attention import tiled_attention

rmsnorm = reference.rmsnorm
rope = reference.rope


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # The kernel writes into a fresh `torch.empty_like`, so its output carries no
    # grad_fn and nothing upstream of attention would receive a gradient. Under
    # autograd that trains a fraction of the model while the loss still falls, so
    # refuse rather than let a backward pass look like it worked.
    if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad):
        raise NotImplementedError(
            "triton attention is forward-only: it has no backward pass"
        )
    # The kernel derives one seq_len from q and writes into empty_like(q), so a
    # longer k would be read past its tile bounds against a mask built for q.
    if k.shape[-2] != q.shape[-2]:
        raise NotImplementedError(
            f"triton attention needs q and k the same length, got {q.shape[-2]} and {k.shape[-2]}"
        )
    b, h, t, d = q.shape
    out = tiled_attention(
        q.reshape(b * h, t, d), k.reshape(b * h, t, d), v.reshape(b * h, t, d)
    )
    return out.reshape(b, h, t, d)

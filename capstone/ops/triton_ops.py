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
    b, h, t, d = q.shape
    out = tiled_attention(
        q.reshape(b * h, t, d), k.reshape(b * h, t, d), v.reshape(b * h, t, d)
    )
    return out.reshape(b, h, t, d)

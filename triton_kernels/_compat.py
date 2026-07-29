# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

import sys


def require_triton():
    try:
        import triton
        import triton.language as tl
    except ImportError:
        print("triton is not available on this platform, skipping")
        sys.exit(0)
    import torch

    if not torch.cuda.is_available():
        print("triton kernels need a CUDA device, skipping")
        sys.exit(0)
    return triton, tl

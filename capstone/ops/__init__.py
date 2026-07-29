# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

"""Op registry: reference (torch-native, default) vs triton (own kernels).

get_ops("reference") | get_ops("triton") — the capstone A/B switch.
"""


def get_ops(kind: str = "reference"):
    if kind == "reference":
        from capstone.ops import reference as mod
    elif kind == "triton":
        from capstone.ops import triton_ops as mod
    else:
        raise ValueError(f"unknown ops kind: {kind!r}")
    return mod

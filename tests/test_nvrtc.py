# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

"""The source rewriting and launch-shape handling, which need no GPU.

`load_many` itself needs a device and a driver, so the checks in
`cuda_primitives` cover that. What is tested here is everything that decides
*what source NVRTC is given*, since getting that wrong drops a kernel or
changes what gets compiled, and both fail in ways that are hard to read off a
compiler log.
"""

import pytest

from cuda_primitives.nvrtc import _dims, _read_source


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_system_includes_are_dropped(tmp_path):
    source = write(
        tmp_path,
        "k.cu",
        "#include <cuda_runtime.h>\n#include <cmath>\n__global__ void k() {}\n",
    )
    out = _read_source(source)
    assert "#include" not in out
    assert "__global__ void k() {}" in out


def test_a_quoted_include_is_pasted_in(tmp_path):
    write(tmp_path, "shared.cuh", "#pragma once\nconstexpr int kWidth = 128;\n")
    source = write(tmp_path, "k.cu", '#include "shared.cuh"\n__global__ void k() {}\n')
    out = _read_source(source)
    assert "constexpr int kWidth = 128;" in out
    # The guard means nothing once the text is inline, and NVRTC need not see it.
    assert "#pragma once" not in out
    assert "#include" not in out


def test_a_header_that_is_not_there_is_skipped_not_fatal(tmp_path):
    source = write(tmp_path, "k.cu", '#include "absent.cuh"\n__global__ void k() {}\n')
    out = _read_source(source)
    assert "absent" not in out
    assert "__global__ void k() {}" in out


def test_includes_inside_the_pasted_header_do_not_come_with_it(tmp_path):
    write(tmp_path, "shared.cuh", '#include <vector>\n#include "deeper.cuh"\nusing T = float;\n')
    write(tmp_path, "deeper.cuh", "constexpr int kDeep = 7;\n")
    source = write(tmp_path, "k.cu", '#include "shared.cuh"\n__global__ void k() {}\n')
    out = _read_source(source)
    assert "using T = float;" in out
    assert "#include" not in out
    # One level only, which is what the kernels here need. If a header ever
    # includes another, this is the line that says so.
    assert "kDeep" not in out


def test_an_indented_include_is_still_an_include(tmp_path):
    source = write(tmp_path, "k.cu", "    #include <cstdio>\n__global__ void k() {}\n")
    assert "#include" not in _read_source(source)


def test_the_word_include_in_a_comment_survives(tmp_path):
    source = write(tmp_path, "k.cu", "// include the tail here\n__global__ void k() {}\n")
    out = _read_source(source)
    assert "// include the tail here" in out


def test_host_launch_wrappers_are_kept(tmp_path):
    """They compile, so they are not the loader's problem.

    `--device-as-default-execution-space` makes an unannotated function
    device-side, and a kernel launch from device code is legal CUDA, so the
    wrappers in these files go through as they are. An explicitly `__host__`
    one would not, and nothing here writes that.
    """
    body = (
        "__global__ void k(float* p) {}\n"
        "void launch_k(float* p) {\n"
        "    const dim3 block(32);\n"
        "    k<<<1, block>>>(p);\n"
        "}\n"
    )
    out = _read_source(write(tmp_path, "k.cu", body))
    assert "k<<<1, block>>>(p);" in out
    assert "launch_k" in out


def test_line_order_and_count_are_otherwise_untouched(tmp_path):
    body = "__global__ void a() {}\n\n__global__ void b() {}\n"
    out = _read_source(write(tmp_path, "k.cu", body))
    assert out.splitlines() == ["__global__ void a() {}", "", "__global__ void b() {}"]


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (4, (4, 1, 1)),
        ((4,), (4, 1, 1)),
        ((4, 8), (4, 8, 1)),
        ((4, 8, 2), (4, 8, 2)),
    ],
)
def test_a_launch_shape_fills_out_to_three_dimensions(given, expected):
    # A .cu file writes dim3(4, 8); the missing dimensions are 1, not 0, or
    # the launch covers nothing at all.
    assert _dims(given) == expected

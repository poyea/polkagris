# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

"""Run a .cu kernel without a host compiler.

`nvcc` needs `cl.exe` even to emit device code, because it preprocesses
through the host compiler, so on a box without MSVC the kernels here cannot
be built the usual way. NVRTC compiles the same source to PTX on its own, and
the driver API loads and launches it, both reachable through ctypes. Device
memory comes from torch tensors, so there is no allocation to manage here.

This runs the kernel, not the host program: `host_vector_add.cpp` still needs
a host compiler, and so does `make run`.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import torch

NVRTC_NAMES = ("nvrtc64_112_0.dll", "nvrtc64_120_0.dll", "libnvrtc.so", "libnvrtc.so.11.2")
DRIVER_NAMES = ("nvcuda.dll", "libcuda.so", "libcuda.so.1")


class Unavailable(RuntimeError):
    """NVRTC or the driver library could not be loaded."""


def _candidate_dirs() -> list[Path]:
    roots = [os.environ.get("CUDA_PATH"), os.environ.get("CUDA_HOME")]
    found = [Path(root) / "bin" for root in roots if root]
    found += sorted(Path(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA").glob("v*/bin"))
    found += [Path("/usr/local/cuda/lib64"), Path("/usr/lib/x86_64-linux-gnu")]
    return found


def _load(names: tuple[str, ...], search: bool) -> ctypes.CDLL:
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            pass
        if not search:
            continue
        for directory in _candidate_dirs():
            path = directory / name
            if path.exists():
                try:
                    return ctypes.CDLL(str(path))
                except OSError:
                    pass
    raise Unavailable(f"none of {', '.join(names)} could be loaded")


def _check(code: int, call: str) -> None:
    if code != 0:
        raise Unavailable(f"{call} failed with {code}")


class Kernel:
    """One compiled kernel, launched with torch tensors as its pointers."""

    def __init__(self, function, driver) -> None:
        self._function = function
        self._driver = driver

    def launch(self, grid: int, block: int, *args) -> None:
        """Pass tensors for pointers and ints for scalars, in the kernel's order."""
        held, pointers = [], []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                cell = ctypes.c_void_p(int(arg.data_ptr()))
            elif isinstance(arg, int):
                cell = ctypes.c_int(arg)
            else:
                raise TypeError(f"cannot pass {type(arg).__name__} to a kernel")
            held.append(cell)
            pointers.append(ctypes.cast(ctypes.byref(cell), ctypes.c_void_p))
        packed = (ctypes.c_void_p * len(pointers))(*pointers)
        _check(
            self._driver.cuLaunchKernel(
                self._function, grid, 1, 1, block, 1, 1, 0, None, packed, None
            ),
            "cuLaunchKernel",
        )


def load(source_path: Path | str, kernel_name: str) -> Kernel:
    """Compile one kernel out of a .cu file and return something launchable."""
    if not torch.cuda.is_available():
        raise Unavailable("no CUDA device")
    nvrtc = _load(NVRTC_NAMES, search=True)
    driver = _load(DRIVER_NAMES, search=False)

    source = Path(source_path).read_text(encoding="utf-8")
    # NVRTC compiles a string and has no include path. The kernels here take
    # nothing from the runtime header, so dropping it costs nothing.
    source = "\n".join(line for line in source.splitlines() if "#include" not in line)

    program = ctypes.c_void_p()
    _check(
        nvrtc.nvrtcCreateProgram(
            ctypes.byref(program), source.encode(), str(source_path).encode(), 0, None, None
        ),
        "nvrtcCreateProgram",
    )
    # Ask for the mangled name before compiling, which is when it is decided.
    _check(nvrtc.nvrtcAddNameExpression(program, kernel_name.encode()), "nvrtcAddNameExpression")
    major, minor = torch.cuda.get_device_capability(0)
    options = (ctypes.c_char_p * 1)(f"--gpu-architecture=compute_{major}{minor}".encode())
    if nvrtc.nvrtcCompileProgram(program, 1, options) != 0:
        size = ctypes.c_size_t()
        nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(size))
        log = ctypes.create_string_buffer(size.value)
        nvrtc.nvrtcGetProgramLog(program, log)
        raise Unavailable(f"nvrtc rejected {Path(source_path).name}: {log.value.decode().strip()}")

    lowered = ctypes.c_char_p()
    _check(
        nvrtc.nvrtcGetLoweredName(program, kernel_name.encode(), ctypes.byref(lowered)),
        "nvrtcGetLoweredName",
    )
    mangled = lowered.value
    size = ctypes.c_size_t()
    nvrtc.nvrtcGetPTXSize(program, ctypes.byref(size))
    ptx = ctypes.create_string_buffer(size.value)
    nvrtc.nvrtcGetPTX(program, ptx)

    # torch owns the primary context; touching the device makes it current.
    torch.zeros(1, device="cuda")
    module = ctypes.c_void_p()
    _check(driver.cuModuleLoadData(ctypes.byref(module), ptx), "cuModuleLoadData")
    function = ctypes.c_void_p()
    _check(driver.cuModuleGetFunction(ctypes.byref(function), module, mangled), "cuModuleGetFunction")
    return Kernel(function, driver)

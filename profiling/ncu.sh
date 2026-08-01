#!/usr/bin/env bash

# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

# Nsight Compute counters for the top kernels found in the nsys timeline.
# Pass a kernel-name regex, e.g. ./ncu.sh 'sgemm|conv'; defaults to the
# convolution GEMMs.
set -euo pipefail

KERNEL_REGEX="${1:-implicit_gemm|wgrad|dgrad}"

ncu --set full --launch-count 3 -k "regex:${KERNEL_REGEX}" \
    python -m profiling.profile_cifar

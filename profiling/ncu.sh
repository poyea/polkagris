#!/usr/bin/env bash

# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

# Nsight Compute counters for the top kernels found in the nsys timeline.
# Fill in -k regex once the top-3 kernels are known.
ncu --set full --launch-count 3 -k "REGEX_TOP_KERNEL" \
    python -m profiling.profile_cifar

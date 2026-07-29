#!/usr/bin/env bash

# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

# Nsight Systems timeline of a short CIFAR training run (WSL2/Linux).
nsys profile -o cifar_timeline --trace=cuda,nvtx,osrt \
    python -m profiling.profile_cifar

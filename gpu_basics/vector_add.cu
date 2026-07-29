// Copyright (c) 2026 John Law
// SPDX-License-Identifier: MIT

// c = a + b. Arithmetic intensity is 1 FLOP per 12 bytes, so bandwidth-bound.

#include <cuda_runtime.h>

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

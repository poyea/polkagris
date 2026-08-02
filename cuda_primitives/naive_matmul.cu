// Copyright (c) 2026 John Law
// SPDX-License-Identifier: MIT

// One thread per C[i][j], K-loop over global memory. No reuse is captured, so
// this runs memory-bound despite matmul being nominally compute-bound.

#include <cuda_runtime.h>

__global__ void naive_matmul(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k)
            acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}

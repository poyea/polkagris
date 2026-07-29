// Copyright (c) 2026 John Law
// SPDX-License-Identifier: MIT

#pragma once

// The tiled kernels carry no bounds checks; bench_sgemm enforces this.
constexpr int kTileMultiple = 128;

// C[M,N] = A[M,K] * B[K,N], row-major. One rung of the ladder each.
void launch_naive(const float* A, const float* B, float* C, int M, int N, int K);
void launch_coalesced(const float* A, const float* B, float* C, int M, int N, int K);
void launch_shared(const float* A, const float* B, float* C, int M, int N, int K);
void launch_blocktile_1d(const float* A, const float* B, float* C, int M, int N, int K);
void launch_blocktile_2d(const float* A, const float* B, float* C, int M, int N, int K);
void launch_vectorized(const float* A, const float* B, float* C, int M, int N, int K);

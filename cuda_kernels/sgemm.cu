// Copyright (c) 2026 John Law
// SPDX-License-Identifier: MIT

#include "sgemm.cuh"

__global__ void k_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}

__global__ void k_coalesced(const float* A, const float* B, float* C, int M, int N, int K) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}

template <int BS>
__global__ void k_shared(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[BS][BS];
    __shared__ float Bs[BS][BS];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * BS + ty;
    const int col = blockIdx.x * BS + tx;

    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += BS) {
        As[ty][tx] = (row < M && k0 + tx < K) ? A[row * K + k0 + tx] : 0.0f;
        Bs[ty][tx] = (k0 + ty < K && col < N) ? B[(k0 + ty) * N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < BS; ++k) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

template <int BM, int BN, int BK, int TM>
__global__ void k_blocktile_1d(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    const int tid = threadIdx.x;

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    const int threadCol = tid % BN;
    const int threadRow = tid / BN;
    const int innerColA = tid % BK, innerRowA = tid / BK;
    const int innerColB = tid % BN, innerRowB = tid / BN;

    float res[TM] = {0.0f};
    for (int bk = 0; bk < K; bk += BK) {
        As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
        __syncthreads();
        A += BK;
        B += BK * N;
        for (int d = 0; d < BK; ++d) {
            const float tmpB = Bs[d * BN + threadCol];
            for (int i = 0; i < TM; ++i)
                res[i] += As[(threadRow * TM + i) * BK + d] * tmpB;
        }
        __syncthreads();
    }
    for (int i = 0; i < TM; ++i) C[(threadRow * TM + i) * N + threadCol] = res[i];
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void k_blocktile_2d(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    constexpr int kThreads = (BM * BN) / (TM * TN);
    constexpr int strideA = kThreads / BK;
    constexpr int strideB = kThreads / BN;
    const int tid = threadIdx.x;

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    const int threadCol = tid % (BN / TN);
    const int threadRow = tid / (BN / TN);
    const int innerColA = tid % BK, innerRowA = tid / BK;
    const int innerColB = tid % BN, innerRowB = tid / BN;

    float acc[TM][TN] = {0.0f};
    float regM[TM], regN[TN];

    for (int bk = 0; bk < K; bk += BK) {
        for (int o = 0; o < BM; o += strideA)
            As[(innerRowA + o) * BK + innerColA] = A[(innerRowA + o) * K + innerColA];
        for (int o = 0; o < BK; o += strideB)
            Bs[(innerRowB + o) * BN + innerColB] = B[(innerRowB + o) * N + innerColB];
        __syncthreads();
        A += BK;
        B += BK * N;
        for (int d = 0; d < BK; ++d) {
            for (int i = 0; i < TM; ++i) regM[i] = As[(threadRow * TM + i) * BK + d];
            for (int j = 0; j < TN; ++j) regN[j] = Bs[d * BN + threadCol * TN + j];
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j) acc[i][j] += regM[i] * regN[j];
        }
        __syncthreads();
    }
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
            C[(threadRow * TM + i) * N + threadCol * TN + j] = acc[i][j];
}

// As is stored transposed so the regM read below is contiguous.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void k_vectorized(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[BK * BM];
    __shared__ float Bs[BK * BN];
    const int tid = threadIdx.x;

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    const int threadCol = tid % (BN / TN);
    const int threadRow = tid / (BN / TN);
    const int innerColA4 = tid % (BK / 4), innerRowA = tid / (BK / 4);
    const int innerColB4 = tid % (BN / 4), innerRowB = tid / (BN / 4);

    float acc[TM][TN] = {0.0f};
    float regM[TM], regN[TN];

    for (int bk = 0; bk < K; bk += BK) {
        const float4 a = *reinterpret_cast<const float4*>(&A[innerRowA * K + innerColA4 * 4]);
        As[(innerColA4 * 4 + 0) * BM + innerRowA] = a.x;
        As[(innerColA4 * 4 + 1) * BM + innerRowA] = a.y;
        As[(innerColA4 * 4 + 2) * BM + innerRowA] = a.z;
        As[(innerColA4 * 4 + 3) * BM + innerRowA] = a.w;
        *reinterpret_cast<float4*>(&Bs[innerRowB * BN + innerColB4 * 4]) =
            *reinterpret_cast<const float4*>(&B[innerRowB * N + innerColB4 * 4]);
        __syncthreads();
        A += BK;
        B += BK * N;
        for (int d = 0; d < BK; ++d) {
            for (int i = 0; i < TM; ++i) regM[i] = As[d * BM + threadRow * TM + i];
            for (int j = 0; j < TN; ++j) regN[j] = Bs[d * BN + threadCol * TN + j];
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j) acc[i][j] += regM[i] * regN[j];
        }
        __syncthreads();
    }
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; j += 4) {
            float4 out;
            out.x = acc[i][j + 0];
            out.y = acc[i][j + 1];
            out.z = acc[i][j + 2];
            out.w = acc[i][j + 3];
            *reinterpret_cast<float4*>(&C[(threadRow * TM + i) * N + threadCol * TN + j]) = out;
        }
    }
}

void launch_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    const dim3 block(32, 32);
    const dim3 grid((M + 31) / 32, (N + 31) / 32);
    k_naive<<<grid, block>>>(A, B, C, M, N, K);
}

void launch_coalesced(const float* A, const float* B, float* C, int M, int N, int K) {
    const dim3 block(32, 32);
    const dim3 grid((N + 31) / 32, (M + 31) / 32);
    k_coalesced<<<grid, block>>>(A, B, C, M, N, K);
}

void launch_shared(const float* A, const float* B, float* C, int M, int N, int K) {
    constexpr int BS = 32;
    const dim3 block(BS, BS);
    const dim3 grid((N + BS - 1) / BS, (M + BS - 1) / BS);
    k_shared<BS><<<grid, block>>>(A, B, C, M, N, K);
}

void launch_blocktile_1d(const float* A, const float* B, float* C, int M, int N, int K) {
    constexpr int BM = 64, BN = 64, BK = 8, TM = 8;
    const dim3 block((BM * BN) / TM);
    const dim3 grid(N / BN, M / BM);
    k_blocktile_1d<BM, BN, BK, TM><<<grid, block>>>(A, B, C, M, N, K);
}

void launch_blocktile_2d(const float* A, const float* B, float* C, int M, int N, int K) {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    const dim3 block((BM * BN) / (TM * TN));
    const dim3 grid(N / BN, M / BM);
    k_blocktile_2d<BM, BN, BK, TM, TN><<<grid, block>>>(A, B, C, M, N, K);
}

void launch_vectorized(const float* A, const float* B, float* C, int M, int N, int K) {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    const dim3 block((BM * BN) / (TM * TN));
    const dim3 grid(N / BN, M / BM);
    k_vectorized<BM, BN, BK, TM, TN><<<grid, block>>>(A, B, C, M, N, K);
}

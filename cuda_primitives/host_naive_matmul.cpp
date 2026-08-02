// Copyright (c) 2026 John Law
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void naive_matmul(const float* A, const float* B, float* C,
                             int M, int N, int K);

#define CHECK(call)                                                        \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__,              \
                    cudaGetErrorString(err));                              \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

int main() {
    const int M = 1024, N = 1024, K = 1024;
    const size_t bytesA = size_t(M) * K * sizeof(float);
    const size_t bytesB = size_t(K) * N * sizeof(float);
    const size_t bytesC = size_t(M) * N * sizeof(float);

    float* hA = (float*)malloc(bytesA);
    float* hB = (float*)malloc(bytesB);
    float* hC = (float*)malloc(bytesC);
    for (size_t i = 0; i < size_t(M) * K; ++i) hA[i] = float((i % 7) - 3) * 0.5f;
    for (size_t i = 0; i < size_t(K) * N; ++i) hB[i] = float((i % 5) - 2) * 0.5f;

    float *dA, *dB, *dC;
    CHECK(cudaMalloc(&dA, bytesA));
    CHECK(cudaMalloc(&dB, bytesB));
    CHECK(cudaMalloc(&dC, bytesC));
    CHECK(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    naive_matmul<<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK(cudaEventRecord(start));
    for (int rep = 0; rep < 10; ++rep)
        naive_matmul<<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= 10.0f;

    CHECK(cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost));
    for (int spot = 0; spot < 5; ++spot) {
        int r = (spot * 211) % M, c = (spot * 401) % N;
        float acc = 0;
        for (int k = 0; k < K; ++k) acc += hA[size_t(r) * K + k] * hB[size_t(k) * N + c];
        if (fabsf(acc - hC[size_t(r) * N + c]) > 1e-2f) {
            fprintf(stderr, "mismatch at (%d,%d): %f != %f\n", r, c,
                    hC[size_t(r) * N + c], acc);
            return 1;
        }
    }

    double gflop = 2.0 * M * N * K / 1e9;
    printf("naive_matmul: %dx%dx%d  %.3f ms  %.1f GFLOP/s (verified)\n",
           M, N, K, ms, gflop / (ms / 1e3));

    free(hA); free(hB); free(hC);
    CHECK(cudaFree(dA)); CHECK(cudaFree(dB)); CHECK(cudaFree(dC));
    return 0;
}

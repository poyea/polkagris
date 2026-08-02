// Copyright (c) 2026 John Law
// SPDX-License-Identifier: MIT

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void vector_add(const float* a, const float* b, float* c, int n);

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
    const int n = 1 << 24;
    const size_t bytes = n * sizeof(float);

    float* ha = (float*)malloc(bytes);
    float* hb = (float*)malloc(bytes);
    float* hc = (float*)malloc(bytes);
    for (int i = 0; i < n; ++i) {
        ha[i] = float(i % 1000);
        hb[i] = float((n - i) % 1000);
    }

    float *da, *db, *dc;
    CHECK(cudaMalloc(&da, bytes));
    CHECK(cudaMalloc(&db, bytes));
    CHECK(cudaMalloc(&dc, bytes));
    CHECK(cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice));

    const int block = 256;
    const int grid = (n + block - 1) / block;

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    vector_add<<<grid, block>>>(da, db, dc, n);
    CHECK(cudaEventRecord(start));
    for (int rep = 0; rep < 100; ++rep)
        vector_add<<<grid, block>>>(da, db, dc, n);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= 100.0f;

    CHECK(cudaMemcpy(hc, dc, bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
        if (hc[i] != ha[i] + hb[i]) {
            fprintf(stderr, "mismatch at %d: %f != %f\n", i, hc[i], ha[i] + hb[i]);
            return 1;
        }
    }

    double gb = 3.0 * bytes / 1e9;
    printf("vector_add: n=%d  %.3f ms  %.1f GB/s (verified)\n", n, ms, gb / (ms / 1e3));

    free(ha); free(hb); free(hc);
    CHECK(cudaFree(da)); CHECK(cudaFree(db)); CHECK(cudaFree(dc));
    return 0;
}

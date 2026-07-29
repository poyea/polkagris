// Copyright (c) 2026 John Law
// SPDX-License-Identifier: MIT

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "sgemm.cuh"

#define CHECK(call)                                                            \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__,                 \
                    cudaGetErrorString(err));                                  \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CHECK_CUBLAS(call)                                                     \
    do {                                                                       \
        cublasStatus_t st = (call);                                            \
        if (st != CUBLAS_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "%s:%d cublas error %d\n", __FILE__, __LINE__,     \
                    int(st));                                                  \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

using Launch = void (*)(const float*, const float*, float*, int, int, int);

struct Rung {
    const char* name;
    Launch launch;
};

static const Rung kLadder[] = {
    {"naive", launch_naive},
    {"coalesced", launch_coalesced},
    {"shared tile", launch_shared},
    {"1d blocktile", launch_blocktile_1d},
    {"2d blocktile", launch_blocktile_2d},
    {"vectorized", launch_vectorized},
};

static float time_ms(void (*body)(void*), void* ctx, int warmup, int reps) {
    for (int i = 0; i < warmup; ++i) body(ctx);
    CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    for (int i = 0; i < reps; ++i) body(ctx);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    return ms / float(reps);
}

struct Ctx {
    const float* dA;
    const float* dB;
    float* dC;
    int M, N, K;
    Launch launch;
    cublasHandle_t handle;
    float alpha, beta;
};

static void run_kernel(void* p) {
    Ctx* c = static_cast<Ctx*>(p);
    c->launch(c->dA, c->dB, c->dC, c->M, c->N, c->K);
}

// cuBLAS is column-major, so asking for C^T = B^T A^T yields row-major C = A B.
static void run_cublas(void* p) {
    Ctx* c = static_cast<Ctx*>(p);
    CHECK_CUBLAS(cublasSgemm(c->handle, CUBLAS_OP_N, CUBLAS_OP_N, c->N, c->M, c->K,
                             &c->alpha, c->dB, c->N, c->dA, c->K, &c->beta, c->dC, c->N));
}

int main(int argc, char** argv) {
    const int n = (argc > 1) ? atoi(argv[1]) : 2048;
    if (n <= 0 || n % kTileMultiple != 0) {
        fprintf(stderr, "size must be a positive multiple of %d\n", kTileMultiple);
        return 1;
    }
    const int M = n, N = n, K = n;
    const size_t elemsA = size_t(M) * K, elemsB = size_t(K) * N, elemsC = size_t(M) * N;

    float* hA = (float*)malloc(elemsA * sizeof(float));
    float* hB = (float*)malloc(elemsB * sizeof(float));
    float* hRef = (float*)malloc(elemsC * sizeof(float));
    float* hOut = (float*)malloc(elemsC * sizeof(float));
    if (!hA || !hB || !hRef || !hOut) {
        fprintf(stderr, "host allocation failed\n");
        return 1;
    }
    for (size_t i = 0; i < elemsA; ++i) hA[i] = float((i % 17) - 8) * 0.125f;
    for (size_t i = 0; i < elemsB; ++i) hB[i] = float((i % 13) - 6) * 0.125f;

    float *dA, *dB, *dC;
    CHECK(cudaMalloc(&dA, elemsA * sizeof(float)));
    CHECK(cudaMalloc(&dB, elemsB * sizeof(float)));
    CHECK(cudaMalloc(&dC, elemsC * sizeof(float)));
    CHECK(cudaMemcpy(dA, hA, elemsA * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, hB, elemsB * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    Ctx ctx{dA, dB, dC, M, N, K, nullptr, handle, 1.0f, 0.0f};

    CHECK(cudaMemset(dC, 0, elemsC * sizeof(float)));
    const float cublas_ms = time_ms(run_cublas, &ctx, 3, 20);
    CHECK(cudaMemcpy(hRef, dC, elemsC * sizeof(float), cudaMemcpyDeviceToHost));

    for (int spot = 0; spot < 8; ++spot) {
        const int r = (spot * 211) % M, c = (spot * 401) % N;
        double want = 0.0;
        for (int k = 0; k < K; ++k)
            want += double(hA[size_t(r) * K + k]) * double(hB[size_t(k) * N + c]);
        const double got = hRef[size_t(r) * N + c];
        if (fabs(want - got) > 1e-2 * fmax(1.0, fabs(want))) {
            fprintf(stderr, "cublas reference wrong at (%d,%d): %f vs %f\n", r, c, got, want);
            return 1;
        }
    }

    const double gflop = 2.0 * double(M) * double(N) * double(K) / 1e9;
    printf("sgemm %dx%dx%d, verified against cuBLAS\n\n", M, N, K);
    printf("%-14s %10s %12s %10s %12s\n", "kernel", "ms", "GFLOP/s", "% cuBLAS", "max diff");
    printf("%-14s %10.3f %12.1f %9.0f%% %12s\n", "cuBLAS", cublas_ms, gflop / (cublas_ms / 1e3),
           100.0, "-");

    int failures = 0;
    for (const Rung& rung : kLadder) {
        ctx.launch = rung.launch;
        CHECK(cudaMemset(dC, 0, elemsC * sizeof(float)));
        rung.launch(dA, dB, dC, M, N, K);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(hOut, dC, elemsC * sizeof(float), cudaMemcpyDeviceToHost));

        double max_diff = 0.0;
        for (size_t i = 0; i < elemsC; ++i)
            max_diff = fmax(max_diff, fabs(double(hOut[i]) - double(hRef[i])));

        const float ms = time_ms(run_kernel, &ctx, 3, 20);
        const bool ok = max_diff <= 1e-2;
        if (!ok) ++failures;
        printf("%-14s %10.3f %12.1f %9.1f%% %12.2e%s\n", rung.name, ms, gflop / (ms / 1e3),
               100.0 * cublas_ms / ms, max_diff, ok ? "" : "  MISMATCH");
    }

    free(hA);
    free(hB);
    free(hRef);
    free(hOut);
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK(cudaFree(dA));
    CHECK(cudaFree(dB));
    CHECK(cudaFree(dC));
    return failures == 0 ? 0 : 1;
}

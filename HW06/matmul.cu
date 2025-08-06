#include "matmul.cuh"
#include <cstdio>

__global__ void matmul_kernel(const float *A, const float *B, float *C,
                              size_t n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n * n) {
        return;
    }

    int row = index / n;
    int col = index % n;

    C[row * n + col] = 0;
    for (int i = 0; i < n; i++) {
        C[row * n + col] += A[row * n + i] * B[i * n + col];
    }

    return;
}

void matmul(const float *A, const float *B, float *C, size_t n,
            unsigned int threads_per_block) {
    int blockCount = (n * n + threads_per_block - 1) / threads_per_block;

    matmul_kernel<<<blockCount, threads_per_block>>>(A, B, C, n);

    return;
}

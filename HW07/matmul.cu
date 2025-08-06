#include "matmul.cuh"

__global__ void matmul_kernel_1(const int *A, const int *B, int *C,
                                unsigned int n, unsigned int block_dim) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = n * block_dim * by;
    int aEnd = aBegin + n - 1;
    int aStep = block_dim;
    int bBegin = block_dim * bx;
    int bStep = block_dim * n;

    int Csub = 0;

    extern __shared__ int shared1[];
    int *As = shared1;
    int *Bs = shared1 + block_dim * block_dim;

    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        As[block_dim * ty + tx] = 0;
        if (a + n * ty + tx < n * n) {
            As[block_dim * ty + tx] = A[a + n * ty + tx];
        }

        Bs[block_dim * ty + tx] = 0;
        if (b + n * ty + tx < n * n) {
            Bs[block_dim * ty + tx] = B[b + n * ty + tx];
        }
        __syncthreads();

        for (int k = 0; k < block_dim; ++k) {
            Csub += As[block_dim * ty + k] * Bs[block_dim * k + tx];
        }
        __syncthreads();
    }

    int c = n * block_dim * by + block_dim * bx;
    if (c + n * ty + tx < n * n) {
        C[c + n * ty + tx] = Csub;
    }

    return;
}

__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n,
                       unsigned int block_dim) {
    int blockSize = block_dim;
    if (block_dim > n) {
        blockSize = n;
    }
    dim3 dimBlock(blockSize, blockSize);

    int gridSize = (n + blockSize - 1) / blockSize;
    dim3 dimGrid(gridSize, gridSize);

    int sharedSize = 2 * blockSize * blockSize * sizeof(int);

    matmul_kernel_1<<<dimGrid, dimBlock, sharedSize>>>(A, B, C, n, blockSize);
    cudaDeviceSynchronize();

    return;
}

__global__ void matmul_kernel_2(const float *A, const float *B, float *C,
                                unsigned int n, unsigned int block_dim) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = n * block_dim * by;
    int aEnd = aBegin + n - 1;
    int aStep = block_dim;
    int bBegin = block_dim * bx;
    int bStep = block_dim * n;

    float Csub = 0;

    extern __shared__ float shared2[];
    float *As = shared2;
    float *Bs = shared2 + block_dim * block_dim;

    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        As[block_dim * ty + tx] = 0;
        if (a + n * ty + tx < n * n) {
            As[block_dim * ty + tx] = A[a + n * ty + tx];
        }

        Bs[block_dim * ty + tx] = 0;
        if (b + n * ty + tx < n * n) {
            Bs[block_dim * ty + tx] = B[b + n * ty + tx];
        }
        __syncthreads();

        for (int k = 0; k < block_dim; ++k) {
            Csub += As[block_dim * ty + k] * Bs[block_dim * k + tx];
        }
        __syncthreads();
    }

    int c = n * block_dim * by + block_dim * bx;
    if (c + n * ty + tx < n * n) {
        C[c + n * ty + tx] = Csub;
    }

    return;
}

__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n,
                       unsigned int block_dim) {
    int blockSize = block_dim;
    if (block_dim > n) {
        blockSize = n;
    }
    dim3 dimBlock(blockSize, blockSize);

    int gridSize = (n + blockSize - 1) / blockSize;
    dim3 dimGrid(gridSize, gridSize);

    int sharedSize = 2 * blockSize * blockSize * sizeof(float);

    matmul_kernel_2<<<dimGrid, dimBlock, sharedSize>>>(A, B, C, n, blockSize);
    cudaDeviceSynchronize();

    return;
}

__global__ void matmul_kernel_3(const double *A, const double *B, double *C,
                                unsigned int n, unsigned int block_dim) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = n * block_dim * by;
    int aEnd = aBegin + n - 1;
    int aStep = block_dim;
    int bBegin = block_dim * bx;
    int bStep = block_dim * n;

    double Csub = 0;

    extern __shared__ double shared3[];
    double *As = shared3;
    double *Bs = shared3 + block_dim * block_dim;

    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        As[block_dim * ty + tx] = 0;
        if (a + n * ty + tx < n * n) {
            As[block_dim * ty + tx] = A[a + n * ty + tx];
        }

        Bs[block_dim * ty + tx] = 0;
        if (b + n * ty + tx < n * n) {
            Bs[block_dim * ty + tx] = B[b + n * ty + tx];
        }
        __syncthreads();

        for (int k = 0; k < block_dim; ++k) {
            Csub += As[block_dim * ty + k] * Bs[block_dim * k + tx];
        }
        __syncthreads();
    }

    int c = n * block_dim * by + block_dim * bx;
    if (c + n * ty + tx < n * n) {
        C[c + n * ty + tx] = Csub;
    }

    return;
}

__host__ void matmul_3(const double *A, const double *B, double *C,
                       unsigned int n, unsigned int block_dim) {
    int blockSize = block_dim;
    if (block_dim > n) {
        blockSize = n;
    }
    dim3 dimBlock(blockSize, blockSize);

    int gridSize = (n + blockSize - 1) / blockSize;
    dim3 dimGrid(gridSize, gridSize);

    int sharedSize = 2 * blockSize * blockSize * sizeof(double);

    matmul_kernel_3<<<dimGrid, dimBlock, sharedSize>>>(A, B, C, n, blockSize);
    cudaDeviceSynchronize();

    return;
}

#include "reduce.cuh"

__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    sdata[tid] = 0;
    if (idx < n) {
        sdata[tid] += g_idata[idx];
    }
    if (idx + blockDim.x < n) {
        sdata[tid] += g_idata[idx + blockDim.x];
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }

    return;
}

__host__ void reduce(float **input, float **output, unsigned int N,
                     unsigned int threads_per_block) {
    while (N > 1) {
        int threadCount = threads_per_block;
        int blockCount =
            (N + threads_per_block * 2 - 1) / (threads_per_block * 2);
        int sharedSize = threads_per_block * sizeof(float);

        reduce_kernel<<<blockCount, threadCount, sharedSize>>>(*input, *output,
                                                               N);
        cudaDeviceSynchronize();

        float *temp = *input;
        *input = *output;
        *output = temp;

        N = blockCount;
    }
    cudaDeviceSynchronize();

    return;
}

#include "stencil.cuh"

__global__ void stencil_kernel(const float *image, const float *mask,
                               float *output, unsigned int n, unsigned int R) {
    extern __shared__ float shared[];
    float *sharedMask = shared;
    int sharedMaskSize = 2 * R + 1;
    float *sharedImage = sharedMask + sharedMaskSize;
    int sharedImageSize = blockDim.x + 2 * R;
    float *sharedOutput = sharedImage + sharedImageSize;

    int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int localIndex = threadIdx.x;

    if (localIndex < 2 * R + 1) {
        sharedMask[localIndex] = mask[localIndex];
    }
    __syncthreads();

    int sR = R;
    if (globalIndex < n) {
        sharedImage[localIndex + sR] = image[globalIndex];
    } else {
        sharedImage[localIndex + sR] = 1;
    }

    if (localIndex < sR) {
        if (globalIndex - sR >= 0 && globalIndex - sR < n) {
            sharedImage[localIndex] = image[globalIndex - sR];
        } else {
            sharedImage[localIndex] = 1;
        }
    }

    if (localIndex >= blockDim.x - sR) {
        if (globalIndex + sR < n) {
            sharedImage[localIndex + 2 * sR] = image[globalIndex + sR];
        } else {
            sharedImage[localIndex + 2 * sR] = 1;
        }
    }
    __syncthreads();

    sharedOutput[localIndex] = 0;
    for (int i = -sR; i <= sR; i++) {
        sharedOutput[localIndex] +=
            sharedImage[localIndex + sR + i] * sharedMask[i + sR];
    }
    __syncthreads();

    if (globalIndex < n) {
        output[globalIndex] = sharedOutput[localIndex];
    }

    return;
}

__host__ void stencil(const float *image, const float *mask, float *output,
                      unsigned int n, unsigned int R,
                      unsigned int threads_per_block) {
    int blockCount = (n + threads_per_block - 1) / threads_per_block;

    int sharedMaskSize = 2 * R + 1;
    int sharedImageSize = threads_per_block + 2 * R;
    int sharedOutputSize = threads_per_block;
    int sharedMemoryBytes =
        (sharedMaskSize + sharedImageSize + sharedOutputSize) * sizeof(float);

    stencil_kernel<<<blockCount, threads_per_block, sharedMemoryBytes>>>(
        image, mask, output, n, R);

    return;
}

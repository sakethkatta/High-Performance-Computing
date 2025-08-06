#include "stencil.cuh"
#include <cstdio>
#include <cstdlib>
#include <ctime>

int main(int argc, char *argv[]) {
    int n = atoi(argv[1]);
    int R = atoi(argv[2]);
    int threads_per_block = atoi(argv[3]);

    float *image = new float[n];
    float *mask = new float[2 * R + 1];

    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        image[i] = -1.0 + (2.0 * rand() / RAND_MAX);
    }
    for (int i = 0; i <= 2 * R; i++) {
        mask[i] = -1.0 + (2.0 * rand() / RAND_MAX);
    }

    float *dImage;
    float *dMask;
    cudaMalloc((void **)&dImage, n * sizeof(float));
    cudaMalloc((void **)&dMask, (2 * R + 1) * sizeof(float));

    cudaMemcpy(dImage, image, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dMask, mask, (2 * R + 1) * sizeof(float),
               cudaMemcpyHostToDevice);

    float *dOutput;
    cudaMalloc((void **)&dOutput, n * sizeof(float));

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    stencil(dImage, dMask, dOutput, n, R, threads_per_block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    float *output = new float[n];
    cudaMemcpy(output, dOutput, n * sizeof(float), cudaMemcpyDeviceToHost);

    printf("%f\n", output[n - 1]);
    printf("%f\n", elapsedTime);

    delete[] image;
    delete[] mask;
    delete[] output;
    cudaFree(dImage);
    cudaFree(dMask);
    cudaFree(dOutput);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

#include "vscale.cuh"
#include <cstdio>
#include <cstdlib>
#include <ctime>

int main(int argc, char **argv) {
    int n = atoi(argv[1]);
    float *a = new float[n];
    float *b = new float[n];

    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        a[i] = -10.0 + (20.0 * rand() / RAND_MAX);
        b[i] = 0.0 + (1.0 * rand() / RAND_MAX);
    }

    float *dA;
    float *dB;
    cudaMalloc((void **)&dA, n * sizeof(float));
    cudaMalloc((void **)&dB, n * sizeof(float));

    cudaMemcpy(dA, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, b, n * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 512;
    int blocks = (n + threads - 1) / threads;

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vscale<<<blocks, threads>>>(dA, dB, n);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaMemcpy(b, dB, n * sizeof(float), cudaMemcpyDeviceToHost);

    printf("%f\n", elapsedTime);
    printf("%f\n", b[0]);
    printf("%f\n", b[n - 1]);

    delete[] a;
    delete[] b;
    cudaFree(dA);
    cudaFree(dB);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

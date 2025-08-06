#include "matmul.cuh"
#include <cstdio>
#include <cstdlib>
#include <ctime>

int main(int argc, char *argv[]) {
    int n = atoi(argv[1]);
    int threads_per_block = atoi(argv[2]);

    float *A = new float[n * n];
    float *B = new float[n * n];

    srand(time(NULL));
    for (int i = 0; i < n * n; i++) {
        A[i] = -1.0 + (2.0 * rand() / RAND_MAX);
        B[i] = -1.0 + (2.0 * rand() / RAND_MAX);
    }

    float *dA;
    float *dB;
    cudaMalloc((void **)&dA, n * n * sizeof(float));
    cudaMalloc((void **)&dB, n * n * sizeof(float));

    cudaMemcpy(dA, A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, n * n * sizeof(float), cudaMemcpyHostToDevice);

    float *dC;
    cudaMalloc((void **)&dC, n * n * sizeof(float));

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul(dA, dB, dC, n, threads_per_block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    float *C = new float[n * n];
    cudaMemcpy(C, dC, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    printf("%f\n", C[n * n - 1]);
    printf("%f\n", elapsedTime);

    delete[] A;
    delete[] B;
    delete[] C;
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

#include "matmul.cuh"
#include <cstdio>
#include <cstdlib>
#include <ctime>

void timeMatmul1(int n, int block_dim) {
    int *A = new int[n * n];
    int *B = new int[n * n];

    srand(time(NULL));
    for (int i = 0; i < n * n; i++) {
        A[i] = -100.0 + (200.0 * rand() / RAND_MAX);
        B[i] = -100.0 + (200.0 * rand() / RAND_MAX);
    }

    int *dA;
    int *dB;
    cudaMalloc((void **)&dA, n * n * sizeof(int));
    cudaMalloc((void **)&dB, n * n * sizeof(int));

    cudaMemcpy(dA, A, n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, n * n * sizeof(int), cudaMemcpyHostToDevice);

    int *dC;
    cudaMalloc((void **)&dC, n * n * sizeof(int));

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_1(dA, dB, dC, n, block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    int *C = new int[n * n];
    cudaMemcpy(C, dC, n * n * sizeof(int), cudaMemcpyDeviceToHost);

    printf("%d\n", C[0]);
    printf("%d\n", C[n * n - 1]);
    printf("%f\n", elapsedTime);

    delete[] A;
    delete[] B;
    delete[] C;
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void timeMatmul2(int n, int block_dim) {
    float *A = new float[n * n];
    float *B = new float[n * n];

    srand(time(NULL));
    for (int i = 0; i < n * n; i++) {
        A[i] = -100.0 + (200.0 * rand() / RAND_MAX);
        B[i] = -100.0 + (200.0 * rand() / RAND_MAX);
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
    matmul_2(dA, dB, dC, n, block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    float *C = new float[n * n];
    cudaMemcpy(C, dC, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    printf("%f\n", C[0]);
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
}

void timeMatmul3(int n, int block_dim) {
    double *A = new double[n * n];
    double *B = new double[n * n];

    srand(time(NULL));
    for (int i = 0; i < n * n; i++) {
        A[i] = -100.0 + (200.0 * rand() / RAND_MAX);
        B[i] = -100.0 + (200.0 * rand() / RAND_MAX);
    }

    double *dA;
    double *dB;
    cudaMalloc((void **)&dA, n * n * sizeof(double));
    cudaMalloc((void **)&dB, n * n * sizeof(double));

    cudaMemcpy(dA, A, n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, n * n * sizeof(double), cudaMemcpyHostToDevice);

    double *dC;
    cudaMalloc((void **)&dC, n * n * sizeof(double));

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_3(dA, dB, dC, n, block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    double *C = new double[n * n];
    cudaMemcpy(C, dC, n * n * sizeof(double), cudaMemcpyDeviceToHost);

    printf("%f\n", C[0]);
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
}

int main(int argc, char *argv[]) {
    int n = atoi(argv[1]);
    int block_dim = atoi(argv[2]);

    timeMatmul1(n, block_dim);
    timeMatmul2(n, block_dim);
    timeMatmul3(n, block_dim);

    return 0;
}

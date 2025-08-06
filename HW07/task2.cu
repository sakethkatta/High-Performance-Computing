#include "reduce.cuh"
#include <cstdio>
#include <cstdlib>
#include <ctime>

int main(int argc, char *argv[]) {
    int N = atoi(argv[1]);
    int threads_per_block = atoi(argv[2]);

    float *input = new float[N];

    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        input[i] = -1.0 + (2.0 * rand() / RAND_MAX);
    }

    float *dInput;
    cudaMalloc((void **)&dInput, N * sizeof(float));

    cudaMemcpy(dInput, input, N * sizeof(float), cudaMemcpyHostToDevice);

    int blockCount = (N + threads_per_block * 2 - 1) / (threads_per_block * 2);

    float *dOutput;
    cudaMalloc((void **)&dOutput, blockCount * sizeof(float));

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    reduce(&dInput, &dOutput, N, threads_per_block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    float sum;
    cudaMemcpy(&sum, dInput, sizeof(float), cudaMemcpyDeviceToHost);

    printf("%f\n", sum);
    printf("%f\n", elapsedTime);

    delete[] input;
    cudaFree(dInput);
    cudaFree(dOutput);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

#include <cstdio>
#include <cstdlib>
#include <ctime>

__global__ void computeLinearCombination(int *dA, int a) {
    int x = threadIdx.x;
    int y = blockIdx.x;
    dA[(y * 8) + x] = (a * x) + y;
}

int main() {
    int *dA;
    cudaMalloc((void **)&dA, 16 * sizeof(int));

    srand(time(NULL));
    int a = 20.0 * rand() / RAND_MAX;

    computeLinearCombination<<<2, 8>>>(dA, a);
    cudaDeviceSynchronize();

    int *hA = new int[16];
    cudaMemcpy(hA, dA, 16 * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 15; i++) {
        printf("%d ", hA[i]);
    }
    printf("%d\n", hA[15]);

    cudaFree(dA);
    delete[] hA;

    return 0;
}

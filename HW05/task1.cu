#include <cstdio>

__global__ void computeFactorial() {
    int factorial = 1;
    for (int i = 1; i <= threadIdx.x + 1; i++) {
        factorial *= i;
    }

    std::printf("%d!=%d\n", threadIdx.x + 1, factorial);
}

int main() {
    computeFactorial<<<1, 8>>>();
    cudaDeviceSynchronize();
    
    return 0;
}

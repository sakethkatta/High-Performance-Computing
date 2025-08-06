#include "matmul.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>

int main(int argc, char *argv[]) {
    int n = atoi(argv[1]);
    int t = atoi(argv[2]);

    float *A = new float[n * n];
    float *B = new float[n * n];
    float *C = new float[n * n];

    srand(time(NULL));
    for (int i = 0; i < n * n; i++) {
        A[i] = -n + (2.0 * n * rand() / RAND_MAX);
        B[i] = -n + (2.0 * n * rand() / RAND_MAX);
    }

    omp_set_num_threads(t);
    std::chrono::high_resolution_clock::time_point start =
        std::chrono::high_resolution_clock::now();
    mmul(A, B, C, n);
    std::chrono::high_resolution_clock::time_point end =
        std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            end - start);

    printf("%f\n", C[0]);
    printf("%f\n", C[(n * n) - 1]);
    printf("%f\n", duration.count());

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}

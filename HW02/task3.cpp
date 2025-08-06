#include "matmul.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

int main(int argc, char **argv) {
    double *A = new double[1000 * 1000];
    double *B = new double[1000 * 1000];
    double *C = new double[1000 * 1000];
    srand(time(NULL));

    for (int i = 0; i < 1000 * 1000; i++) {
        A[i] = -1000.0 + (2000.0 * rand() / RAND_MAX);
        B[i] = -1000.0 + (2000.0 * rand() / RAND_MAX);
    }

    std::vector<double> A_vector(A, A + (1000 * 1000));
    std::vector<double> B_vector(B, B + (1000 * 1000));
    printf("%d\n", 1000);

    std::chrono::high_resolution_clock::time_point start =
        std::chrono::high_resolution_clock::now();
    mmul1(A, B, C, 1000);
    std::chrono::high_resolution_clock::time_point end =
        std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            end - start);
    printf("%f\n", duration.count());
    printf("%f\n", C[(1000 * 1000) - 1]);

    start = std::chrono::high_resolution_clock::now();
    mmul2(A, B, C, 1000);
    end = std::chrono::high_resolution_clock::now();
    duration =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            end - start);
    printf("%f\n", duration.count());
    printf("%f\n", C[(1000 * 1000) - 1]);

    start = std::chrono::high_resolution_clock::now();
    mmul3(A, B, C, 1000);
    end = std::chrono::high_resolution_clock::now();
    duration =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            end - start);
    printf("%f\n", duration.count());
    printf("%f\n", C[(1000 * 1000) - 1]);

    start = std::chrono::high_resolution_clock::now();
    mmul4(A_vector, B_vector, C, 1000);
    end = std::chrono::high_resolution_clock::now();
    duration =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            end - start);
    printf("%f\n", duration.count());
    printf("%f\n", C[(1000 * 1000) - 1]);

    delete[] A;
    delete[] B;
    delete[] C;
    A_vector = std::vector<double>();
    B_vector = std::vector<double>();

    return 0;
}

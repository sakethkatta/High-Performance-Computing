#include "scan.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>

int main(int argc, char **argv) {
    int n = atoi(argv[1]);
    float *arr = new float[n];
    float *output = new float[n];
    srand(time(NULL));

    for (int i = 0; i < n; i++) {
        arr[i] = -1.0 + (2.0 * rand() / RAND_MAX);
    }

    std::chrono::high_resolution_clock::time_point start =
        std::chrono::high_resolution_clock::now();
    scan(arr, output, n);
    std::chrono::high_resolution_clock::time_point end =
        std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            end - start);

    printf("%f\n", duration.count());
    printf("%f\n", output[0]);
    printf("%f\n", output[n - 1]);

    delete[] arr;
    delete[] output;

    return 0;
}

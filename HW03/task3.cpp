#include "msort.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>

int main(int argc, char *argv[]) {
    int n = atoi(argv[1]);
    int t = atoi(argv[2]);
    int ts = atoi(argv[3]);

    int *arr = new int[n];

    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        arr[i] = -1000.0 + (2000.0 * rand() / RAND_MAX);
    }

    omp_set_num_threads(t);
    std::chrono::high_resolution_clock::time_point start =
        std::chrono::high_resolution_clock::now();
    msort(arr, n, ts);
    std::chrono::high_resolution_clock::time_point end =
        std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            end - start);

    printf("%d\n", arr[0]);
    printf("%d\n", arr[n - 1]);
    printf("%f\n", duration.count());

    delete[] arr;

    return 0;
}

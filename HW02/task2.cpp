#include "convolution.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>

int main(int argc, char **argv) {
    int n = atoi(argv[1]);
    int m = atoi(argv[2]);
    float *image = new float[n * n];
    float *output = new float[n * n];
    float *mask = new float[m * m];
    srand(time(NULL));

    for (int i = 0; i < n * n; i++) {
        image[i] = -10.0 + (20.0 * rand() / RAND_MAX);
    }

    for (int j = 0; j < m * m; j++) {
        mask[j] = -1.0 + (2.0 * rand() / RAND_MAX);
    }

    std::chrono::high_resolution_clock::time_point start =
        std::chrono::high_resolution_clock::now();
    convolve(image, output, n, mask, m);
    std::chrono::high_resolution_clock::time_point end =
        std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            end - start);

    printf("%f\n", duration.count());
    printf("%f\n", output[0]);
    printf("%f\n", output[(n * n) - 1]);

    delete[] image;
    delete[] output;
    delete[] mask;

    return 0;
}

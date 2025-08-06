#include "convolution.h"

void convolve(const float *image, float *output, std::size_t n,
              const float *mask, std::size_t m) {
#pragma omp parallel for
    for (std::size_t x = 0; x < n; x++) {
        for (std::size_t y = 0; y < n; y++) {
            output[(x * n) + y] = 0;
            for (std::size_t i = 0; i < m; i++) {
                for (std::size_t j = 0; j < m; j++) {
                    std::size_t image_i = x + i - ((m - 1) / 2);
                    std::size_t image_j = y + j - ((m - 1) / 2);
                    if ((image_i >= 0 && image_i < n) &&
                        (image_j >= 0 && image_j < n)) {
                        output[(x * n) + y] +=
                            mask[(i * m) + j] * image[(image_i * n) + image_j];
                    } else if ((image_i >= 0 && image_i < n) ||
                               (image_j >= 0 && image_j < n)) {
                        output[(x * n) + y] += mask[(i * m) + j];
                    }
                }
            }
        }
    }
    return;
}

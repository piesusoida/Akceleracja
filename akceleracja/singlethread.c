#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

// Windows export macro
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

EXPORT void process_image(uint8_t *pixels, int width, int height) {
    int x, y, i;
    int gx, gy;
    int pos;

    // Allocate a temporary buffer for the result
    uint8_t *output = (uint8_t *)malloc(width * height);
    if (!output) {
        printf("Error: Unable to allocate memory.\n");
        return;
    }

    // Sobel operator kernels
    int Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    int Gy[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    // Apply Sobel filter (skip borders)
    for (y = 1; y < height - 1; y++) {
        for (x = 1; x < width - 1; x++) {
            gx = 0;
            gy = 0;

            // Compute gradients
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int pixel = pixels[(y + ky) * width + (x + kx)];
                    gx += pixel * Gx[ky + 1][kx + 1];
                    gy += pixel * Gy[ky + 1][kx + 1];
                }
            }

            // Gradient magnitude (clamped to 0–255)
            int magnitude = (int)sqrt(gx * gx + gy * gy);
            if (magnitude > 255) magnitude = 255;
            if (magnitude < 0) magnitude = 0;

            output[y * width + x] = (uint8_t)magnitude;
        }
    }

    // Copy result back into the input buffer
    for (i = 0; i < width * height; i++) {
        pixels[i] = output[i];
    }

    free(output);
    printf("Applied Sobel filter on image %dx%d\n", width, height);
}

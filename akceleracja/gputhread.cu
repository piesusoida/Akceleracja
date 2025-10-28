#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

// CUDA kernel for Sobel filter
__global__ void sobel_kernel(uint8_t *input, uint8_t *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1)
        return;

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

    int gx = 0;
    int gy = 0;

    //gradient
    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int pixel = input[(y + ky) * width + (x + kx)];
            gx += pixel * Gx[ky + 1][kx + 1];
            gy += pixel * Gy[ky + 1][kx + 1];
        }
    }

    int magnitude = (int)sqrtf((float)(gx * gx + gy * gy));
    if (magnitude > 255) magnitude = 255;
    if (magnitude < 0) magnitude = 0;

    output[y * width + x] = (uint8_t)magnitude;
}

// Host function to launch kernel
extern "C" EXPORT void process_image(uint8_t *pixels, int width, int height) {
    uint8_t *d_input = NULL, *d_output = NULL;
    size_t size = width * height * sizeof(uint8_t);

    // Allocate GPU memory
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    // Copy image data to GPU
    cudaMemcpy(d_input, pixels, size, cudaMemcpyHostToDevice);

    // Launch CUDA kernel
    dim3 block(16, 16); //16x16 blocks
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    sobel_kernel<<<grid, block>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();

    // Copy result back to CPU memory
    cudaMemcpy(pixels, d_output, size, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);

    printf("Applied Sobel filter on image %dx%d (CUDA)\n", width, height);
}

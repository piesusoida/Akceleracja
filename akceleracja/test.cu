#include <stdio.h>

__global__ void testKernel() {
    printf("Hello from GPU!\n");
}

int main() {
    testKernel<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}

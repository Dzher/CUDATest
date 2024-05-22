#include <stdio.h>

int main()
{
    int gpu_count = 0;
    cudaError_t error = cudaGetDeviceCount(&gpu_count);

    if (error != cudaSuccess || gpu_count == 0) {
        printf("You don't have any GPU!\n");
        exit(-1);
    }
    else {
        printf("The number of you GPU is %d.\n", gpu_count);
    }

    int gpu_index = 0;
    if (cudaSuccess != cudaSetDevice(gpu_index)) {
        printf("Fail to set GPU 0 for computing.\n");
    }
    else {
        printf("Success set GPU 0 for computing.\n");
    }
    return 0;
}
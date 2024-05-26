#ifndef __CUDA__DEVICE_SET_AND_GET__
#define __CUDA__DEVICE_SET_AND_GET__

#include "stdio.h"

void getAndSetGpu0()
{
    int gpu_count;
    if (cudaSuccess != cudaGetDeviceCount(&gpu_count)) {
        printf("Error: Cannot get your device count!\n");
        return;
    }
    else {
        printf("Get GPU Count Success with number:%d\n", gpu_count);
    }

    if (cudaSuccess != cudaSetDevice(0)) {
        printf("Error: Cannot initial your device!\n");
        return;
    }
    else {
        printf("Initial GPU 0 Success!\n");
    }
}

#endif  //__CUDA__DEVICE_SET_AND_GET__
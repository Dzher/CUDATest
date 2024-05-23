#include <stdio.h>

void printfGPUComputeXY()
{
    int gpu_count = 0;
    cudaGetDeviceCount(&gpu_count);
    if (gpu_count > 0) {
        cudaDeviceProp cuda_device_prop;
        for (int gpu_index = 0; gpu_index < gpu_count; ++gpu_index) {
            cudaGetDeviceProperties(&cuda_device_prop, gpu_index);
            printf("Your GPU Compute Ability is: %d.%d\n", cuda_device_prop.major, cuda_device_prop.minor);
        }
    }
}

int main()
{
    printfGPUComputeXY();
}
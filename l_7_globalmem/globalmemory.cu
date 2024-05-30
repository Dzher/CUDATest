#include <array>
#include <iostream>

__device__ int device_var = 1234;
__device__ int device_array[2];

__global__ void devicePrint()
{
    device_array[0] += device_var;
    device_array[1] += device_var;
    printf("device var = %d, device array[0] = %d, device array[1] = %d\n", device_var, device_array[0],
           device_array[1]);
}

int main()
{
    cudaDeviceProp cuda_prop;
    cudaGetDeviceProperties(&cuda_prop, 0);
    std::cout << "The GPU name is " << cuda_prop.name << std::endl;

    std::array<int, 2> host_array{0, 0};
    cudaMemcpyToSymbol(device_array, &host_array, sizeof(int) * host_array.size());

    dim3 cuda_grid{1};
    dim3 cuda_block{1};

    devicePrint<<<cuda_grid, cuda_block>>>();
    cudaDeviceSynchronize();

    cudaMemcpyFromSymbol(&host_array, device_array, sizeof(int) * host_array.size());

    std::cout << "host array[0] = " << host_array[0] << ","
              << " host array[1] = " << host_array[1] << std::endl;

    cudaDeviceReset();
    return 0;
}
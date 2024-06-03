#include <array>
#include <cstddef>
#include <iostream>

extern __shared__ float shared_array[];

__global__ void deviceToSharedMemory(float* device_array, const int device_array_size)
{
    const int thread_id = threadIdx.x;
    const int block_id = blockIdx.x;

    const int current_thread_id = thread_id + blockDim.x * block_id;

    if (current_thread_id < device_array_size) {
        shared_array[thread_id] = device_array[current_thread_id];
    }
    __syncthreads();

    printf("shared_memory value is %f at block idx %d\n", shared_array[thread_id], block_id);
}

int main()
{
    cudaDeviceProp cuda_prop;
    cudaGetDeviceProperties(&cuda_prop, 0);
    std::cout << "The GPU you use named: " << cuda_prop.name << std::endl;

    constexpr int array_size = 64;
    std::array<float, array_size> host_array{};
    constexpr int array_memory_size = array_size * sizeof(float);

    for (int index = 0; index < array_size; ++index) {
        host_array[index] = float(index);
    }

    float* device_array = nullptr;
    cudaMalloc(&device_array, array_memory_size);
    cudaMemcpy(device_array, &host_array, array_memory_size, cudaMemcpyHostToDevice);

    dim3 block_dim = 32;
    dim3 grid_dim = 2;

    deviceToSharedMemory<<<grid_dim, block_dim, 32>>>(device_array, array_size);

    cudaFree(device_array);
    cudaDeviceReset();

    return 0;
}
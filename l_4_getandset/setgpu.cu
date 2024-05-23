#include <stdio.h>

void setGPU()
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
}

__device__ float add(const float x, const float y)
{
    return x + y;
}

__global__ void addFromGpu(float* mem_a, float* mem_b, float* result, const int adder_size)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < adder_size) {
        result[id] = add(mem_a[id], mem_b[id]);
    }
}

void randomInitialFloatData(float* mem, int count)
{
    for (int i = 0; i < count; ++i) {
        mem[i] = (float)(rand() / 10000.f);
    }
}

void memHostDeviceCpy()
{
    int mem_size = 512;
    size_t byte_size = mem_size * sizeof(float);

    float* host_float_a = new float[byte_size];
    float* host_float_b = new float[byte_size];
    float* host_float_result = new float[byte_size];

    if (host_float_a && host_float_b && host_float_result) {
        memset(host_float_a, 0, byte_size);
        memset(host_float_b, 0, byte_size);
        memset(host_float_result, 0, byte_size);
    }
    else {
        printf("Fail to allocate host memory!\n");
        exit(-1);
    }

    float* device_float_a;
    float* device_float_b;
    float* device_float_result;

    if (cudaSuccess == cudaMalloc(&device_float_a, byte_size) &&
        cudaSuccess == cudaMalloc(&device_float_b, byte_size) &&
        cudaSuccess == cudaMalloc(&device_float_result, byte_size)) {
        cudaMemset(device_float_a, 0, byte_size);
        cudaMemset(device_float_b, 0, byte_size);
        cudaMemset(device_float_result, 0, byte_size);
    }
    else {
        printf("Fail to allocate memory!\n");
        free(host_float_a);
        free(host_float_b);
        free(host_float_result);
        exit(-1);
    }

    randomInitialFloatData(host_float_a, mem_size);
    randomInitialFloatData(host_float_b, mem_size);

    cudaMemcpy(device_float_a, host_float_a, byte_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_float_b, host_float_b, byte_size, cudaMemcpyHostToDevice);

    dim3 block(32);
    dim3 grid((mem_size + block.x - 1) / 32);

    addFromGpu<<<grid, block>>>(device_float_a, device_float_b, device_float_result, mem_size);
    cudaDeviceSynchronize();

    cudaMemcpy(host_float_result, device_float_result, byte_size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; ++i) {
        printf("id=%d\tmatrix_a:%.4f\tmatrix_b:%.4f\tresult=%.4f\n", i + 1, host_float_a[i], host_float_b[i],
               host_float_result[i]);
    }

    free(host_float_a);
    free(host_float_b);
    free(host_float_result);
    cudaFree(device_float_a);
    cudaFree(device_float_b);
    cudaFree(device_float_result);

    cudaDeviceReset();
    return;
}

int main()
{
    setGPU();
    memHostDeviceCpy();

    return 0;
}
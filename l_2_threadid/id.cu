#include <stdio.h>

__global__ void printThreadId()
{
    const auto block_id = blockIdx.x;
    const auto thread_id = threadIdx.x;
    const int id = threadIdx.x + blockIdx.x * blockDim.x;

    printf("Hello from block %d and thread %d, global id %d\n", block_id, thread_id, id);
}

int main()
{
    printThreadId<<<2, 4>>>();
    cudaDeviceSynchronize();

    return 0;
}
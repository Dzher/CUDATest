#include <stdio.h>
#include "../tools/errorcheck.cu"

__global__ void testFromGpu()
{
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    printf("Test from the %d thread\n", id);
}

int main()
{
    dim3 block(1025);
    dim3 grid(1);

    testFromGpu<<<grid, block>>>();
    cudaErrorCheck(cudaGetLastError(), __FILE__, __LINE__);
    cudaErrorCheck(cudaDeviceSynchronize(), __FILE__, __LINE__);

    cudaDeviceSynchronize();
}
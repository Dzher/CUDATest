#include <stdio.h>

__global__ void helloFromGpu()
{
    printf("Hello world from the GPU\n");
}

int main()
{
    helloFromGpu<<<4, 4>>>();
    cudaDeviceSynchronize();

    return 0;
}
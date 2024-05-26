#include <vector>
#include "../tools/deviceinit.cu"
#include "../tools/errorcheck.cu"

void memcpyWithCheck()
{
    getAndSetGpu0();

    // set host float memory
    std::vector<float> float_mem(4, 0.0);

    // set device memory
    float* device_float_mem;
    cudaError_t error = cudaErrorCheck(cudaMalloc(&device_float_mem, 4), __FILE__, __LINE__);
    cudaMemset(device_float_mem, 0, 4);

    // error cudaMemcpyKind value here
    cudaErrorCheck(cudaMemcpy(device_float_mem, &float_mem, 4, cudaMemcpyDeviceToHost), __FILE__, __LINE__);

    cudaErrorCheck(cudaFree(device_float_mem), __FILE__, __LINE__);

    cudaErrorCheck(cudaDeviceReset(), __FILE__, __LINE__);

    return;
}

int main()
{
    memcpyWithCheck();
    return 0;
}

#ifndef __CUDA_ERROR_CHECK__FUNC__
#define __CUDA_ERROR_CHECK__FUNC__

#include <stdio.h>

cudaError_t cudaErrorCheck(cudaError_t error_code, const char* file_name, int error_line)
{
    if (error_code != cudaSuccess) {
        printf("CUDA Error: \ncode=%d, name=%s, description=%s\nfile=%s, line%d\n", error_code,
               cudaGetErrorName(error_code), cudaGetErrorString(error_code), file_name, error_line);
        return error_code;
    }
    return error_code;
}

#endif  //__CUDA_ERROR_CHECK__FUNC__

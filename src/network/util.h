#pragma once

#include <cuda.h>
#include <cuda_fp16.h>

#include <iostream>

#define CHECK_CUDNN(call)                                                      \
    do {                                                                       \
        cudnnStatus_t status = (call);                                         \
        if (status != CUDNN_STATUS_SUCCESS) {                                  \
            std::cerr << "cuDNN error at " << __FILE__ << ":" << __LINE__      \
                      << " code=" << status << " ("                            \
                      << cudnnGetErrorString(status) << ")" << std::endl;      \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)


#define CHECK_CUDA(expr) \
    do { \
        cudaError_t err = (expr); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                    << " at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

inline int ceildiv(int n, int m) {
    return (n + m - 1) / m;
}

template<typename T>
inline void copyConvertToDevice(T* d_dst, float* src, int n) {
    std::cerr << "Unsupported conversion" << std::endl;
}

template<>
inline void copyConvertToDevice<__half>(__half* d_dst, float* src, int n) {
    __half* t = static_cast<__half*>(malloc(n * sizeof(__half)));

    for (int i = 0; i < n; i++)
        t[i] = __float2half_rn(src[i]);

    cudaMemcpy(d_dst, t, n * sizeof(__half), cudaMemcpyHostToDevice);

    free(t);
}

template<>
inline void copyConvertToDevice<float>(float* d_dst, float* src, int n) {
    cudaMemcpy(d_dst, src, n * sizeof(float), cudaMemcpyHostToDevice);
}

void convertToOHWC(__half* d_mem, int o, int c, int h, int w);

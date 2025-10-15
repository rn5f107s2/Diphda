#include "util.h"

__global__ void OCHWToOHWC(__half* d_src, __half* d_dst, int O, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n   = O * C * H * W;

    if (idx >= n)
        return;

    int o = idx / (C * H * W);
    int c = (idx % (C * H * W)) / (H * W);
    int h = (idx % (H * W)) / W;
    int w = idx % W;

    int ohwc = o * (C * H * W) + h * (C * W) + w * C + c;

    d_dst[ohwc] = d_src[idx];
}

void convertToOHWC(__half* d_mem, int o, int c, int h, int w) {
    int n = o * h * w * c;

    __half* d_temp;
    cudaMalloc(&d_temp, n * sizeof(__half));

    cudaMemcpy(d_temp, d_mem, n * sizeof(__half), cudaMemcpyDeviceToDevice);

    const int threads = 256;
    const int blocks  = ceildiv(n, threads);

    OCHWToOHWC<<<threads, blocks>>>(d_temp, d_mem, o, c, h, w);

    cudaFree(d_temp);
}
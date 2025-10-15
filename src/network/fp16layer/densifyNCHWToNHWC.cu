#include "densifyNCHWToNHWC.h"
#include "../util.h"

__global__ void densifyNCHWToNHWC(int* sparse, __half* dense, int bs, int N, int C, int H, int W) {
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    int batch    = threadId / N;
    int idx      = threadId % N;

    if (batch >= bs)
        return;

    if (sparse[N * batch + idx] == -1)
        return;

    int nchw = sparse[N * batch + idx];
    int c = nchw / (H * W);
    int h = (nchw % (H * W)) / W;
    int w = nchw % W;

    int nhwc = h * (W * C) + w * C + c;

    dense[(C * H * W) * batch + nhwc] = 1.0f;
}

int DensifyNCHWToNHWCLayer::loadWeights(float* weights) {
    return 0;
}

__half* DensifyNCHWToNHWCLayer::forward(int* d_input) {
    int threads = 256;
    int blocks = ceildiv(inSize * batchSize, threads);

    cudaStream_t stream;
    cudnnGetStream(handle, &stream);

    cudaMemset(d_output, 0, channels * height * width * batchSize * sizeof(__half));

    densifyNCHWToNHWC<<<threads, blocks, 0, stream>>>(d_input, d_output, batchSize, inSize, channels, height, width);

    CHECK_CUDA(cudaDeviceSynchronize());

    return d_output;
}
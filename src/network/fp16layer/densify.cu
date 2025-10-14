#include "densify.h"
#include "../util.h"

__global__ void densify(int* sparse, __half* dense, int N, int out, int bs) {
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    int batch    = threadId / N;
    int idx      = threadId % N;

    if (batch >= bs)
        return;

    if (sparse[N * batch + idx] == -1)
        return;

    dense[out * batch + sparse[N * batch + idx]] = 1.0f;
}

int DensifyLayer::loadWeights(float* weights) {
    return 0;
}

__half* DensifyLayer::forward(int* d_input) {
    int threads = 256;
    int blocks = ceildiv(inSize * batchSize, threads);

    cudaStream_t stream;
    cudnnGetStream(handle, &stream);

    cudaMemset(d_output, 0, outSize * batchSize * sizeof(__half));

    densify<<<threads, blocks, 0, stream>>>(d_input, d_output, inSize, outSize, batchSize);

    return d_output;
}
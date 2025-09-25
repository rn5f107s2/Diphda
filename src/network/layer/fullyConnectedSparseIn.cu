#include "fullyConnectedSparseIn.h"
#include "../util.h"

__global__ void fcfwbatchedrelusparsein(int batchSize, int* in, int nIn, int out, float* output, float* weights, float* biases) {
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    int batch    = threadId / out;
    int idx      = threadId % out;

    if (batch >= batchSize)
        return;

    output[batch * out + idx] = biases[idx];

    for (int i = 0; i < nIn; i++) {
        int inIdx = in[i + batch * nIn];

        if (inIdx == -1)
            break;

        output[batch * out + idx] += weights[inIdx * out + idx];
    }

    if (output[batch * out + idx] < 0)
        output[batch * out + idx] = 0;
}

SparseInFullyConnectedLayer::SparseInFullyConnectedLayer(const cudnnHandle_t& hndl, int bs, int is, int os) : handle(hndl), batchSize(bs), inSize(is), outSize(os) {
    cudaMalloc(&d_weights, inSize * outSize * sizeof(float));
    cudaMalloc(&d_biases, outSize * sizeof(float));
    cudaMalloc(&d_output, outSize * batchSize * sizeof(float));
}

int SparseInFullyConnectedLayer::loadWeights(float* weights) {
    int nWeights = inSize * outSize;

    cudaMemcpy(d_weights, weights, nWeights * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, weights + nWeights, outSize * sizeof(float), cudaMemcpyHostToDevice);

    return nWeights + outSize;
}

float* SparseInFullyConnectedLayer::forward(int* d_input) {
    int threads = 256;
    int blocks = ceildiv(batchSize * outSize, threads);

    cudaStream_t stream; 
    
    cudnnGetStream(handle, &stream);

    fcfwbatchedrelusparsein<<<blocks, threads, 0, stream>>>(batchSize,
                                                            d_input,
                                                            32,
                                                            outSize, 
                                                            d_output, 
                                                            d_weights, 
                                                            d_biases);

    return d_output;
}
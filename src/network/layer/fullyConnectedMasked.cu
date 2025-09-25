#include "fullyConnectedMasked.h"
#include "../util.h"

__global__ void fcfwbatchedsparseout(int batchSize, int in, int* out, int nOut, int outSize, float* input, float* output, float* weights, float* biases) {
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    int batch    = threadId / nOut;

    if (batch >= batchSize)
        return;

    int idx    = threadId % nOut;
    int outIdx = out[idx + nOut * batch];
    
    if (outIdx == -1)
        return;

    output[batch * nOut + idx] = biases[outIdx];

    for (int i = 0; i < in; i++)
        output[batch * nOut + idx] += weights[i * outSize + outIdx] * input[batch * in + i];
}

MaskedFullyConnectedLayer::MaskedFullyConnectedLayer(const cudnnHandle_t& hndl, int bs, int is, int os) : handle(hndl), batchSize(bs), inSize(is), outSize(os) {
    cudaMalloc(&d_weights, inSize * outSize * sizeof(float));
    cudaMalloc(&d_biases, outSize * sizeof(float));
    cudaMalloc(&d_output, 218 * batchSize * sizeof(float));
}

int MaskedFullyConnectedLayer::loadWeights(float* weights) {
    int nWeights = inSize * outSize;

    cudaMemcpy(d_weights, weights, nWeights * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, weights + nWeights, outSize * sizeof(float), cudaMemcpyHostToDevice);

    return nWeights + outSize;
}

float* MaskedFullyConnectedLayer::forward(float* d_input, int* d_mask) {
    int threads = 256;
    int blocks = ceildiv(batchSize * 218, threads);

    cudaStream_t stream; 
    
    cudnnGetStream(handle, &stream);

    fcfwbatchedsparseout<<<blocks, threads, 0, stream>>>(batchSize, 
                                                         inSize, 
                                                         d_mask, 
                                                         218, 
                                                         outSize, 
                                                         d_input, 
                                                         d_output, 
                                                         d_weights, 
                                                         d_biases);

    return d_output;
}
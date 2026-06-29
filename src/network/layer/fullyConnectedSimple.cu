#include "fullyConnectedSimple.h"

#include "../util.h"

__global__ void fcfwbatched(int batchSize, int in, int out, float* input, float* output, float* weights, float* biases) {
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    int batch    = threadId / out;
    int idx      = threadId % out;

    if (batch >= batchSize)
        return;

    output[batch * out + idx] = biases[idx];

    for (int i = 0; i < in; i++)
        output[batch * out + idx] += weights[i * out + idx] * input[batch * in + i];
}

FullyConnectedLayerSimple::FullyConnectedLayerSimple(const cudnnHandle_t& hndl, int bs, int inSize, int outSize) : handle(hndl),
                                                                                                               in(inSize), 
                                                                                                               out(outSize),
                                                                                                               batchSize(bs) {
    cudaMalloc(&d_weights, inSize * outSize * sizeof(float));
    cudaMalloc(&d_biases, outSize * sizeof(float));
    cudaMalloc(&d_output, 218 * batchSize * sizeof(float));
}

int FullyConnectedLayerSimple::loadWeights(float* weights) {
    int nWeights = in * out;

    cudaMemcpy(d_weights, weights, nWeights * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, weights + nWeights, out * sizeof(float), cudaMemcpyHostToDevice);

    return nWeights + out;
}

float* FullyConnectedLayerSimple::forward(float* d_input) {
    int blocks = ceildiv(batchSize * out, 256);

    cudaStream_t stream;
    cudnnGetStream(handle, &stream);

    fcfwbatched<<<blocks, 256, 0, stream>>>(batchSize, 
                                            in, 
                                            out, 
                                            d_input, 
                                            d_output, 
                                            d_weights, 
                                            d_biases);

    return d_output;
}
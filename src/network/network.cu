#include <iostream>
#include <chrono>

#include "network.h"

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

__global__ void fcfwbatchedrelusparsein(int batchSize, int* in, int nIn, int inSize, int out, float* output, float* weights, float* biases) {
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

__global__ void fcfwbatchedsparseout(int batchSize, int in, int* out, int nOut, int outSize, float* input, float* output, float* weights, float* biases) {
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    int batch    = threadId / nOut;

    if (batch >= batchSize)
        return;

    int idx    = threadId % nOut;
    int outIdx = out[idx];
    
    if (outIdx == -1)
        return;

    output[batch * nOut + idx] = biases[outIdx];

    for (int i = 0; i < in; i++)
        output[batch * nOut + idx] += weights[i * outSize + outIdx] * input[batch * in + i];
}


__global__ void fcfwrelubatched(int batchSize, int in, int out, float* input, float* output, float* weights, float* biases) {
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    int batch    = threadId / out;
    int idx      = threadId % out;

    if (batch >= batchSize)
        return;

    output[batch * out + idx] = biases[idx];

    for (int i = 0; i < in; i++)
        output[batch * out + idx] += weights[i * out + idx] * input[batch * in + i];

    if (output[batch * out + idx] < 0)
        output[batch * out + idx] = 0;
}

CudaNetwork::CudaNetwork(int bs, float* policyWeights, float* valueWeights) : batchSize(bs) {
    cudaMalloc(&d_valueOutput , batchSize *        3 * sizeof(float));
    cudaMalloc(&d_policyOutput, batchSize * maxMoves * sizeof(float));

    cudaMalloc(&d_valueIntermediate , batchSize * valueLayer1Size  * sizeof(float));
    cudaMalloc(&d_policyIntermediate, batchSize * policyLayer1Size * sizeof(float));

    cudaMalloc(&d_policyOutIndices, batchSize * maxMoves  * sizeof(int));
    cudaMalloc(&d_inputIndices    , batchSize * maxInputs * sizeof(int));

    int nValueWeights  = 768 * valueLayer1Size  + valueLayer1Size  + valueLayer1Size  * valueLayer2Size  + valueLayer2Size;
    int nPolicyWeights = 768 * policyLayer1Size + policyLayer1Size + policyLayer1Size * policyLayer2Size + policyLayer2Size;

    cudaMalloc(&d_valueWeights , nValueWeights  * sizeof(float));
    cudaMalloc(&d_policyWeights, nPolicyWeights * sizeof(float));

    cudaMemcpy(d_valueWeights , valueWeights , nValueWeights  * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_policyWeights, policyWeights, nPolicyWeights * sizeof(float), cudaMemcpyHostToDevice);
}

void CudaNetwork::forward(int* inputIndices, int* policyOutputIndices, float* valueOutput, float* policyOutput) {
    const int threads = 256;

    cudaMemcpy(d_inputIndices    , inputIndices       , maxInputs * batchSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_policyOutIndices, policyOutputIndices, maxMoves  * batchSize * sizeof(int), cudaMemcpyHostToDevice);

    {
        int blocks = (batchSize * valueLayer1Size + threads - 1) / threads;
        fcfwbatchedrelusparsein<<<blocks, threads>>>(batchSize,
                                                     d_inputIndices,
                                                     maxInputs,
                                                     768, 
                                                     valueLayer1Size, 
                                                     d_valueIntermediate, 
                                                     &d_valueWeights[0], 
                                                     &d_valueWeights[768 * valueLayer1Size]);
    }

    {
        int blocks = (batchSize * policyLayer1Size + threads - 1) / threads;
        fcfwbatchedrelusparsein<<<blocks, threads>>>(batchSize, 
                                                     d_inputIndices,
                                                     maxInputs,
                                                     768, 
                                                     policyLayer1Size, 
                                                     d_policyIntermediate, 
                                                     &d_policyWeights[0], 
                                                     &d_policyWeights[768 * policyLayer1Size]);
    }

    cudaDeviceSynchronize();

    {
        const int weightsOffset = 768 * valueLayer1Size + valueLayer1Size;

        int blocks = (batchSize * valueLayer2Size + threads - 1) / threads;
        fcfwbatched<<<blocks, threads>>>(batchSize, 
                                         valueLayer1Size, 
                                         valueLayer2Size, 
                                         d_valueIntermediate, 
                                         d_valueOutput, 
                                         &d_valueWeights[weightsOffset], 
                                         &d_valueWeights[weightsOffset + valueLayer1Size * valueLayer2Size]);
    }

    {
        const int weightsOffset = 768 * policyLayer1Size + policyLayer1Size;

        int blocks = (batchSize * maxMoves + threads - 1) / threads;
        fcfwbatchedsparseout<<<blocks, threads>>>(batchSize, 
                                                  policyLayer1Size,
                                                  d_policyOutIndices,
                                                  maxMoves, 
                                                  policyLayer2Size,
                                                  d_policyIntermediate, 
                                                  d_policyOutput, 
                                                  &d_policyWeights[weightsOffset], 
                                                  &d_policyWeights[weightsOffset + policyLayer1Size * policyLayer2Size]);
    }

    cudaDeviceSynchronize();

    cudaMemcpy(valueOutput , d_valueOutput , batchSize *        3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(policyOutput, d_policyOutput, batchSize * maxMoves * sizeof(float), cudaMemcpyDeviceToHost);
}

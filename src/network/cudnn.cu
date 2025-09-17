#include <cudnn.h>
#include <stdio.h>
#include <iostream>

#include "cudnn.h"

__global__ void densify(int* sparse, float* dense, int N, int out, int bs) {
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    int batch    = threadId / N;
    int idx      = threadId % N;

    if (batch >= bs)
        return;

    if (sparse[N * batch + idx] == -1)
        return;

    dense[out * batch + sparse[N * batch + idx]] = 1.0f;
}

__global__ void mask(float* inputs, float* outputs, int* mask, int maxTId) {
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;

    if (threadId >= maxTId)
        return;

    if (mask[threadId] != -1)
        outputs[threadId] = inputs[mask[threadId]];
}

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

ConvLayer::ConvLayer(cudnnHandle_t& hndl, int bs, int ic, int oc, int kw, int kh, int h, int w, bool activate) : handle(hndl),
                                                                                                                 batchSize(bs), 
                                                                                                                 inChannels(ic),
                                                                                                                 outChannels(oc),
                                                                                                                 kernelWidth(kw),
                                                                                                                 kernelHeight(kh),
                                                                                                                 height(h),
                                                                                                                 width(w) {
    cudnnCreateTensorDescriptor(&inputDesc);
    cudnnCreateTensorDescriptor(&outputDesc);
    cudnnCreateTensorDescriptor(&biasDesc);
    cudnnCreateFilterDescriptor(&kernelDesc);
    cudnnCreateConvolutionDescriptor(&convDesc);
    cudnnCreateActivationDescriptor(&actDesc);

    cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, inChannels, height, width);
    cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, outChannels, height, width);
    cudnnSetTensor4dDescriptor(biasDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, outChannels, 1, 1);
    cudnnSetFilter4dDescriptor(kernelDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, outChannels, inChannels, kernelHeight, kernelWidth);
    cudnnSetConvolution2dDescriptor(convDesc, kernelHeight / 2, kernelWidth / 2, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    cudnnSetActivationDescriptor(actDesc, activate ? CUDNN_ACTIVATION_RELU : CUDNN_ACTIVATION_IDENTITY, CUDNN_PROPAGATE_NAN, 0.0);

    cudnnGetConvolutionForwardWorkspaceSize(handle, inputDesc, kernelDesc, convDesc, outputDesc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, &workspaceSize);

    cudaMalloc(&d_workspace, workspaceSize);
    cudaMalloc(&d_output, outChannels * height * width * batchSize * sizeof(float));
    cudaMalloc(&d_biases, outChannels * sizeof(float));
    cudaMalloc(&d_weights, inChannels * outChannels * kernelHeight * kernelWidth * sizeof(float));
}

void ConvLayer::loadWeights(float* weights) {
    int nWeights = inChannels * outChannels * kernelHeight * kernelWidth;

    cudaMemcpy(d_weights, weights, nWeights * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, weights + nWeights, outChannels * sizeof(float), cudaMemcpyHostToDevice);
}

float* ConvLayer::forward(float* d_input) {
    float alpha1 = 1.0f;
    float alpha2 = 0.0f;

    cudnnConvolutionBiasActivationForward(handle, &alpha1, 
                                          inputDesc, d_input, 
                                          kernelDesc, d_weights,
                                          convDesc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, 
                                          d_workspace, workspaceSize, 
                                          &alpha2, outputDesc, d_output, 
                                          biasDesc, d_biases, 
                                          actDesc, 
                                          outputDesc, d_output);

    return d_output;
}

FullyConnectedLayer::FullyConnectedLayer(cudnnHandle_t& hndl, int bs, int inSize, int outSize, bool activate) 
    : cl(ConvLayer(hndl, bs, inSize, outSize, 1, 1, 1, 1, activate)),
      in(inSize),
      out(outSize) {}

void FullyConnectedLayer::loadWeights(float* weights) {
    float* wT = (float*) malloc((in * out + out) * sizeof(float));

    for (int i = 0; i < in; i++)
        for (int j = 0; j < out; j++) {
            int i1 = i * out + j;
            int i2 = j * in + i;

            wT[i2] = weights[i1];
        }

    for (int i = in * out; i < in * out + out; i++)
        wT[i] = weights[i];

    cl.loadWeights(wT);

    free(wT);
}

float* FullyConnectedLayer::forward(float* d_input) {
    return cl.forward(d_input);
}

SparseInFullyConnectedLayer::SparseInFullyConnectedLayer(cudnnHandle_t& hndl, int bs, int is, int os) : handle(hndl), batchSize(bs), inSize(is), outSize(os) {
    cudaMalloc(&d_weights, inSize * outSize * sizeof(float));
    cudaMalloc(&d_biases, outSize * sizeof(float));
    cudaMalloc(&d_output, outSize * batchSize * sizeof(float));
}

void SparseInFullyConnectedLayer::loadWeights(float* weights, float* biases) {
    cudaMemcpy(d_biases, biases, outSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, inSize * outSize * sizeof(float), cudaMemcpyHostToDevice);
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

SparseOutFullyConnectedLayer::SparseOutFullyConnectedLayer(cudnnHandle_t& hndl, int bs, int is, int os) : handle(hndl), batchSize(bs), inSize(is), outSize(os) {
    cudaMalloc(&d_weights, inSize * outSize * sizeof(float));
    cudaMalloc(&d_biases, outSize * sizeof(float));
    cudaMalloc(&d_output, 218 * batchSize * sizeof(float));
}

void SparseOutFullyConnectedLayer::loadWeights(float* weights, float* biases) {
    cudaMemcpy(d_biases, biases, outSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, inSize * outSize * sizeof(float), cudaMemcpyHostToDevice);
}

float* SparseOutFullyConnectedLayer::forward(float* d_input, int* d_mask) {
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

void CudNNNetwork::forward(int* inputIndices, int* policyOutputIndices, float* valueOutput, float* policyOutput) {
    cudaMemcpy(d_sparseInput, inputIndices, sizeof(int) * batchSize * 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_policyMask, policyOutputIndices, sizeof(int) * batchSize * 218, cudaMemcpyHostToDevice);
    cudaMemset(d_denseInput, 0, 768 * batchSize * sizeof(float));

    cudaDeviceSynchronize();

    float* p = fcp1.forward(d_sparseInput);
    p = fcp2.forward(p, d_policyMask);

    float* v = fcv1.forward(d_sparseInput);
    v = fcv2->forward(v);
    
    cudaStreamSynchronize(valueStream);
    cudaStreamSynchronize(policyStream);

    cudaMemcpy(valueOutput, v, batchSize * 1 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(policyOutput, p, batchSize * 218 * sizeof(float), cudaMemcpyDeviceToHost);
}

FullyConnectedLayerCUDA::FullyConnectedLayerCUDA(cudnnHandle_t& hndl, int bs, int inSize, int outSize) : handle(hndl),
                                                                                                         in(inSize), 
                                                                                                         out(outSize),
                                                                                                         batchSize(bs) {
    cudaMalloc(&d_weights, inSize * outSize * sizeof(float));
    cudaMalloc(&d_biases, outSize * sizeof(float));
    cudaMalloc(&d_output, 218 * batchSize * sizeof(float));
}

void FullyConnectedLayerCUDA::loadWeights(float* weights) {
    int nWeights = in * out;

    cudaMemcpy(d_weights, weights, nWeights * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, weights + nWeights, out * sizeof(float), cudaMemcpyHostToDevice);
}

float* FullyConnectedLayerCUDA::forward(float* d_input) {
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


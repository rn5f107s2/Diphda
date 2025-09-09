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

void ConvLayer::loadWeights(float* weights, float* biases) {
    cudaMemcpy(d_biases, biases, outChannels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, inChannels * outChannels * kernelHeight * kernelWidth * sizeof(float), cudaMemcpyHostToDevice);
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

void FullyConnectedLayer::loadWeights(float* weights, float* biases) {
    float* wT = (float*) malloc(in * out * sizeof(float));

    for (int i = 0; i < in; i++)
        for (int j = 0; j < out; j++) {
            int i1 = i * out + j;
            int i2 = j * in + i;

            wT[i2] = weights[i1];
        }

    cl.loadWeights(wT, biases);

    free(wT);
}

float* FullyConnectedLayer::forward(float* d_input) {
    return cl.forward(d_input);
}

void CudNNNetwork::forward(int* inputIndices, int* policyOutputIndices, float* valueOutput, float* policyOutput) {
    cudaMemcpy(d_sparseInput, inputIndices, sizeof(int) * batchSize * 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_policyMask, policyOutputIndices, sizeof(int) * batchSize * 218, cudaMemcpyHostToDevice);
    cudaMemset(d_denseInput, 0, 768 * batchSize * sizeof(float));

    cudaDeviceSynchronize();

    int blocks = ((32 * batchSize) + 255) / 256;
    densify<<<blocks, 256>>>(d_sparseInput, d_denseInput, 32, 768, batchSize);

    cudaDeviceSynchronize();

    float* p = fcp1.forward(d_denseInput);
    cudaDeviceSynchronize();
    p = fcp2.forward(p);
    cudaDeviceSynchronize();


    blocks = ((218 * batchSize) + 255) / 256;
    mask<<<blocks, 256>>>(p, d_policySparseOutput, d_policyMask, batchSize * 218);
    cudaDeviceSynchronize();


    float* v = fcv1.forward(d_denseInput);
    cudaDeviceSynchronize();

    float temp[1024];
    cudaMemcpy(temp, v, 1024 * 4, cudaMemcpyDeviceToHost);

    v = fcv2.forward(v);
    cudaDeviceSynchronize();

    cudaMemcpy(valueOutput, v, batchSize * 1 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(policyOutput, d_policySparseOutput, batchSize * 218 * sizeof(float), cudaMemcpyDeviceToHost);
}

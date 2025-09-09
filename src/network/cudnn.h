#pragma once

#include <cuda_runtime.h>
#include <cudnn.h>

__global__ void densify(int* sparse, float* dense, int N, int out, int bs);
__global__ void mask(float* inputs, float* outputs, int* mask);

struct ConvLayer {
    const int batchSize, inChannels, outChannels, kernelWidth, kernelHeight, height, width;

    const cudnnHandle_t& handle;

    float* d_weights, *d_biases, *d_workspace, *d_output;

    size_t workspaceSize;

    cudnnTensorDescriptor_t inputDesc, outputDesc, biasDesc;
    cudnnFilterDescriptor_t kernelDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnActivationDescriptor_t actDesc;

    ConvLayer(cudnnHandle_t& hndl, int bs, int ic, int oc, int kw, int kh, int h, int w, bool activate = true);

    float* forward(float* d_input);
    void loadWeights(float* weights, float* biases);
};

struct FullyConnectedLayer {
    ConvLayer cl;

    int in, out;

    FullyConnectedLayer(cudnnHandle_t& hndl, int bs, int inSize, int outSize, bool activate = true);

    float* forward(float* d_input);
    void loadWeights(float* weights, float* biases);
};

struct CudNNNetwork {
    FullyConnectedLayer fcp1;
    FullyConnectedLayer fcp2;
    FullyConnectedLayer fcv1;
    FullyConnectedLayer fcv2;

    cudnnHandle_t handle;

    float* d_denseInput, *d_policySparseOutput;

    int *d_sparseInput, *d_policyMask;

    int batchSize;

    CudNNNetwork(int bs, float* policyWeights, float* valueWeights) : fcp1(FullyConnectedLayer(handle, bs, 768, 256)),
                                                                      fcp2(FullyConnectedLayer(handle, bs, 256, 4096, false)),
                                                                      fcv1(FullyConnectedLayer(handle, bs, 768, 1024)),
                                                                      fcv2(FullyConnectedLayer(handle, bs, 1024, 1, false)) {
        cudnnCreate(&handle);

        fcp1.loadWeights(policyWeights, policyWeights + (768 * 256));
        fcp2.loadWeights(policyWeights + (768 * 256) + 256, policyWeights + (768 * 256) + 256 + (256 * 4096));
        fcv1.loadWeights(valueWeights, valueWeights + (768 * 1024));
        fcv2.loadWeights(valueWeights + (768 * 1024) + 1024, valueWeights + (768 * 1024) + 1024 + (1024 * 1));

        cudaMalloc(&d_denseInput, 768 * sizeof(float) * bs);
        cudaMalloc(&d_sparseInput, 32 * sizeof(int) * bs);
        cudaMalloc(&d_policyMask, 218 * sizeof(int) * bs);
        cudaMalloc(&d_policySparseOutput, 218 * sizeof(float) * bs);

        batchSize = bs;
    }

    void forward(int* inputIndices, int* policyOutputIndices, float* valueOutput, float* policyOutput);
};
#pragma once

#include <cuda_runtime.h>
#include <cudnn.h>

__global__ void densify(int* sparse, float* dense, int N, int out, int bs);
__global__ void mask(float* inputs, float* outputs, int* mask);

struct ConvLayer {
    const cudnnHandle_t& handle;

    const int batchSize, inChannels, outChannels, kernelWidth, kernelHeight, height, width;

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

struct SparseInFullyConnectedLayer {
    const cudnnHandle_t& handle;

    const int batchSize, inSize, outSize;

    float* d_weights, *d_biases, *d_output;

    SparseInFullyConnectedLayer(cudnnHandle_t& hndl, int bs, int is, int os);

    float* forward(int* d_input);
    void loadWeights(float* weights, float* biases);
};

struct CudNNNetwork {
    SparseInFullyConnectedLayer fcp1;
    FullyConnectedLayer fcp2;
    SparseInFullyConnectedLayer fcv1;
    FullyConnectedLayer fcv2;

    cudnnHandle_t valueHandle;
    cudnnHandle_t policyHandle;

    cudaStream_t valueStream;
    cudaStream_t policyStream;

    float* d_denseInput, *d_policySparseOutput;

    int *d_sparseInput, *d_policyMask;

    int batchSize;

    CudNNNetwork(int bs, float* policyWeights, float* valueWeights) : fcp1(SparseInFullyConnectedLayer(policyHandle, bs, 768, 256)),
                                                                      fcp2(FullyConnectedLayer(policyHandle, bs, 256, 4096, false)),
                                                                      fcv1(SparseInFullyConnectedLayer(valueHandle , bs, 768, 1024)),
                                                                      fcv2(FullyConnectedLayer(valueHandle , bs, 1024, 1, false)) {
        cudnnCreate(&policyHandle);
        cudnnCreate(&valueHandle);

        cudaStreamCreate(&valueStream);
        cudaStreamCreate(&policyStream);

        cudnnSetStream(policyHandle, policyStream);
        cudnnSetStream(valueHandle , valueStream );

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

inline int ceildiv(int n, int m) {
    return (n + m - 1) / m;
}
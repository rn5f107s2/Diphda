#pragma once

#include <cuda_runtime.h>
#include <cudnn.h>

class DenseLayer {
public:
    virtual float* forward(float* d_input) = 0;
    virtual void loadWeights(float* weights) = 0;
};

class SparseLayer {
public:
    virtual float* forward(int* d_input) = 0;
    virtual void loadWeights(float* weights) = 0;
};

class MaskedLayer {
public:
    virtual float* forward(float* d_input, int* d_mask) = 0;
    virtual void loadWeights(float* weights) = 0;
};

class ConvLayer : public DenseLayer {
    const cudnnHandle_t& handle;

    const int batchSize, inChannels, outChannels, kernelWidth, kernelHeight, height, width;

    float* d_weights, *d_biases, *d_workspace, *d_output;

    size_t workspaceSize;

    cudnnTensorDescriptor_t inputDesc, outputDesc, biasDesc;
    cudnnFilterDescriptor_t kernelDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnActivationDescriptor_t actDesc;

public:
    ConvLayer(cudnnHandle_t& hndl, int bs, int ic, int oc, int kw, int kh, int h, int w, bool activate = true);

    float* forward(float* d_input) override;
    void loadWeights(float* weights) override;
};

class FullyConnectedLayer : public DenseLayer {
    ConvLayer cl;

    int in, out;

public:
    FullyConnectedLayer(cudnnHandle_t& hndl, int bs, int inSize, int outSize, bool activate = true);

    float* forward(float* d_input) override;
    void loadWeights(float* weights) override;
};

class FullyConnectedLayerCUDA : public DenseLayer {
    const cudnnHandle_t& handle;

    const int in, out, batchSize;

    float* d_weights, *d_biases, *d_output;

public:
    FullyConnectedLayerCUDA(cudnnHandle_t& hndl, int bs, int inSize, int outSize);

    float* forward(float* d_input) override;
    void loadWeights(float* weights) override;
};

struct SparseInFullyConnectedLayer : public SparseLayer {
    const cudnnHandle_t& handle;

    const int batchSize, inSize, outSize;

    float* d_weights, *d_biases, *d_output;

public:
    SparseInFullyConnectedLayer(cudnnHandle_t& hndl, int bs, int is, int os);

    float* forward(int* d_input) override;
    void loadWeights(float* weights) override;
};

struct MaskedFullyConnectedLayer : public MaskedLayer {
    const cudnnHandle_t& handle;

    const int batchSize, inSize, outSize;

    float* d_weights, *d_biases, *d_output;

public:
    MaskedFullyConnectedLayer(cudnnHandle_t& hndl, int bs, int is, int os);

    float* forward(float* d_input, int* d_mask) override;
    void loadWeights(float* weights) override;
};

struct CudNNNetwork {
    SparseLayer* fcp1;
    MaskedLayer* fcp2;
    SparseLayer* fcv1;
    DenseLayer* fcv2;

    cudnnHandle_t valueHandle;
    cudnnHandle_t policyHandle;

    cudaStream_t valueStream;
    cudaStream_t policyStream;

    float* d_denseInput, *d_policySparseOutput;

    int *d_sparseInput, *d_policyMask;

    int batchSize;

    CudNNNetwork(int bs, float* policyWeights, float* valueWeights) : fcp1(new SparseInFullyConnectedLayer(policyHandle, bs, 768, 256)),
                                                                      fcp2(new MaskedFullyConnectedLayer(policyHandle, bs, 256, 4096)),
                                                                      fcv1(new SparseInFullyConnectedLayer(valueHandle , bs, 768, 1024)),
                                                                      fcv2(new FullyConnectedLayerCUDA(valueHandle , bs, 1024, 1)) {
        cudnnCreate(&policyHandle);
        cudnnCreate(&valueHandle);

        cudaStreamCreate(&valueStream);
        cudaStreamCreate(&policyStream);

        cudnnSetStream(policyHandle, policyStream);
        cudnnSetStream(valueHandle , valueStream );

        fcp1->loadWeights(policyWeights);
        fcp2->loadWeights(policyWeights + (768 * 256) + 256);
        fcv1->loadWeights(valueWeights);
        fcv2->loadWeights(valueWeights + (768 * 1024) + 1024);

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
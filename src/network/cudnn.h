#pragma once

#include <cuda_runtime.h>
#include <cudnn.h>
#include <vector>

class DenseLayer {
public:
    virtual float* forward(float* d_input) = 0;
    virtual int loadWeights(float* weights) = 0;
};

class SparseLayer {
public:
    virtual float* forward(int* d_input) = 0;
    virtual int loadWeights(float* weights) = 0;
};

class MaskedLayer {
public:
    virtual float* forward(float* d_input, int* d_mask) = 0;
    virtual int loadWeights(float* weights) = 0;
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
    ConvLayer(const cudnnHandle_t& hndl, int bs, int ic, int oc, int kw, int kh, int h, int w, bool activate = true);

    float* forward(float* d_input) override;
    int loadWeights(float* weights) override;
};

class FullyConnectedLayer : public DenseLayer {
    ConvLayer cl;

    int in, out;

public:
    FullyConnectedLayer(const cudnnHandle_t& hndl, int bs, int inSize, int outSize, bool activate = true);

    float* forward(float* d_input) override;
    int loadWeights(float* weights) override;
};

class FullyConnectedLayerCUDA : public DenseLayer {
    const cudnnHandle_t& handle;

    const int in, out, batchSize;

    float* d_weights, *d_biases, *d_output;

public:
    FullyConnectedLayerCUDA(const cudnnHandle_t& hndl, int bs, int inSize, int outSize);

    float* forward(float* d_input) override;
    int loadWeights(float* weights) override;
};

class SparseInFullyConnectedLayer : public SparseLayer {
    const cudnnHandle_t& handle;

    const int batchSize, inSize, outSize;

    float* d_weights, *d_biases, *d_output;

public:
    SparseInFullyConnectedLayer(const cudnnHandle_t& hndl, int bs, int is, int os);

    float* forward(int* d_input) override;
    int loadWeights(float* weights) override;
};

class MaskedFullyConnectedLayer : public MaskedLayer {
    const cudnnHandle_t& handle;

    const int batchSize, inSize, outSize;

    float* d_weights, *d_biases, *d_output;

public:
    MaskedFullyConnectedLayer(const cudnnHandle_t& hndl, int bs, int is, int os);

    float* forward(float* d_input, int* d_mask) override;
    int loadWeights(float* weights) override;
};

class DensifyLayer : public SparseLayer {
    const cudnnHandle_t& handle;

    const int batchSize, inSize, outSize;

    float* d_output;

public:
    DensifyLayer(const cudnnHandle_t& hndl, int bs, int in, int out) : handle(hndl), batchSize(bs), inSize(in), outSize(out) {
        cudaMalloc(&d_output, outSize * batchSize * sizeof(float));
    }

    float* forward(int* d_input) override;
    int loadWeights(float* weights) override;
};

struct ValueNetwork {
    const cudnnHandle_t& handle;

    const int batchSize;

    SparseLayer* featureTransformer;

    std::vector<DenseLayer*> layerStack;

    ValueNetwork(const cudnnHandle_t& hndl, int bs) : handle(hndl), batchSize(bs) {
        featureTransformer = new DensifyLayer(handle, batchSize, 32, 768);

        // CONVOLUTION_2D(12, 64) RELU CONVOLUTION_2D(64, 64) RELU CONVOLUTION_2D(64, 1) RELU FULLY_CONNECTED(64, 1)
        layerStack.push_back(new ConvLayer(handle, batchSize, 12, 64, 3, 3, 8, 8));
        layerStack.push_back(new ConvLayer(handle, batchSize, 64, 64, 3, 3, 8, 8));
        layerStack.push_back(new ConvLayer(handle, batchSize, 64,  1, 3, 3, 8, 8));
        layerStack.push_back(new FullyConnectedLayerCUDA(handle, batchSize, 64, 1));
    }

    float* forward(int* d_input) {
        float* curr = featureTransformer->forward(d_input);

        for (DenseLayer* l : layerStack)
            curr = l->forward(curr);

        return curr;
    }

    void loadWeights(float* weights) {
        weights += featureTransformer->loadWeights(weights);

        for (DenseLayer* l : layerStack)
            weights += l->loadWeights(weights);
    }
};

struct PolicyNetwork {
    const cudnnHandle_t& handle;

    const int batchSize;

    SparseLayer* featureTransformer;

    std::vector<DenseLayer*> layerStack;

    MaskedLayer* policyMaskingLayer;

    PolicyNetwork(const cudnnHandle_t& hndl, int bs) : handle(hndl), batchSize(bs) {
        featureTransformer = new SparseInFullyConnectedLayer(handle, batchSize, 768, 256);
        policyMaskingLayer = new MaskedFullyConnectedLayer(handle, batchSize, 256, 4096);
    }

    float* forward(int* d_input, int* d_mask) {
        float* curr = featureTransformer->forward(d_input);

        for (DenseLayer* l : layerStack)
            curr = l->forward(curr);

        curr = policyMaskingLayer->forward(curr, d_mask);

        return curr;
    }

    void loadWeights(float* weights) {
        weights += featureTransformer->loadWeights(weights);

        for (DenseLayer* l : layerStack)
            weights += l->loadWeights(weights);

        policyMaskingLayer->loadWeights(weights);
    }
};

struct DualNetwork {
    cudnnHandle_t valueHandle;
    cudnnHandle_t policyHandle;

    cudaStream_t valueStream;
    cudaStream_t policyStream;

    ValueNetwork valueNet;
    PolicyNetwork policyNet;

    int* d_input, *d_policyMask;

    const int batchSize;

    DualNetwork(int bs, float* policyWeights, float* valueWeights) : batchSize(bs), valueNet(ValueNetwork(valueHandle, bs)), policyNet(PolicyNetwork(policyHandle, bs)) {
        cudnnCreate(&policyHandle);
        cudnnCreate(&valueHandle);

        cudaStreamCreate(&valueStream);
        cudaStreamCreate(&policyStream);

        cudnnSetStream(policyHandle, policyStream);
        cudnnSetStream(valueHandle , valueStream );

        cudaMalloc(&d_input, 32 * sizeof(int) * batchSize);
        cudaMalloc(&d_policyMask, 218 * sizeof(int) * batchSize);

        valueNet.loadWeights(valueWeights);
        policyNet.loadWeights(policyWeights);
    }

    void forward(int* inputIndices, int* policyOutputIndices, float* valueOutput, float* policyOutput);
};

inline int ceildiv(int n, int m) {
    return (n + m - 1) / m;
}
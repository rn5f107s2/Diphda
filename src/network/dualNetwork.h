#pragma once

#include <cudnn.h>
#include <vector>

#include "layer/layerTypes.h"

#include "layer/conv.h"
#include "layer/fullyConnectedSimple.h"
#include "layer/densify.h"
#include "layer/fullyConnectedSparseIn.h"
#include "layer/fullyConnectedMasked.h"

class ValueNetwork {
    const cudnnHandle_t& handle;

    const int batchSize;

    SparseLayer* featureTransformer;

    std::vector<DenseLayer*> layerStack;

public:
    ValueNetwork(const cudnnHandle_t& hndl, int bs);

    float* forward(int* d_input);
    void loadWeights(float* weights);
};

class PolicyNetwork {
    const cudnnHandle_t& handle;

    const int batchSize;

    SparseLayer* featureTransformer;

    std::vector<DenseLayer*> layerStack;

    MaskedLayer* policyMaskingLayer;

public:
    PolicyNetwork(const cudnnHandle_t& hndl, int bs);

    float* forward(int* d_input, int* d_mask);
    void loadWeights(float* weights);
};

class DualNetwork {
    cudnnHandle_t valueHandle;
    cudnnHandle_t policyHandle;

    cudaStream_t valueStream;
    cudaStream_t policyStream;

    ValueNetwork valueNet;
    PolicyNetwork policyNet;

    int* d_input, *d_policyMask;

    const int batchSize;

public:
    DualNetwork(int bs, float* policyWeights, float* valueWeights);

    void forward(int* inputIndices, int* policyOutputIndices, float* valueOutput, float* policyOutput);
};
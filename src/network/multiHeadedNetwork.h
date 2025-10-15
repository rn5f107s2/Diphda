#pragma once

#include <cudnn.h>
#include <vector>

#include "fp16layer/layerTypes.h"
#include "fp16layer/conv.h"
#include "fp16layer/fullyConnectedSimple.h"
#include "fp16layer/a0block.h"
#include "fp16layer/fullyConnectedMasked.h"
#include "fp16layer/densify.h"

class ValueHead {
    const cudnnHandle_t& handle;

    const int batchSize;

    std::vector<DenseLayer*> layerStack;

    ValueOutput* ol;
 
public:
    ValueHead(const cudnnHandle_t& hndl, int bs);

    float* forward(__half* d_input);
    float* loadWeights(float* weights);
};

class PolicyHead {
    const cudnnHandle_t& handle;

    const int batchSize;

    std::vector<DenseLayer*> layerStack;

    MaskedLayer* policyMaskingLayer;

public:
    PolicyHead(const cudnnHandle_t& hndl, int bs);

    float* forward(__half* d_input, int* d_mask);
    float* loadWeights(float* weights);
};

class MultiHeadedNetwork {
    SparseLayer* featureTransformer;

    std::vector<DenseLayer*> layerStack;

    cudnnHandle_t valueHandle;
    cudnnHandle_t policyHandle;

    cudaStream_t valueStream;
    cudaStream_t policyStream;

    ValueHead* valueHead;
    PolicyHead* policyHead;

    int* d_input, *d_policyMask;

    const int batchSize;

public:
    MultiHeadedNetwork(int bs, float* weights);

    void forward(int* inputIndices, int* policyOutputIndices, float* valueOutput, float* policyOutput);
};
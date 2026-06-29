#pragma once

#include <cudnn.h>

#include "layerTypes.h"

class MaskedFullyConnectedLayer : public MaskedLayer {
    const cudnnHandle_t& handle;

    const int batchSize, inSize, outSize;

    float* d_weights, *d_biases, *d_output;

public:
    MaskedFullyConnectedLayer(const cudnnHandle_t& hndl, int bs, int is, int os);

    float* forward(float* d_input, int* d_mask) override;
    int loadWeights(float* weights) override;
};
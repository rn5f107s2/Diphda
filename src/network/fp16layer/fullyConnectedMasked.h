#pragma once

#include <cudnn.h>

#include "layerTypes.h"

class MaskedFullyConnectedLayer : public MaskedLayer {
    const cudnnHandle_t& handle;

    const int batchSize, inSize, outSize;

    __half* d_weights, *d_biases;
    float *d_output;

public:
    MaskedFullyConnectedLayer(const cudnnHandle_t& hndl, int bs, int is, int os);

    float* forward(__half* d_input, int* d_mask) override;
    int loadWeights(float* weights) override;
};
#pragma once

#include <cudnn.h>

#include "layerTypes.h"

class FullyConnectedLayerSimple : public ValueOutput {
    const cudnnHandle_t& handle;

    const int in, out, batchSize;

    __half* d_weights, *d_biases;
    float *d_output;

public:
    FullyConnectedLayerSimple(const cudnnHandle_t& hndl, int bs, int inSize, int outSize);

    float* forward(__half* d_input) override;
    int loadWeights(float* weights) override;
};
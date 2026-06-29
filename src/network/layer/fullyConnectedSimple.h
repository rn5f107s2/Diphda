#pragma once

#include <cudnn.h>

#include "layerTypes.h"

class FullyConnectedLayerSimple : public DenseLayer {
    const cudnnHandle_t& handle;

    const int in, out, batchSize;

    float* d_weights, *d_biases, *d_output;

public:
    FullyConnectedLayerSimple(const cudnnHandle_t& hndl, int bs, int inSize, int outSize);

    float* forward(float* d_input) override;
    int loadWeights(float* weights) override;
};
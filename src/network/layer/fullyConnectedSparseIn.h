#pragma once

#include "layerTypes.h"

#include <cudnn.h>

class SparseInFullyConnectedLayer : public SparseLayer {
    const cudnnHandle_t& handle;

    const int batchSize, inSize, outSize;

    float* d_weights, *d_biases, *d_output;

public:
    SparseInFullyConnectedLayer(const cudnnHandle_t& hndl, int bs, int is, int os);

    float* forward(int* d_input) override;
    int loadWeights(float* weights) override;
};
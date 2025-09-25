#pragma once

#include "conv.h"

#include <cudnn.h>

class FullyConnectedLayer : public DenseLayer {
    ConvLayer cl;

    int in, out;

public:
    FullyConnectedLayer(const cudnnHandle_t& hndl, int bs, int inSize, int outSize, bool activate = true);

    float* forward(float* d_input) override;
    int loadWeights(float* weights) override;
};
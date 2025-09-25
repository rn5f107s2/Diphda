#pragma once

#include <cudnn.h>

#include "layerTypes.h"

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
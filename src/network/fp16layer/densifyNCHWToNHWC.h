#pragma once

#include <cudnn.h>
#include <cuda_fp16.h>

#include "layerTypes.h"

class DensifyNCHWToNHWCLayer : public SparseLayer {
    const cudnnHandle_t& handle;

    const int batchSize, inSize, channels, height, width;

    __half* d_output;

public:
    DensifyNCHWToNHWCLayer(const cudnnHandle_t& hndl, int bs, int in, int c, int h, int w) : handle(hndl), batchSize(bs), inSize(in), channels(c), height(h), width(w) {
        cudaMalloc(&d_output, channels * height * width * batchSize * sizeof(__half));
    }

    __half* forward(int* d_input) override;
    int loadWeights(float* weights) override;
};
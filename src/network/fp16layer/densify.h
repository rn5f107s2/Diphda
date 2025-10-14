#pragma once

#include <cudnn.h>
#include <cuda_fp16.h>

#include "layerTypes.h"

class DensifyLayer : public SparseLayer {
    const cudnnHandle_t& handle;

    const int batchSize, inSize, outSize;

    __half* d_output;

public:
    DensifyLayer(const cudnnHandle_t& hndl, int bs, int in, int out) : handle(hndl), batchSize(bs), inSize(in), outSize(out) {
        cudaMalloc(&d_output, outSize * batchSize * sizeof(__half));
    }

    __half* forward(int* d_input) override;
    int loadWeights(float* weights) override;
};
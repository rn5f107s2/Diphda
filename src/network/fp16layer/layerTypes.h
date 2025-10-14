#pragma once

#include <cuda_fp16.h>

class DenseLayer {
public:
    virtual __half* forward(__half* d_input) = 0;
    virtual int loadWeights(float* weights) = 0;
};

class SparseLayer {
public:
    virtual __half* forward(int* d_input) = 0;
    virtual int loadWeights(float* weights) = 0;
};

class MaskedLayer {
public:
    virtual float* forward(__half* d_input, int* d_mask) = 0;
    virtual int loadWeights(float* weights) = 0;
};

class ValueOutput {
public:
    virtual float* forward(__half* d_input) = 0;
    virtual int loadWeights(float* weights) = 0; 
};
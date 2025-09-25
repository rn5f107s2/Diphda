#pragma once

class DenseLayer {
public:
    virtual float* forward(float* d_input) = 0;
    virtual int loadWeights(float* weights) = 0;
};

class SparseLayer {
public:
    virtual float* forward(int* d_input) = 0;
    virtual int loadWeights(float* weights) = 0;
};

class MaskedLayer {
public:
    virtual float* forward(float* d_input, int* d_mask) = 0;
    virtual int loadWeights(float* weights) = 0;
};
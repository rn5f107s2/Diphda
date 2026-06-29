#pragma once

#include <cudnn.h>

#include "layerTypes.h"

class A0Block : public DenseLayer {
    const cudnnHandle_t& handle;

    const int batchSize, channels, kernelWidth, kernelHeight, height, width;

    float* d_weightsc1, *d_weightsc2, *d_biasesc1, *d_biasesc2, *d_outputc1, *d_outputc2, *d_workspace;

    size_t workspaceSize;

    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

    cudnnTensorDescriptor_t inputDesc, outputDesc, biasDesc;
    cudnnFilterDescriptor_t kernelDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnActivationDescriptor_t actDesc;

public:
    A0Block(const cudnnHandle_t& hndl, int bs, int channels, int kw, int kh, int h, int w);

    float* forward(float* d_input) override;
    int loadWeights(float* weights) override;
};
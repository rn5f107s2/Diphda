#pragma once

#include <cudnn.h>

#include "layerTypes.h"

class ConvLayer : public DenseLayer {
    const cudnnHandle_t& handle;

    const int batchSize, inChannels, outChannels, kernelWidth, kernelHeight, height, width;

    __half* d_weights, *d_biases, *d_workspace, *d_output;

    size_t workspaceSize;

    cudnnTensorDescriptor_t inputDesc, outputDesc, biasDesc;
    cudnnFilterDescriptor_t kernelDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnActivationDescriptor_t actDesc;

    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

public:
    ConvLayer(const cudnnHandle_t& hndl, int bs, int ic, int oc, int kw, int kh, int h, int w, bool activate = true);

    __half* forward(__half* d_input) override;
    int loadWeights(float* weights) override;
};
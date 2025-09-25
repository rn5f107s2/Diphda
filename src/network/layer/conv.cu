#include "conv.h"

ConvLayer::ConvLayer(const cudnnHandle_t& hndl, int bs, int ic, int oc, int kw, int kh, int h, int w, bool activate) : handle(hndl),
                                                                                                                 batchSize(bs), 
                                                                                                                 inChannels(ic),
                                                                                                                 outChannels(oc),
                                                                                                                 kernelWidth(kw),
                                                                                                                 kernelHeight(kh),
                                                                                                                 height(h),
                                                                                                                 width(w) {
    cudnnCreateTensorDescriptor(&inputDesc);
    cudnnCreateTensorDescriptor(&outputDesc);
    cudnnCreateTensorDescriptor(&biasDesc);
    cudnnCreateFilterDescriptor(&kernelDesc);
    cudnnCreateConvolutionDescriptor(&convDesc);
    cudnnCreateActivationDescriptor(&actDesc);

    cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, inChannels, height, width);
    cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, outChannels, height, width);
    cudnnSetTensor4dDescriptor(biasDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, outChannels, 1, 1);
    cudnnSetFilter4dDescriptor(kernelDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, outChannels, inChannels, kernelHeight, kernelWidth);
    cudnnSetConvolution2dDescriptor(convDesc, kernelHeight / 2, kernelWidth / 2, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    cudnnSetActivationDescriptor(actDesc, activate ? CUDNN_ACTIVATION_RELU : CUDNN_ACTIVATION_IDENTITY, CUDNN_PROPAGATE_NAN, 0.0);

    cudnnGetConvolutionForwardWorkspaceSize(handle, inputDesc, kernelDesc, convDesc, outputDesc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, &workspaceSize);

    cudaMalloc(&d_workspace, workspaceSize);
    cudaMalloc(&d_output, outChannels * height * width * batchSize * sizeof(float));
    cudaMalloc(&d_biases, outChannels * sizeof(float));
    cudaMalloc(&d_weights, inChannels * outChannels * kernelHeight * kernelWidth * sizeof(float));
}

int ConvLayer::loadWeights(float* weights) {
    int nWeights = inChannels * outChannels * kernelHeight * kernelWidth;

    cudaMemcpy(d_weights, weights, nWeights * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, weights + nWeights, outChannels * sizeof(float), cudaMemcpyHostToDevice);

    return nWeights + outChannels;
}

float* ConvLayer::forward(float* d_input) {
    float alpha1 = 1.0f;
    float alpha2 = 0.0f;

    cudnnConvolutionBiasActivationForward(handle, &alpha1, 
                                          inputDesc, d_input, 
                                          kernelDesc, d_weights,
                                          convDesc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, 
                                          d_workspace, workspaceSize, 
                                          &alpha2, outputDesc, d_output, 
                                          biasDesc, d_biases, 
                                          actDesc, 
                                          outputDesc, d_output);

    return d_output;
}
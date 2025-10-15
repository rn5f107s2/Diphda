#include "conv.h"
#include "../util.h"

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

    cudnnTensorFormat_t layout = (outChannels == 8 && inChannels == 8) ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;

    cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, batchSize, inChannels, height, width);
    cudnnSetTensor4dDescriptor(outputDesc, layout, CUDNN_DATA_HALF, batchSize, outChannels, height, width);
    cudnnSetTensor4dDescriptor(biasDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, 1, outChannels, 1, 1);
    cudnnSetFilter4dDescriptor(kernelDesc, CUDNN_DATA_HALF, CUDNN_TENSOR_NHWC, outChannels, inChannels, kernelWidth, kernelHeight);

    cudnnSetConvolution2dDescriptor(convDesc, kernelHeight / 2, kernelWidth / 2, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_HALF);
    cudnnSetActivationDescriptor(actDesc, activate ? CUDNN_ACTIVATION_RELU : CUDNN_ACTIVATION_IDENTITY, CUDNN_PROPAGATE_NAN, 0.0f);

    cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH);

    cudaMalloc(&d_output, batchSize * outChannels * height * width * sizeof(__half));
    cudaMalloc(&d_weights, outChannels * inChannels * kernelHeight * kernelWidth * sizeof(__half));
    cudaMalloc(&d_biases, outChannels * sizeof(__half));

    cudnnGetConvolutionForwardWorkspaceSize(handle, inputDesc, kernelDesc, convDesc, outputDesc, algo, &workspaceSize);

    cudaMalloc(&d_workspace, workspaceSize);
    cudaMalloc(&d_output, outChannels * height * width * batchSize * sizeof(__half));
    cudaMalloc(&d_biases, outChannels * sizeof(__half));
    cudaMalloc(&d_weights, inChannels * outChannels * kernelHeight * kernelWidth * sizeof(__half));
}

int ConvLayer::loadWeights(float* weights) {
    int nWeights = inChannels * outChannels * kernelHeight * kernelWidth;

    copyConvertToDevice(d_weights, weights, nWeights);
    copyConvertToDevice(d_biases, weights + nWeights, outChannels);

    convertToOHWC(d_weights, outChannels, inChannels, kernelHeight, kernelWidth);

    return nWeights + outChannels;
}

__half* ConvLayer::forward(__half* d_input) {
    float alpha1 = 1.0f;
    float alpha2 = 0.0f;

    CHECK_CUDNN(cudnnConvolutionBiasActivationForward(handle, &alpha1, 
                                          inputDesc, d_input, 
                                          kernelDesc, d_weights,
                                          convDesc, algo, 
                                          d_workspace, workspaceSize, 
                                          &alpha2, outputDesc, d_output, 
                                          biasDesc, d_biases, 
                                          actDesc, 
                                          outputDesc, d_output));

    return d_output;
}

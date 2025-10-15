#include "a0block.h"
#include "../util.h"

A0Block::A0Block(const cudnnHandle_t& hndl, int bs, int c, int kw, int kh, int h, int w) : handle(hndl),
                                                                                           batchSize(bs),
                                                                                           channels(c),
                                                                                           kernelHeight(kh),
                                                                                           kernelWidth(kw),
                                                                                           height(h),
                                                                                           width(w) {
    cudnnCreateTensorDescriptor(&inputDesc);
    cudnnCreateTensorDescriptor(&outputDesc);
    cudnnCreateTensorDescriptor(&biasDesc);
    cudnnCreateFilterDescriptor(&kernelDesc);
    cudnnCreateConvolutionDescriptor(&convDesc);
    cudnnCreateActivationDescriptor(&actDesc);

    cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize, channels, height, width);
    cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize, channels, height, width);
    cudnnSetTensor4dDescriptor(biasDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, 1, channels, 1, 1);
    cudnnSetFilter4dDescriptor(kernelDesc, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW, channels, channels, kernelHeight, kernelWidth);
    cudnnSetConvolution2dDescriptor(convDesc, kernelHeight / 2, kernelWidth / 2, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_HALF);
    cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0);

    cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH);

    cudnnGetConvolutionForwardWorkspaceSize(handle, inputDesc, kernelDesc, convDesc, outputDesc, algo, &workspaceSize);

    cudaMalloc(&d_workspace, workspaceSize);
    cudaMalloc(&d_outputc1, channels * height * width * batchSize * sizeof(__half));
    cudaMalloc(&d_outputc2, channels * height * width * batchSize * sizeof(__half));
    cudaMalloc(&d_biasesc1, channels * sizeof(__half));
    cudaMalloc(&d_biasesc2, channels * sizeof(__half));
    cudaMalloc(&d_weightsc1, channels * channels * kernelHeight * kernelWidth * sizeof(__half)); 
    cudaMalloc(&d_weightsc2, channels * channels * kernelHeight * kernelWidth * sizeof(__half));                                                                                                                                         
}

int A0Block::loadWeights(float* weights) {
    int nWeightsPerConv = channels * channels * kernelHeight * kernelWidth;

    copyConvertToDevice(d_weightsc1, weights, nWeightsPerConv);
    copyConvertToDevice(d_biasesc1, weights + nWeightsPerConv, channels);
    copyConvertToDevice(d_weightsc2, weights + nWeightsPerConv + channels, nWeightsPerConv);
    copyConvertToDevice(d_biasesc2, weights + 2 * nWeightsPerConv + channels, channels);

    return 2 * (nWeightsPerConv + channels);
}

__half* A0Block::forward(__half* d_input) {
    float alpha1 = 1.0f;
    float alpha2 = 0.0f;

    CHECK_CUDNN(cudnnConvolutionBiasActivationForward(handle, &alpha1, 
                                          inputDesc, d_input, 
                                          kernelDesc, d_weightsc1,
                                          convDesc, algo, 
                                          d_workspace, workspaceSize, 
                                          &alpha2, outputDesc, d_outputc2, 
                                          biasDesc, d_biasesc1, 
                                          actDesc, 
                                          outputDesc, d_outputc1));

    float alpha2_secondConv = 1.0f;

    CHECK_CUDNN(cudnnConvolutionBiasActivationForward(handle, &alpha1, 
                                          outputDesc, d_outputc1, 
                                          kernelDesc, d_weightsc2,
                                          convDesc, algo, 
                                          d_workspace, workspaceSize, 
                                          &alpha2_secondConv, inputDesc, d_input, 
                                          biasDesc, d_biasesc2, 
                                          actDesc, 
                                          outputDesc, d_outputc2));

    return d_outputc2;
}
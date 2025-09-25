#include "multiHeadedNetwork.h"

ValueHead::ValueHead(const cudnnHandle_t& hndl, int bs) : handle(hndl), batchSize(bs) {
    // #define VALUE_HEAD CONVOLUTION_2D(8, 32) RELU CONVOLUTION_2D(32, 2) RELU FULLY_CONNECTED(128, 1)
    layerStack.push_back(new ConvLayer(handle, batchSize, 8, 32, 3, 3, 8, 8));
    layerStack.push_back(new ConvLayer(handle, batchSize, 32, 2, 3, 3, 8, 8));
    layerStack.push_back(new FullyConnectedLayerSimple(handle, batchSize, 128, 1));
}

float* ValueHead::forward(float* d_input) {
    float* curr = d_input;

    for (DenseLayer* l : layerStack)
        curr = l->forward(curr);

    return curr;
}

float* ValueHead::loadWeights(float* weights) {
    for (DenseLayer* l : layerStack)
        weights += l->loadWeights(weights);

    return weights;
}

float* PolicyHead::forward(float* d_input, int* d_mask) {
    float* curr = d_input;

    for (DenseLayer* l : layerStack)
        curr = l->forward(curr);

    curr = policyMaskingLayer->forward(curr, d_mask);

    return curr;
}

float* PolicyHead::loadWeights(float* weights) {
    for (DenseLayer* l : layerStack)
        weights += l->loadWeights(weights);

    weights += policyMaskingLayer->loadWeights(weights);

    return weights;
}

PolicyHead::PolicyHead(const cudnnHandle_t& hndl, int bs) : handle(hndl), batchSize(bs) {
    // #define POLICY_HEAD CONVOLUTION_2D(8, 8) RELU FULLY_CONNECTED(8 * 64, 4096)
    layerStack.push_back(new ConvLayer(handle, batchSize, 8, 8, 3, 3, 8, 8));
    policyMaskingLayer = new MaskedFullyConnectedLayer(handle, batchSize, 8 * 64, 4096);
}

MultiHeadedNetwork::MultiHeadedNetwork(int bs, float* weights) : batchSize(bs), valueHead(ValueHead(valueHandle, bs)), policyHead(PolicyHead(policyHandle, bs)) {
    cudnnCreate(&policyHandle);
    cudnnCreate(&valueHandle);

    cudaStreamCreate(&valueStream);
    cudaStreamCreate(&policyStream);

    cudnnSetStream(policyHandle, policyStream);
    cudnnSetStream(valueHandle , valueStream );

    cudaMalloc(&d_input, 32 * sizeof(int) * batchSize);
    cudaMalloc(&d_policyMask, 218 * sizeof(int) * batchSize);

    featureTransformer = new DensifyLayer(valueHandle, batchSize, 32, 768);

    layerStack.push_back(new ConvLayer(valueHandle, batchSize, 12, 8, 3, 3, 8, 8));

    for (int i = 0; i < 1; i++)
        layerStack.push_back(new A0Block(valueHandle, batchSize, 8, 3, 3, 8, 8));
    
    weights += featureTransformer->loadWeights(weights);

    for (DenseLayer* l : layerStack)
        weights += l->loadWeights(weights);

    weights = valueHead.loadWeights(weights);
    weights = policyHead.loadWeights(weights);
}

void MultiHeadedNetwork::forward(int* inputIndices, int* policyOutputIndices, float* valueOutput, float* policyOutput) {
    cudaMemcpy(d_input, inputIndices, sizeof(int) * batchSize * 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_policyMask, policyOutputIndices, sizeof(int) * batchSize * 218, cudaMemcpyHostToDevice);

    float* shared = featureTransformer->forward(d_input);

    for (DenseLayer* l : layerStack)
        shared = l->forward(shared);

    cudaDeviceSynchronize();

    float* v = valueHead.forward(shared);
    float* p = policyHead.forward(shared, d_policyMask);

    cudaStreamSynchronize(valueStream);
    cudaStreamSynchronize(policyStream);

    cudaMemcpy(valueOutput, v, batchSize * 1 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(policyOutput, p, batchSize * 218 * sizeof(float), cudaMemcpyDeviceToHost);
}
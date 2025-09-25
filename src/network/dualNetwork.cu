#include "dualNetwork.h"

ValueNetwork::ValueNetwork(const cudnnHandle_t& hndl, int bs) : handle(hndl), batchSize(bs) {
    featureTransformer = new SparseInFullyConnectedLayer(handle, batchSize, 768, 1024);
    layerStack.push_back(new FullyConnectedLayerSimple(handle, batchSize, 1024, 1));
}

float* ValueNetwork::forward(int* d_input) {
    float* curr = featureTransformer->forward(d_input);

    for (DenseLayer* l : layerStack)
        curr = l->forward(curr);

    return curr;
}

void ValueNetwork::loadWeights(float* weights) {
    weights += featureTransformer->loadWeights(weights);

    for (DenseLayer* l : layerStack)
        weights += l->loadWeights(weights);
}

PolicyNetwork::PolicyNetwork(const cudnnHandle_t& hndl, int bs) : handle(hndl), batchSize(bs) {
    featureTransformer = new SparseInFullyConnectedLayer(handle, batchSize, 768, 256);
    policyMaskingLayer = new MaskedFullyConnectedLayer(handle, batchSize, 256, 4096);
}

float* PolicyNetwork::forward(int* d_input, int* d_mask) {
    float* curr = featureTransformer->forward(d_input);

    for (DenseLayer* l : layerStack)
        curr = l->forward(curr);

    curr = policyMaskingLayer->forward(curr, d_mask);

    return curr;
}

void PolicyNetwork::loadWeights(float* weights) {
    weights += featureTransformer->loadWeights(weights);

    for (DenseLayer* l : layerStack)
        weights += l->loadWeights(weights);

    policyMaskingLayer->loadWeights(weights);
}

DualNetwork::DualNetwork(int bs, float* policyWeights, float* valueWeights) : batchSize(bs), valueNet(ValueNetwork(valueHandle, bs)), policyNet(PolicyNetwork(policyHandle, bs)) {
    cudnnCreate(&policyHandle);
    cudnnCreate(&valueHandle);

    cudaStreamCreate(&valueStream);
    cudaStreamCreate(&policyStream);

    cudnnSetStream(policyHandle, policyStream);
    cudnnSetStream(valueHandle , valueStream );

    cudaMalloc(&d_input, 32 * sizeof(int) * batchSize);
    cudaMalloc(&d_policyMask, 218 * sizeof(int) * batchSize);

    valueNet.loadWeights(valueWeights);
    policyNet.loadWeights(policyWeights);
}

void DualNetwork::forward(int* inputIndices, int* policyOutputIndices, float* valueOutput, float* policyOutput) {
    cudaMemcpy(d_input, inputIndices, sizeof(int) * batchSize * 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_policyMask, policyOutputIndices, sizeof(int) * batchSize * 218, cudaMemcpyHostToDevice);

    float* v = valueNet.forward(d_input);
    float* p = policyNet.forward(d_input, d_policyMask);

    cudaStreamSynchronize(valueStream);
    cudaStreamSynchronize(policyStream);

    cudaMemcpy(valueOutput, v, batchSize * 1 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(policyOutput, p, batchSize * 218 * sizeof(float), cudaMemcpyDeviceToHost);
}
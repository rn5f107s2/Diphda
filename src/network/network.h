#pragma once

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <fstream>
#include <iostream>

struct CudaNetwork {
    const int batchSize;

    const int valueLayer1Size = 1024;
    const int valueLayer2Size = 3;

    const int policyLayer1Size = 512;
    const int policyLayer2Size = 4096;

    const int maxMoves  = 218;
    const int maxInputs = 32;

    float *d_policyOutput, *d_valueOutput;
    float *d_valueIntermediate, *d_policyIntermediate;
    float *d_policyWeights;
    float *d_valueWeights;
    
    int* d_policyOutIndices;
    int* d_inputIndices;

    CudaNetwork(int batchSize, float* policyWeights, float* valueWeights);
    void forward(int* inputIndices, int* policyOutputIndices, float* valueOutput, float* policyOutput);
};

class Network {
public:
    Network(int bs) : batchSize(bs) {
        policyOutputBatched = (float*) malloc(sizeof(float) * 218              * batchSize);
        valueOutputBatched  = (float*) malloc(sizeof(float) * valueLayer2Size  * batchSize);
    }

    void forward(int* inputIndices, int* policyOutputIndices) {
        cudaNetwork->forward(inputIndices, policyOutputIndices, valueOutputBatched, policyOutputBatched);
    }

    float* getPolicy(int batchIdx) {
        return &policyOutputBatched[218 * batchIdx];
    }

    float* getValue(int batchIdx) {
        return &valueOutputBatched[valueLayer2Size * batchIdx];
    }

    void loadWeights(std::string filename) {
        int nValueWeights  = 768 * valueLayer1Size  + valueLayer1Size  + valueLayer1Size  * valueLayer2Size  + valueLayer2Size;
        int nPolicyWeights = 768 * policyLayer1Size + policyLayer1Size + policyLayer1Size * policyLayer2Size + policyLayer2Size;

        float* valueWeights  = (float*) malloc(nValueWeights  * sizeof(float));
        float* policyWeights = (float*) malloc(nPolicyWeights * sizeof(float));

        std::ifstream weights(filename);

        weights.read((char*) valueWeights , nValueWeights  * sizeof(float));
        weights.read((char*) policyWeights, nPolicyWeights * sizeof(float));

        if (cudaNetwork)
            delete cudaNetwork;

        cudaNetwork = new CudaNetwork(batchSize, policyWeights, valueWeights);

        free(policyWeights);
        free(valueWeights);
    }

private:
    const int batchSize;

    float* policyOutputBatched, *valueOutputBatched;

    const int valueLayer1Size = 1024;
    const int valueLayer2Size = 3;

    const int policyLayer1Size = 512;
    const int policyLayer2Size = 4096;

    CudaNetwork* cudaNetwork = nullptr;
};
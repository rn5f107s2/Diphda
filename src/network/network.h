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

    float *d_input, *d_policyOutput, *d_valueOutput;
    float *d_valueIntermediate, *d_policyIntermediate;
    float *d_policyWeights;
    float *d_valueWeights;

    CudaNetwork(int batchSize, float* policyWeights, float* valueWeights);
    void forward(float* input, float* valueOutput, float* policyOutput);
};

class Network {
public:
    Network() {
        policyOutputBatched = (float*) malloc(sizeof(float) * policyLayer2Size * batchSize);
        valueOutputBatched  = (float*) malloc(sizeof(float) * valueLayer2Size  * batchSize);
    }

    void forward() {
        float* input = (float*) malloc(batchSize * 768 * sizeof(float));

        for (int i = 0; i < 768; i++)
            input[i] = float(rand()) / float(RAND_MAX);

        cudaNetwork->forward(input, valueOutputBatched, policyOutputBatched);
    }

    float* getPolicy(int batchIdx) {
        return &policyOutputBatched[policyLayer2Size * batchIdx];
    }

    float* getVaue(int batchIdx) {
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

        for (int i = 0; i < nValueWeights; i++)
            valueWeights[i] = float(rand()) / float(RAND_MAX);

        for (int i = 0; i < nPolicyWeights; i++)
            policyWeights[i] = float(rand()) / float(RAND_MAX);

        if (cudaNetwork)
            delete cudaNetwork;

        cudaNetwork = new CudaNetwork(batchSize, policyWeights, valueWeights);

        free(policyWeights);
        free(valueWeights);
    }

private:
    const int batchSize = 1;

    float* policyOutputBatched, *valueOutputBatched;

    const int valueLayer1Size = 1024;
    const int valueLayer2Size = 3;

    const int policyLayer1Size = 512;
    const int policyLayer2Size = 4096;

    CudaNetwork* cudaNetwork = nullptr;
};
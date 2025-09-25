#pragma once

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <fstream>
#include <iostream>
#include <cuda_runtime.h>
#include <cudnn.h>

#include "multiHeadedNetwork.h"
#include "dualNetwork.h"

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

#ifdef MULTI_HEAD
    void loadWeights(std::string filename) {
        // Hm
        int nWeights  = 16903169;

        float* w  = (float*) malloc(nWeights  * sizeof(float));

        std::ifstream weights(filename);

        weights.read((char*) w, nWeights * sizeof(float));

        if (cudaNetwork)
            delete cudaNetwork;

        cudaNetwork = new MultiHeadedNetwork(batchSize, w);

        free(w);
    }
#else

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

        cudaNetwork = new DualNetwork(batchSize, policyWeights, valueWeights);

        free(policyWeights);
        free(valueWeights);
    }
#endif


private:
    const int batchSize;

    float* policyOutputBatched, *valueOutputBatched;

    const int valueLayer1Size = 1024;
    const int valueLayer2Size = 1;

    const int policyLayer1Size = 256;
    const int policyLayer2Size = 4096;

#ifndef MULTI_HEAD
    DualNetwork* cudaNetwork = nullptr;
#else
    MultiHeadedNetwork* cudaNetwork = nullptr;
#endif
};
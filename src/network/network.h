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

    const int policyLayer1Size = 256;
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
    Network() {
        policyOutputBatched = (float*) malloc(sizeof(float) * policyLayer2Size * batchSize);
        valueOutputBatched  = (float*) malloc(sizeof(float) * valueLayer2Size  * batchSize);
    }

    void forward() {
        int* input = (int*) malloc(batchSize * 32 * sizeof(int));

        bool used[768];

        for (int i = 0; i < 32 * batchSize; i++) {
            if (i % 32 == 0)
                memset(used, 0, 768);

            int r     = (float(rand()) / float(RAND_MAX)) * 768;
            int batch = i / 32;

            if (!used[r])
                input[(i % 32) + batch * 32] = r;

            used[r] = true;
        }

        int* policyOutputIndices = (int*) malloc(sizeof(int) * batchSize * 218);
        memset(policyOutputIndices, -1, sizeof(int) * 218);
        policyOutputIndices[0] = 215;
        policyOutputIndices[1] = 4032;
        policyOutputIndices[2] = 1045;

        for (int i = 1; i < batchSize; i++)
            memcpy(&policyOutputIndices[218 * i], policyOutputIndices, sizeof(int) * 218);

        cudaNetwork->forward(input, policyOutputIndices, valueOutputBatched, policyOutputBatched);

        free(input);
        free(policyOutputIndices);
    }

    float* getPolicy(int batchIdx) {
        return &policyOutputBatched[policyLayer2Size * batchIdx];
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
    const int batchSize = 2 << 16;

    float* policyOutputBatched, *valueOutputBatched;

    const int valueLayer1Size = 1024;
    const int valueLayer2Size = 3;

    const int policyLayer1Size = 512;
    const int policyLayer2Size = 4096;

    CudaNetwork* cudaNetwork = nullptr;
};
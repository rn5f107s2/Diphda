#pragma once

#include <vector>

#include "node.h"
#include "../network/network.h"

class Evaluator {
private:
    Network* net;

    const int batchSize;
    int* policyIndices;
    int* valueIndices;

    std::vector<Node*> nodes;

    void writeValueIndices(Position& pos);
    void writePolicyIndices(Position& pos, MoveList& ml);

public:
    void addNode(Position& pos, MoveList& ml, Node* node);
    void forward(float temperature = 1.0);

    Evaluator(int bs) : batchSize(bs) {
        policyIndices = (int*) malloc(218 * sizeof(int) * batchSize);
         valueIndices = (int*) malloc( 32 * sizeof(int) * batchSize);

        memset(policyIndices, -1, sizeof(218 * sizeof(int) * batchSize));

        net = new Network(batchSize);

        net->loadWeights("test256.bin");

        nodes.reserve(batchSize);
    }
};
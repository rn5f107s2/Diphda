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

    void forward(float temperature = 1.0) {
        net->forward(valueIndices, policyIndices);

        for (int i = 0; i < nodes.size(); i++) {
            float* value = net->getValue(i);

            //float w = std::exp(value[2] / 2);
            //float d = std::exp(value[1] / 2);
            //float l = std::exp(value[0] / 2);

            //float sum = w + d + l;

            //float q = w / sum - l / sum;

            float q = std::tanh(value[0] / 2);

            nodes[i]->virtualLoss(true);
            nodes[i]->backpropagate(-q);
            nodes[i]->labelPolicies(net->getPolicy(i), temperature);

            nodes[i]->info.waiting(false);
        }

        nodes.clear();
    }

    void writeValueIndices(Position& pos) {
        pos.toChess768Dense(valueIndices + 32 * nodes.size());
    }

    void writePolicyIndices(Position& pos, MoveList& ml){
        int* indices = policyIndices + 218 * nodes.size();

        for (int i = 0; i < ml.length(); i++)
            indices[i] = pos.indexOf(ml[i]);

        memset(indices + ml.length(), -1, (218 - ml.length()) * sizeof(int));
    }

public:
    void addNode(Position& pos, MoveList& ml, Node* node) {
        writeValueIndices(pos);
        writePolicyIndices(pos, ml);

        nodes.push_back(node);

        if (nodes.size() == batchSize)
            forward();
    }

    void distribute(float temperature = 1.0) {
        forward(temperature);
    }

    Evaluator(int bs) : batchSize(bs) {
        policyIndices = (int*) malloc(218 * sizeof(int) * batchSize);
         valueIndices = (int*) malloc( 32 * sizeof(int) * batchSize);

        memset(policyIndices, -1, sizeof(218 * sizeof(int) * batchSize));

        net = new Network(batchSize);

        net->loadWeights("test256.bin");

        nodes.reserve(batchSize);
    }
};
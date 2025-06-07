#include "node.h"
#include "evaluator.h"
#include "../network/network.h"

void Evaluator::forward(float temperature) {
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

void Evaluator::writeValueIndices(Position& pos) {
    pos.toChess768Dense(valueIndices + 32 * nodes.size());
}

void Evaluator::writePolicyIndices(Position& pos, MoveList& ml){
    int* indices = policyIndices + 218 * nodes.size();

    for (int i = 0; i < ml.length(); i++)
        indices[i] = pos.indexOf(ml[i]);

    memset(indices + ml.length(), -1, (218 - ml.length()) * sizeof(int));
}

void Evaluator::addNode(Position& pos, MoveList& ml, Node* node) {
    writeValueIndices(pos);
    writePolicyIndices(pos, ml);

    nodes.push_back(node);

    if (nodes.size() == batchSize)
        forward();
}
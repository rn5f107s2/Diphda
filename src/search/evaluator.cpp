#include "node.h"
#include "evaluator.h"
#include "../network/network.h"

#include <chrono>

void Evaluator::forwardInternal() {
    CollectedData& data = collector.getHalf(false);

    net->forward(data.inputIndices, data.policyIndices);

    for (int i = 0; i < data.idx; i++) {
        float* value = net->getValue(i);

        //float w = std::exp(value[2] / 2);
        //float d = std::exp(value[1] / 2);
        //float l = std::exp(value[0] / 2);

        //float sum = w + d + l;

        //float q = w / sum - l / sum;

        float q = std::tanh(value[0]);

        data.nodes[i]->virtualLoss(true);
        data.nodes[i]->backpropagate(-q);
        data.nodes[i]->labelPolicies(net->getPolicy(i), data.temperatures[i]);

        data.nodes[i]->info.waiting(false);
    }

    data.forwardEarly = false;

    data.clear();
}

void CollectedData::pushBack(Node* node, Position& pos, MoveList& ml, float temp) {
    nodes       [idx] = node;
    temperatures[idx] = temp;

    writeInputIndices(pos);
    writePolicyIndices(pos, ml);

    idx++;
}

void CollectedData::writeInputIndices(Position& pos) {
    pos.toChess768Dense(inputIndices + 32 * idx);
}

void CollectedData::writePolicyIndices(Position& pos, MoveList& ml){
    int* indices = policyIndices + 218 * idx;

    for (int i = 0; i < ml.length(); i++)
        indices[i] = pos.indexOf(ml[i]);

    memset(indices + ml.length(), -1, (218 - ml.length()) * sizeof(int));
}

CollectedData& Collector::getHalf(bool active) {
    return active ? data[activeHalf] : data[!activeHalf];
}

void Collector::addNode(Node* node, Position& pos, MoveList& ml, float temp) {
    node->info.waiting(true);
    node->virtualLoss(false);

    data[activeHalf].pushBack(node, pos, ml, temp);
}

void Evaluator::forward(float temperature) {
    std::unique_lock<std::mutex> lk(mtx);
    cv.wait(lk, [&] { return !evaluating; } );

    collector.switchActive();
    batchReady = evaluating = true;

    lk.unlock();
    cv.notify_one();
}

void Evaluator::forwardBlocking(float temperature) {
    forward(temperature);
    std::unique_lock<std::mutex> lk(mtx);
    cv.wait(lk, [&] { return !evaluating; } );
}


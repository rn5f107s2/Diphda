#include "node.h"
#include "evaluator.h"
#include "../network/network.h"

#include <chrono>

void Evaluator::forwardInternal() {
    bool half = !activeHalf.load(std::memory_order_relaxed);

    net->forward(valueIndices[half], policyIndices[half]);

    for (int i = 0; i < nodes[half].size(); i++) {
        float* value = net->getValue(i);

        //float w = std::exp(value[2] / 2);
        //float d = std::exp(value[1] / 2);
        //float l = std::exp(value[0] / 2);

        //float sum = w + d + l;

        //float q = w / sum - l / sum;

        float q = std::tanh(value[0] / 2);

        nodes[half][i]->virtualLoss(true);
        nodes[half][i]->backpropagate(-q);
        nodes[half][i]->labelPolicies(net->getPolicy(i), temp);

        nodes[half][i]->info.waiting(false);
    }

    nodes[half].clear();
}

void Evaluator::writeValueIndices(Position& pos, bool half) {
    pos.toChess768Dense(valueIndices[half] + 32 * nodes[half].size());
}

void Evaluator::writePolicyIndices(Position& pos, MoveList& ml, bool half){
    int* indices = policyIndices[half] + 218 * nodes[half].size();

    for (int i = 0; i < ml.length(); i++)
        indices[i] = pos.indexOf(ml[i]);

    memset(indices + ml.length(), -1, (218 - ml.length()) * sizeof(int));
}

void Evaluator::addNode(Position& pos, MoveList& ml, Node* node) {
    bool half = activeHalf.load(std::memory_order_relaxed);

    node->info.waiting(true);
    node->virtualLoss(false);

    writeValueIndices(pos, half);
    writePolicyIndices(pos, ml, half);

    nodes[half].push_back(node);

    if (nodes[half].size() == batchSize)
        forward();
}

void Evaluator::forward(float temperature) {
    std::unique_lock<std::mutex> lk(mtx);
    cv.wait(lk, [&] { return !evaluating; } );

    activeHalf.fetch_xor(1, std::memory_order_relaxed);
    batchReady = evaluating = true;
    temp = temperature;

    lk.unlock();
    cv.notify_one();
}

void Evaluator::forwardBlocking(float temperature) {
    forward(temperature);
    std::unique_lock<std::mutex> lk(mtx);
    cv.wait(lk, [&] { return !evaluating; } );
}


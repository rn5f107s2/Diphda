#pragma once

#include <vector>
#include <atomic>
#include <mutex>
#include <thread>
#include <condition_variable>

#include "node.h"
#include "../network/network.h"

class Evaluator {
private:
    Network* net;

    const int batchSize;
    std::array<int*, 2> policyIndices;
    std::array<int*, 2> valueIndices;

    std::atomic<int8_t> activeHalf;

    std::array<std::vector<Node*>, 2> nodes;

    std::mutex mtx;
    std::condition_variable cv;

    std::thread evaluationThread;

    bool batchReady;
    bool evaluating;

    float temp;

    void writeValueIndices(Position& pos, bool half);
    void writePolicyIndices(Position& pos, MoveList& ml, bool half);
    void forwardInternal();

public:
    void addNode(Position& pos, MoveList& ml, Node* node);
    void forward(float temperature = 1.0f);
    void forwardBlocking(float temperature = 1.0f);

    Evaluator(int bs) : batchSize(bs) {
        policyIndices[0] = (int*) malloc(218 * sizeof(int) * batchSize);
        policyIndices[1] = (int*) malloc(218 * sizeof(int) * batchSize);
         valueIndices[0] = (int*) malloc( 32 * sizeof(int) * batchSize);
         valueIndices[1] = (int*) malloc( 32 * sizeof(int) * batchSize);

        memset(policyIndices[0], -1, sizeof(218 * sizeof(int) * batchSize));
        memset(policyIndices[1], -1, sizeof(218 * sizeof(int) * batchSize));

        net = new Network(batchSize);

        net->loadWeights("test256.bin");

        activeHalf = evaluating = batchReady = 0;
        temp = 1.0f;

        nodes[0].reserve(batchSize);
        nodes[1].reserve(batchSize);

        evaluationThread = std::thread(
            [&] {
                while (true) {
                    std::unique_lock<std::mutex> lk(mtx);
                    cv.wait(lk, [&] { return batchReady; } );

                    forwardInternal();

                    evaluating = batchReady = false;

                    lk.unlock();
                    cv.notify_one();
                }
            }
        );
    }
};
#pragma once

#include <vector>
#include <atomic>
#include <mutex>
#include <thread>
#include <condition_variable>

#include "node.h"
#include "../network/network.h"

extern std::string defaultEvalFile;

struct CollectedData {
    int* inputIndices;
    int* policyIndices;
    float* temperatures;

    Node** nodes;

    std::mutex mtx;

    volatile int idx = 0;
    const int batchSize;
    bool forwardEarly = false;

    CollectedData(int bs) : batchSize(bs) {
        inputIndices  = (int*  ) malloc(batchSize * 32  * sizeof(int  ));
        policyIndices = (int*  ) malloc(batchSize * 218 * sizeof(int  ));
        temperatures  = (float*) malloc(batchSize *       sizeof(float));

        nodes = (Node**) malloc(batchSize * sizeof(Node*));

        memset(policyIndices, -1, 218 * sizeof(int  ) * batchSize);
        memset(inputIndices , -1, 32  * sizeof(int  ) * batchSize);
        memset(temperatures ,  0,       sizeof(float) * batchSize);
    }

    ~CollectedData() {
        free(inputIndices);
        free(policyIndices);
        free(temperatures);

        free(nodes);
    }

    void writeInputIndices(Position& pos, int index);
    void writePolicyIndices(Position& pos, MoveList& ml, int index);
    void pushBack(Node* node, Position& pos, MoveList& ml, float temp);

    void clear() {
        idx = 0;
    }

    bool isFull() {
        return idx >= batchSize;
    }
};

class Collector {
private:
    std::array<CollectedData, 2> data;

    bool activeHalf = 0;

public:
    Collector(int batchSize) : data({ CollectedData(batchSize), CollectedData(batchSize) }) {}

    CollectedData& getHalf(bool active);
    void addNode(Node* node, Position& pos, MoveList& ml, float temp);

    void switchActive() {
        activeHalf = !activeHalf;
    }

    void earlyFull() {
        data[activeHalf].forwardEarly = true;
    }
};

class Evaluator {
private:
    Network* net;

    const int batchSize;

    std::mutex mtx;
    std::condition_variable cv;

    std::thread evaluationThread;

    Collector collector;

    bool batchReady = false;
    bool evaluating = false;

    void forwardInternal();

public:
    void forward();
    void forwardBlocking();

    Collector& getCollector() {
        return collector;
    }

    Evaluator(int bs) : batchSize(bs), collector(Collector(bs)) {
        net = new Network(batchSize);

        net->loadWeights(defaultEvalFile);

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
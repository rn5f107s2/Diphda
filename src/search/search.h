#pragma once

#include <cstdint>
#include <chrono>
#include <functional>

#include "node.h"
#include "evaluator.h"
#include "parameters.h"
#include "../games/game.h"
#include "../network/network.h"

struct SearchTime;

class Searcher {
public:
    void search(Position& pos, SearchTime& st);
    void clear();

    Searcher() {
        evaluator = new Evaluator(100);
    }

    OptionsContainer& getOptions() {
        return params.getOptions();
    }

    void updateOptions() {
        params.update();
    }

private:
    bool     priorPosExists = false;
    Position priorPos;
    Node*    root = nullptr;

    void prepareNewRoot(Position& pos);
    Node* findNewRoot(Position& pos);
    Node* createNewRoot();

    void sendInfo(uint64_t nodes, std::chrono::steady_clock::time_point& begin);
    std::string getPv(Node* n, std::function<double(Node&)> func);

    Node* selectBest(Node* n, std::function<double(Node&)> func);

    Evaluator* evaluator;

    SearchParameters params;

    std::vector<Node*> disjunctSubtrees;
};

struct SearchTime {
    std::chrono::steady_clock::time_point searchBegin;

    uint64_t time      = std::numeric_limits<uint64_t>::max();
    uint64_t increment = std::numeric_limits<uint64_t>::max();

    int nodesLimit = std::numeric_limits<int>::max();
    int depthLimit = std::numeric_limits<int>::max();

    int movesToGo = std::numeric_limits<int>::max();

    SearchTime() {
        searchBegin = std::chrono::steady_clock::now();
    }
};

static const uint64_t MOVE_OVERHEAD = 10;

inline bool shouldStop(SearchTime& st, uint64_t visits) {
    if (visits >= st.nodesLimit)
        return true;

    if (visits & 511)
        return false;

    uint64_t elapsed       = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - st.searchBegin).count();
    uint64_t allocatedTime = std::max(st.time / st.movesToGo, uint64_t((st.time * 0.05) + (st.increment * 0.5))) - MOVE_OVERHEAD;

    if (elapsed >= allocatedTime)
        return true;

    return false;
}
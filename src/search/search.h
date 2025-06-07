#pragma once

#include <cstdint>

#include "node.h"
#include "evaluator.h"
#include "../games/game.h"
#include "../network/network.h"

class Searcher {
public:
    void search(Position& pos);
    void clear();

    Searcher() {
        evaluator = new Evaluator(100);
    }

private:
    bool     priorPosExists = false;
    Position priorPos;
    Node*    root = nullptr;

    Node* findNewRoot(Position& pos);
    Node* createNewRoot();

    Evaluator* evaluator;

    std::vector<Node*> disjunctSubtrees;
};
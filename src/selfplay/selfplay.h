#pragma once

#include "../search/searchparams.h"
#include "../search/search.h"

class SelfplaySearcher {
private:
    Evaluator* evaluator;

    int  nodes     = 0;
    bool rootReady = false;

    Move played;

    Node* root;

    Position pos;

    SearchParameters params;

    void cleanupRoot();

public:
    SelfplaySearcher(Evaluator* eval) : evaluator(eval) {
        pos.setStartpos();
    }

    void addSingle();
    void doPlayout();
    void prepareRoot();
    void playMove(Move m);
    void startNewGame();

    bool isTerminal();

    int currentSearchNodes() {
        return nodes;
    }

    Node* getRoot() {
        return root;
    }

    Position& getPos() {
        return pos;
    }
};
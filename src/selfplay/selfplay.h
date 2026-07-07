#pragma once

#include "../search/searchparams.h"
#include "../search/search.h"
#include "selfplayparams.h"

class SelfplaySearcher {
private:
    Evaluator* evaluator;

    int  nodes     = 0;
    int  moveCount = 0;
    bool rootReady = false;
    bool dirichlet = false;

    Move played;

    Node* root;

    Position pos;

    SearchParameters params;
    SelfplayParmeters selfplayParams;

    void cleanupRoot();
    void doPlayout();
    void prepareRoot();
    void applyDirichlet();
    float getTemperature();

    Node* chooseBestTerminal();

public:
    SelfplaySearcher(Evaluator* eval) : evaluator(eval) {
        startNewGame();
    }

    void addSingle(int samplesPerGame);
    void playMove(Move m);
    void startNewGame();

    bool isTerminal();

    Node* chooseAction();

    bool ww() {
        return (moveCount & 1) ? pos.isLost() : pos.isWon();
    }

    bool d() {
        return pos.isDrawn();
    }

    bool wl() {
        return (moveCount & 1) ? pos.isWon() : pos.isDrawn();
    }

    bool shouldStop() {
        return nodes >= selfplayParams.playouts;
    }

    Node* getRoot() {
        return root;
    }

    Position& getPos() {
        return pos;
    }
};

inline float lerp(float a, float b, float epsilon) {
    return a * (1 - epsilon) + b * epsilon;
}

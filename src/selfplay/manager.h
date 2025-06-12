#pragma once

#include "selfplay.h"

class SelfplayManager {
private:
    Evaluator* eval;
    SelfplaySearcher s;

public:
    SelfplayManager() : eval(new Evaluator(2)), s(eval) {}

    void play() {
        std::cout << "Starting game!" << std::endl;

        while (!s.isTerminal()) {
            if (s.currentSearchNodes() >= 5000) {
                Move bm = s.getRoot()->select(0)->getMove();

                std::cout << s.getPos().toString() << std::endl;
                std::cout << bm.toString() << std::endl;

                s.playMove(bm);
            }   

            s.addSingle();

            eval->forwardBlocking();
        }
    }
};
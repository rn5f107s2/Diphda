#include <iostream>
#include <chrono>

#include "uci.h"
#include "games/chess/attacks.h"
#include "games/chess/position.h"
#include "network/network.h"
#include "selfplay/manager.h"

void start(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "selfplay") {
        SelfplayManager sm; sm.run();
    }

    UCIHandler* uci = new UCIHandler();
    uci->start(argc, argv);
    delete uci;
}

int main(int argc, char** argv) {
    Chess::Attacks::init();

    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "-w") {
            defaultEvalFile = argv[++i];
        }
    }

    start(argc, argv);
}
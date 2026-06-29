#include <iostream>
#include <chrono>

#include "uci.h"
#include "games/chess/attacks.h"
#include "games/chess/position.h"
#include "network/network.h"
#include "selfplay/manager.h"

int main(int argc, char** argv) {
    Chess::Attacks::init();

    SelfplayManager sm; sm.run();

    UCIHandler* uci = new UCIHandler();
    uci->start(argc, argv);
    delete uci;
}
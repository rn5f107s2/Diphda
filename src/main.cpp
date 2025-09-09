#include <iostream>
#include <chrono>

#include "uci.h"
#include "games/chess/attacks.h"
#include "games/chess/position.h"
#include "network/network.h"
#include "network/cudnn.h"

int main(int argc, char** argv) {
    Chess::Attacks::init();

    UCIHandler* uci = new UCIHandler();
    uci->start(argc, argv);
    delete uci;
}
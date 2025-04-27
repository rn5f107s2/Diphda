#include <iostream>

#include "uci.h"
#include "games/chess/attacks.h"
#include "network/network.h"

int main(int argc, char** argv) {
    Chess::Attacks::init();

    Network* n = new Network();
    n->loadWeights("/dev/urandom");
    n->forward();

    UCIHandler* uci = new UCIHandler();
    uci->start(argc, argv);
    delete uci;
}
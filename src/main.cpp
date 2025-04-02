#include <iostream>

#include "uci.h"
#include "games/chess/attacks.h"

int main(int argc, char** argv) {
    Chess::Attacks::init();

    UCIHandler* uci = new UCIHandler();
    uci->start(argc, argv);
    delete uci;
}
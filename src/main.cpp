#include <iostream>

#include "uci.h"
#include "attacks.h"
#include "movegen.h"
#include "position.h"
#include  "perft.h"

int main(int argc, char** argv) {
    Attacks::init();

    UCIHandler* uci = new UCIHandler();
    uci->start(argc, argv);
    delete uci;
}
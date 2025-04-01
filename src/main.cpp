#include <iostream>

#include "uci.h"
#include "attacks.h"
#include "movegen.h"
#include "position.h"
#include  "perft.h"

int main(int argc, char** argv) {
    Attacks::init();

    MoveList ml;
    Position pos;
    pos.setPosition("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    
    perft<true>(pos, 5);

    UCIHandler* uci = new UCIHandler();
    uci->start(argc, argv);
    delete uci;
}
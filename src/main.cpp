#include <iostream>

#include "uci.h"
#include "attacks.h"
#include "movegen.h"

int main(int argc, char** argv) {
    Attacks::init();

    MoveList ml;
    Position pos;
    pos.setPosition("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    generateMoves(pos, ml);

    UCIHandler* uci = new UCIHandler();
    uci->start(argc, argv);
    delete uci;
}
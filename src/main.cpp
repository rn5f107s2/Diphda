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
    pos.setPosition("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8");
    
    perft<true>(pos, 5);

    UCIHandler* uci = new UCIHandler();
    uci->start(argc, argv);
    delete uci;
}
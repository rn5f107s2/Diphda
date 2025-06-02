#include "search.h"

#include <cmath>
#include <chrono>

void search(Position& pos) {
    MoveList ml;
    pos.generateMoves(ml);

    std::cout << "info depth 1 score cp " << rand() / 3000000 - 300 << std::endl;
    std::cout << "bestmove " << ml[rand() % ml.length()].toString() << std::endl;
}
#pragma once

#include "position.h"
#include "move.h"

#include <iostream>

class MoveList {
public:
    void pushBack(Move move) {
        std::cout << "Added " << move.toString() << std::endl;
        moves[index++] = move;
    }

private:
    std::array<Move, 256> moves;
    int index = 0;
};
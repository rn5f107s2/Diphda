#pragma once

#include <iostream>
#include <array>

#include "move.h"

class MoveList {
public:
    void pushBack(Move move) {
        moves[position++] = move;
    }

    auto begin() {
        return moves.begin();
    }

    auto end() {
        return moves.begin() + position;
    }

    size_t length() {
        return position;
    }

private:
    std::array<Move, 256> moves;
    int position = 0;
};
#pragma once

#include <iostream>
#include <array>

#include "chess/move.h"

class MoveList {
public:
    void pushBack(Chess::Move move) {
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
    std::array<Chess::Move, 256> moves;
    int position = 0;
};
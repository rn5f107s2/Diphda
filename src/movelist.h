#pragma once

#include <iostream>
#include <array>

#include "games/gamemove.h"

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

    Move operator[](int i) { return moves[i]; }

private:
    std::array<Move, Move::MAX_LEGAL> moves;
    int position = 0;
};
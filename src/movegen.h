#pragma once

#include "movelist.h"

class Position;

namespace Movegen {

void generateMoves(const Position &pos, MoveList &ml);

}

inline void generateMoves(const Position &pos, MoveList &ml) {
    return Movegen::generateMoves(pos, ml);
}
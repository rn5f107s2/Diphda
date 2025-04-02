#pragma once

#include "../movelist.h"

namespace Chess {

class Position;

namespace Movegen {

void generateMoves(const Position &pos, MoveList &ml);

} // Namespace Movegen

inline void generateMoves(const Position &pos, MoveList &ml) {
    return Movegen::generateMoves(pos, ml);
}

} // Namespace Chess
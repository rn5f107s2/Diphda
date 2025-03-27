#pragma once

#include "position.h"
#include "movelist.h"

namespace Movegen {

void generateMoves(Position &pos, MoveList &ml);

}

inline void generateMoves(Position &pos, MoveList &ml) {
    return Movegen::generateMoves(pos, ml);
}
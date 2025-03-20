#pragma once

#include "types.h"

#include <array>

namespace Attacks {

const extern std::array<Bitboard, Square::NONE> knightAttacks;
const extern std::array<Bitboard, Square::NONE> kingAttacks;

}

inline Bitboard getKnightAttacks(Square square) {
    return Attacks::knightAttacks[int(square)];
}

inline Bitboard getKingAttacks(Square square) {
    return Attacks::kingAttacks[int(square)];
}
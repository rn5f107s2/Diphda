#pragma once

#include "types.h"

#include <array>

namespace Attacks {
    extern std::array<Bitboard, Square::NONE> knightAttacks;
    extern std::array<Bitboard, Square::NONE> kingAttacks;

    extern std::array<Bitboard, Square::COUNT> bishopMasks;
    extern std::array<Bitboard, Square::COUNT> rookMasks;

    extern std::array<int, Square::COUNT> bishopShifts;
    extern std::array<int, Square::COUNT> rookShifts;

    extern std::array<std::array<Bitboard,  512>, Square::COUNT> bishopTable;
    extern std::array<std::array<Bitboard, 4096>, Square::COUNT> rookTable;

    const extern std::array<uint64_t, Square::COUNT> bishopMagics;
    const extern std::array<uint64_t, Square::COUNT> rookMagics;

    void init();
}

inline Bitboard getPawnMoves(Square square, Bitboard occupied, Color color) {
    Rank doubleMoveRank = color == Color::WHITE ? Rank::RANK_2 : Rank::RANK_7;

    auto up = color == Color::WHITE ? [](Bitboard bb) { return bb << 8;  } 
                                    : [](Bitboard bb) { return bb >> 8;  };
                                    
    Bitboard moves = up(Bitboard(square)) & ~occupied;

    if (square.getRank() == doubleMoveRank)
        moves |= up(moves) & ~occupied;

    return moves;
}

inline Bitboard getPawnAttacks(Square square, Color color) {
    auto up = color == Color::WHITE ? [](Bitboard bb) { return bb << 8;  } 
                                    : [](Bitboard bb) { return bb >> 8;  };

    return up(Bitboard(square)) >> 1 | up(Bitboard(square)) << 1;
}

inline Bitboard getKnightAttacks(Square square) {
    return Attacks::knightAttacks[int(square)];
}

inline Bitboard getBishopAttacks(Square square, Bitboard occupied) {
    Bitboard mask  = Attacks::bishopMasks[square];
    uint64_t magic = Attacks::bishopMagics[square];
    int      shift = Attacks::bishopShifts[square];

    int index = ((occupied & mask) * magic) >> shift;

    return Attacks::bishopTable[square][index];
}

inline Bitboard getRookAttacks(Square square, Bitboard occupied) {
    Bitboard mask  = Attacks::rookMasks[square];
    uint64_t magic = Attacks::rookMagics[square];
    int      shift = Attacks::rookShifts[square];

    int index = ((occupied & mask) * magic) >> shift;

    return Attacks::rookTable[square][index];
}

inline Bitboard getQueenAttacks(Square square, Bitboard occupied) {
    return getBishopAttacks(square, occupied) | getRookAttacks(square, occupied);
}

inline Bitboard getKingAttacks(Square square) {
    return Attacks::kingAttacks[int(square)];
}

template<PieceType TYPE>
constexpr Bitboard getAttacks(Square square, Bitboard occupied = Bitboard(0), Color color = Color::WHITE) {
    switch (TYPE)
    {
        case PieceType::PAWN  : return getPawnAttacks(square, color);
        case PieceType::KNIGHT: return getKnightAttacks(square);
        case PieceType::BISHOP: return getBishopAttacks(square, occupied);
        case PieceType::ROOK  : return getRookAttacks(square, occupied);
        case PieceType::QUEEN : return getQueenAttacks(square, occupied);
        default /* KING */    : return getKingAttacks(square);
    }
}
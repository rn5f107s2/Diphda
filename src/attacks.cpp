#include <array>

#include "attacks.h"
#include "types.h"

namespace Attacks {

std::array<Bitboard, Square::NONE> initJumperAttacks(Bitboard (*slowAttacks)(Square square));

Bitboard knightAttacksSlow(Square square);
Bitboard kingAttacksSlow(Square square);

const std::array<Bitboard, Square::NONE> knightAttacks = initJumperAttacks(&knightAttacksSlow);
const std::array<Bitboard, Square::NONE> kingAttacks   = initJumperAttacks(&kingAttacksSlow);

std::array<Bitboard, Square::NONE> initJumperAttacks(Bitboard (*slowAttacks)(Square square)) {
    std::array<Bitboard, Square::NONE> out = {};

    for (Square square = Square::H1; square < Square::NONE; ++square) {
        out[int(square)] = slowAttacks(square);
    }

    return out;
}

Bitboard knightAttacksSlow(Square square) {
    Bitboard bb      = Bitboard(square);
    Bitboard attacks = Bitboard(0);

    if (square.getFile() > File::H_FILE)
        attacks |= bb << 15 | bb >> 17;

    if (square.getFile() > File::G_FILE)
        attacks |= bb << 6 | bb >>  10;

     if (square.getFile() < File::A_FILE)
        attacks |= bb << 17 | bb >> 15;

    if (square.getFile() < File::B_FILE)
        attacks |= bb << 10 | bb >>  6;

    return attacks;
}

Bitboard kingAttacksSlow(Square square) {
    Bitboard bb      = Bitboard(square);
    Bitboard attacks = bb << 8 | bb >> 8;

    if (square.getFile() > File::H_FILE)
        attacks |= bb << 7 | bb >> 9 | bb >> 1;

    if (square.getFile() < File::A_FILE)
        attacks |= bb << 9 | bb >> 7 | bb << 1;


    return attacks;
}

Bitboard sliderAttacksSlow(Square square, Bitboard blocker, bool bishop) {
    Bitboard squareL      = Bitboard(square);
    Bitboard attacks      = Bitboard(0);
    Bitboard releventEdge = Bitboard(0);
    Bitboard edges[4]     = { Bitboard(Rank::RANK_1), Bitboard(File::A_FILE), Bitboard(File::H_FILE), Bitboard(Rank::RANK_8) };

    const int  firstDirection = bishop ? 9 : 1;
    const int secondDirection = bishop ? 7 : 8;

    int shifts[4] = {firstDirection, secondDirection, -firstDirection, -secondDirection};

    auto shift = [](Bitboard bb, int amount) { return amount >= 0 ? bb << amount : bb >> amount; };

    for (int i = 0; i < 4; i++)
        if (!(edges[i] & squareL))
            releventEdge |= edges[i];

    for (int i = 0; i < 4; i++) {
        Bitboard directionAttacks = shift(squareL, shifts[i]);

        if (!(kingAttacksSlow(square) & directionAttacks))
            continue;

        while (!(directionAttacks & (releventEdge | blocker)))
            directionAttacks |= shift(directionAttacks, shifts[i]);

        attacks |= directionAttacks;
    }

    return attacks;
}

}

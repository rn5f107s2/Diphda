#include "movegen.h"
#include "movelist.h"
#include "position.h"
#include "attacks.h"

namespace Movegen {

bool isPinned(const Position &pos, Square from, Square to) {
    if (!(pos.getPinned() & Bitboard(from)))
        return false;

    return !(lineBB(pos.getKingSquare(pos.getSideToMove()), from) & Bitboard(to));
}

Bitboard getPostEPOccupied(const Position &pos, Square from, Square to) {
    return pos.getOccupied() ^ Bitboard(from) ^ Bitboard(to) ^ Bitboard(Square(to.getFile(), from.getRank()));
}

void generatePromotions(MoveList &ml, Square from, Square to) {
    for (PromotionPiece promo = PromotionPiece::QUEEN; promo >= PromotionPiece::KNIGHT; --promo) {
        ml.pushBack(Move(from, to, MoveType::PROMO, promo));
    }
}

void generatePawnMoves(const Position &pos, MoveList &ml, Square from, Bitboard target) {
    Color us = pos.getSideToMove();

    Bitboard occupied = pos.getOccupied();
    Bitboard opponent = pos.getPieces(~us) | Bitboard(pos.getEPSquare());
    Bitboard moves = (getPawnAttacks(from, us) & opponent) | getPawnMoves(from, occupied, us);

    moves &= target;

    while (moves) {
        Square   to   = popLSB(moves);
        MoveType type = to == pos.getEPSquare() ? MoveType::EN_PASSANT : MoveType::NORMAL;

        if (isPinned(pos, from, to))
            continue;

        if (   type == MoveType::EN_PASSANT
            && pos.isAttacked(pos.getKingSquare(us), pos.getOccupied() ^ Bitboard(from) ^ Bitboard(to) ^ getPostEPOccupied(pos, from, to)))
            continue;

        to.getRank().isBackrank() ? generatePromotions(ml, from, to) : ml.pushBack(Move(from, to, type));
    }
}

template<PieceType TYPE>
void generateMoves(const Position &pos, MoveList &ml, Square from, Bitboard target) {
    if constexpr (TYPE == PieceType::PAWN)
        return generatePawnMoves(pos, ml, from, target);

    Bitboard own     = pos.getPieces(pos.getSideToMove());
    Bitboard attacks = getAttacks<TYPE>(from, pos.getOccupied(), pos.getSideToMove()) & ~own;

    attacks &= target;

    while (attacks) {
        Square to = popLSB(attacks);

        if (TYPE != PieceType::KING && isPinned(pos, from, to))
            continue;

        if (TYPE == PieceType::KING && pos.isAttacked(to, pos.getOccupied() ^ Bitboard(from)))
            continue;

        ml.pushBack(Move(from, to));
    }
}

template<PieceType TYPE>
void generateMoves(const Position &pos, MoveList &ml, Bitboard target) {
    Bitboard pieceBB = pos.getPieces<TYPE>(pos.getSideToMove());

    while (pieceBB) {
        Square from = popLSB(pieceBB);

        generateMoves<TYPE>(pos, ml, from, target);
    }
}

void generateMoves(const Position &pos, MoveList &ml) {
    Bitboard checkers = pos.getCheckers();

    Square   checkerSquare = checkers ? lsb(checkers) : Square(Square::NONE);
    Bitboard checkMask     = checkers ? Bitboard(checkerSquare) | betweenBB(pos.getKingSquare(pos.getSideToMove()), checkerSquare) 
                                      : Bitboard(-1);

    if (!(checkers && multipleBits(checkers))) {
        generateMoves<PieceType::PAWN  >(pos, ml, checkMask);
        generateMoves<PieceType::KNIGHT>(pos, ml, checkMask);
        generateMoves<PieceType::BISHOP>(pos, ml, checkMask);
        generateMoves<PieceType::ROOK  >(pos, ml, checkMask);
        generateMoves<PieceType::QUEEN >(pos, ml, checkMask);
    }

    generateMoves<PieceType::KING>(pos, ml, Bitboard(-1));
}

}

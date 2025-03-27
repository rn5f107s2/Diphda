#include "movegen.h"
#include "movelist.h"
#include "position.h"
#include "attacks.h"

namespace Movegen {

void generatePromotions(MoveList &ml, Square from, Square to) {
    for (PromotionPiece promo = PromotionPiece::QUEEN; promo >= PromotionPiece::KNIGHT; --promo) {
        ml.pushBack(Move(from, to, MoveType::PROMO, promo));
    }
}

void generatePawnMoves(Position &pos, MoveList &ml, Square from) {
    Color us = pos.getSideToMove();

    Bitboard occupied = pos.getOccupied();
    Bitboard opponent = pos.getPieces(~us) | Bitboard(pos.getEPSquare());
    Bitboard moves = (getPawnAttacks(from, us) & opponent) | getPawnMoves(from, occupied, us);

    while (moves) {
        Square to = popLSB(moves);

        MoveType type = to == pos.getEPSquare() ? MoveType::EN_PASSANT : MoveType::NORMAL; 

        to.getRank().isBackrank() ? generatePromotions(ml, from, to) : ml.pushBack(Move(from, to, type));
    }
}

template<PieceType TYPE>
void generateMoves(Position &pos, MoveList &ml, Square from) {
    if constexpr (TYPE == PieceType::PAWN)
        return generatePawnMoves(pos, ml, from);

    Bitboard own = pos.getPieces(pos.getSideToMove());
    Bitboard attacks = getAttacks<TYPE>(from, pos.getOccupied(), pos.getSideToMove()) & ~own;

    while (attacks) {
        Square to = popLSB(attacks);

        ml.pushBack(Move(from, to));
    }
}

template<PieceType TYPE>
void generateMoves(Position &pos, MoveList &ml) {
    Bitboard pieceBB = pos.getPieces<TYPE>(pos.getSideToMove());

    while (pieceBB) {
        Square from = popLSB(pieceBB);

        generateMoves<TYPE>(pos, ml, from);
    }
}

void generateMoves(Position &pos, MoveList &ml) {
    generateMoves<PieceType::PAWN  >(pos, ml);
    generateMoves<PieceType::KNIGHT>(pos, ml);
    generateMoves<PieceType::BISHOP>(pos, ml);
    generateMoves<PieceType::ROOK  >(pos, ml);
    generateMoves<PieceType::QUEEN >(pos, ml);
    generateMoves<PieceType::KING  >(pos, ml);
}

}

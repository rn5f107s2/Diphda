#include "movegen.h"
#include "position.h"
#include "attacks.h"
#include "../../movelist.h"

namespace Chess {

namespace Movegen {

bool isPinned(const Position &pos, Square from, Square to) {
    if (!(pos.getPinned() & Bitboard(from)))
        return false;

    return !(lineBB(pos.getKingSquare(pos.getSideToMove()), from) & Bitboard(to));
}

void generatePromotions(MoveList &ml, Square from, Square to) {
    for (PromotionPiece promo = PromotionPiece::QUEEN; promo >= PromotionPiece::KNIGHT; --promo) {
        ml.pushBack(Move(from, to, MoveType::PROMO, promo));
    }
}

void generatePawnMoves(const Position &pos, MoveList &ml, Square from, Bitboard target) {
    Color us = pos.getSideToMove();

    Bitboard occupied = pos.getOccupied();
    Bitboard opponent = pos.getPieces(~us) | (pos.getEPSquare() != Square::NONE ? Bitboard(pos.getEPSquare()) : Bitboard(0));
    Bitboard moves = (getPawnAttacks(from, us) & opponent) | getPawnMoves(from, occupied, us);

    moves &= target;

    while (moves) {
        Square   to   = popLSB(moves);
        MoveType type = to == pos.getEPSquare() ? MoveType::EN_PASSANT : MoveType::NORMAL;

        if (isPinned(pos, from, to))
            continue;

        if (type == MoveType::EN_PASSANT) {
            Square capturedSquare = Square(to.getFile(), from.getRank());
            
            if (pos.isAttacked(pos.getKingSquare(us), pos.getOccupied() ^ Bitboard(from) ^ Bitboard(to) ^ Bitboard(capturedSquare), capturedSquare))
                continue;
        }

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

void generateCastling(const Position &pos, MoveList &ml) {
    Color us = pos.getSideToMove();
    Rank backRank = us == Color::WHITE ? Rank::RANK_1 : Rank::RANK_8;

    if (pos.canCastle(CastlingRights::valueOf(us, false))) {
        Square sq1 = Square(File::F_FILE, backRank);
        Square sq2 = Square(File::G_FILE, backRank);
        
        if (pos.getOccupied() & (Bitboard(sq1) | Bitboard(sq2)))
            goto queenside;

        for (Square sq : {sq1, sq2}) {
            if (pos.isAttacked(sq, pos.getOccupied()))
                goto queenside;
        }

        ml.pushBack(Move(pos.getKingSquare(us), Square(File::G_FILE, backRank), MoveType::CASTLING));
    }

queenside:

    if (pos.canCastle(CastlingRights::valueOf(pos.getSideToMove(), true))) {
        Square sq1 = Square(File::D_FILE, backRank);
        Square sq2 = Square(File::C_FILE, backRank);
        
        if (pos.getOccupied() & (Bitboard(sq1) | Bitboard(sq2) | Bitboard(Square(File::B_FILE, backRank))))
            return;

        for (Square sq : {sq1, sq2}) {
            if (pos.isAttacked(sq, pos.getOccupied()))
                return;
        }

        ml.pushBack(Move(pos.getKingSquare(us), Square(File::C_FILE, backRank), MoveType::CASTLING));
    }
}

void generateMoves(const Position &pos, MoveList &ml) {
    Bitboard checkers = pos.getCheckers();

    Square   checkerSquare = checkers ? lsb(checkers) : Square(Square::NONE);
    Bitboard checkMask     = checkers ? Bitboard(checkerSquare) | betweenBB(pos.getKingSquare(pos.getSideToMove()), checkerSquare) 
                                      : Bitboard(-1);

    Bitboard pawnCheckMask = pos.getEPSquare() == Square::NONE ? checkMask : checkMask | Bitboard(pos.getEPSquare());

    if (!(checkers && multipleBits(checkers))) {
        generateMoves<PieceType::PAWN  >(pos, ml, pawnCheckMask);
        generateMoves<PieceType::KNIGHT>(pos, ml, checkMask);
        generateMoves<PieceType::BISHOP>(pos, ml, checkMask);
        generateMoves<PieceType::ROOK  >(pos, ml, checkMask);
        generateMoves<PieceType::QUEEN >(pos, ml, checkMask);
    }

    generateMoves<PieceType::KING>(pos, ml, Bitboard(-1));

    if (!checkers)
        generateCastling(pos, ml);
}

} // Namespace Movegen

} // Namespace Chess

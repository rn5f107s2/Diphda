#pragma once

#include <array>
#include <string>
#include <cstring>
#include <cmath>

#include "types.h"
#include "move.h"
#include "movegen.h"
#include "attacks.h"
#include "../../movelist.h"

namespace Chess {

class Position {
public:
    void setStartpos();
    void setPosition(std::string fen);

    template<PieceType TYPE>
    Bitboard getPieces(Color color) const;
    Bitboard getPieces(PieceType pieceType) const;
    Bitboard getPieces(Color color) const;
    Bitboard getOccupied() const;
    Bitboard getCheckers() const;
    Bitboard getPinned() const;
    Square getKingSquare(Color color) const;

    Color  getSideToMove() const;
    Square getEPSquare() const;

    bool isAttacked(Square square, Bitboard occupied) const;
    bool isAttacked(Square square, Bitboard occupied, Square capturedPawn) const;

    bool canCastle(CastlingRights::Value side) const;

    void generateMoves(MoveList &ml);

    void makeMove(Move move);

    int  indexOf(Move move);
    void toChess768Dense(int* indices);

    bool isWon();
    bool isDrawn();
    bool isLost();

    double simpleQ();

    std::string toString();

    Position() {
        clear();
    }

private:
    std::array<Bitboard, 2                    > colors;
    std::array<Bitboard, int(PieceType::COUNT)> pieces;

    Color          sideToMove;
    Square         enPassantSquare;
    CastlingRights castlingRights;

    Bitboard pinnedPieces;
    Bitboard checkers;

    int legalMoves;

    void parsePieces(std::string piecesFen);
    void parseSideToMove(std::string stmFen);
    void parseCastling(std::string castlingFen);

    void  addPiece(Piece piece, Square square);
    void  removePiece(Piece piece, Square square);
    Piece getPieceOn(Square square) const;

    void initPinnedPieces();
    void initCheckers();

    Bitboard attackersTo(Square square);

    void clear();
};

inline Piece Position::getPieceOn(Square square) const {
    Color     color = Color::BLACK; 
    PieceType type  = PieceType::NO_TYPE;

    for (PieceType pt = PieceType::PAWN; pt <= PieceType::KING; ++pt) {
        if (!(getPieces(pt) & Bitboard(square)))
            continue;

        type = pt;
        break;
    }

    // Black is default, so no extra check needed
    if (getPieces(Color::WHITE) & Bitboard(square))
        color = Color::WHITE;

    return Piece(color, type);
}

template<PieceType TYPE>
inline Bitboard Position::getPieces(Color color) const {
    return getPieces(color) & getPieces(TYPE);
}

inline Bitboard Position::getPieces(PieceType type) const {
    return pieces[int(type)];
}

inline Bitboard Position::getPieces(Color color) const {
    return colors[int(color)];
}

inline Bitboard Position::getOccupied() const {
    return getPieces(Color::WHITE) | getPieces(Color::BLACK);
}

inline Bitboard Position::getPinned() const {
    return pinnedPieces;
}

inline Bitboard Position::getCheckers() const {
    return checkers;
}

inline Square Position::getKingSquare(Color color) const {
    return lsb(getPieces<PieceType::KING>(color));
}

inline Color Position::getSideToMove() const {
    return sideToMove;
}

inline Square Position::getEPSquare() const {
    return enPassantSquare;
}

inline bool Position::canCastle(CastlingRights::Value side) const {
    return castlingRights.canCastle(side);
}


inline Bitboard Position::attackersTo(Square square) {
    Color them = ~sideToMove;

    return  getPawnAttacks  (square, sideToMove   ) &  getPieces<PieceType::PAWN  >(them)
          | getKnightAttacks(square               ) &  getPieces<PieceType::KNIGHT>(them)
          | getBishopAttacks(square, getOccupied()) & (getPieces<PieceType::BISHOP>(them) | getPieces<PieceType::QUEEN>(them))
          | getRookAttacks  (square, getOccupied()) & (getPieces<PieceType::ROOK  >(them) | getPieces<PieceType::QUEEN>(them))
          | getKingAttacks  (square               ) & (getPieces<PieceType::KING  >(them));
}

inline bool Position::isAttacked(Square square, Bitboard occupied) const {
    Color them = ~sideToMove;

    return    getPawnAttacks  (square, sideToMove) &  getPieces<PieceType::PAWN>  (them)
           || getKnightAttacks(square            ) &  getPieces<PieceType::KNIGHT>(them)
           || getKingAttacks  (square            ) & (getPieces<PieceType::KING  >(them))
           || getBishopAttacks(square, occupied  ) & (getPieces<PieceType::BISHOP>(them) | getPieces<PieceType::QUEEN>(them))
           || getRookAttacks  (square, occupied  ) & (getPieces<PieceType::ROOK  >(them) | getPieces<PieceType::QUEEN>(them));
}

inline bool Position::isAttacked(Square square, Bitboard occupied, Square capturedPawn) const {
    Color them = ~sideToMove;

    return    getPawnAttacks  (square, sideToMove) & (getPieces<PieceType::PAWN>  (them) ^ Bitboard(capturedPawn))
           || getKnightAttacks(square            ) &  getPieces<PieceType::KNIGHT>(them)
           || getKingAttacks  (square            ) & (getPieces<PieceType::KING  >(them))
           || getBishopAttacks(square, occupied  ) & (getPieces<PieceType::BISHOP>(them) | getPieces<PieceType::QUEEN>(them))
           || getRookAttacks  (square, occupied  ) & (getPieces<PieceType::ROOK  >(them) | getPieces<PieceType::QUEEN>(them));
}

inline void Position::generateMoves(MoveList &ml) {
    initPinnedPieces();
    initCheckers();

    Chess::generateMoves(*this, ml);

    legalMoves = ml.length();
}

inline bool Position::isWon() {
    return false;
}

inline bool Position::isDrawn() {
    return !legalMoves && !checkers;
}

inline bool Position::isLost() {
    return !legalMoves && checkers;
}

inline double Position::simpleQ() {
    double score = 0;
    double values[5] = { 0.1, 0.3, 0.3, 0.5, 0.9 };

    for (PieceType pt = PieceType::PAWN; pt < PieceType::KING; ++pt) {
        score += __builtin_popcountll(pieces[int(pt)] & colors[int( sideToMove)]) * values[int(pt)];
        score -= __builtin_popcountll(pieces[int(pt)] & colors[int(~sideToMove)]) * values[int(pt)];
    }

    return std::tanh(score);
}

inline void Position::clear() {
    colors.fill(0);
    pieces.fill(0);

    legalMoves = -1;
}

inline int Position::indexOf(Move move) {
    return move.getFrom() * 64 + move.getTo();
}

inline void Position::toChess768Dense(int* indices) {
    int idx = 0;

    for (Color c : {sideToMove, ~sideToMove}) {
        for (PieceType pt = PieceType::PAWN; pt != PieceType::NO_TYPE; ++pt) {
            Bitboard pieces = getPieces(c) & getPieces(pt);

            while (pieces) {
                int square = popLSB(pieces);
                Piece relativePiece = Piece(c == sideToMove ? Color::WHITE : Color::BLACK, pt);

                if (sideToMove == Color::BLACK)
                    square ^= 56;

                indices[idx++] = 64 * relativePiece + square;
            }
        }
    }

    memset(indices + idx, -1, (32 - idx) * sizeof(int));
}

} // Namespace Chess
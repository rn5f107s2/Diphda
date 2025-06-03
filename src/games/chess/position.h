#pragma once

#include <array>
#include <string>
#include <cstring>
#include <cmath>
#include <vector>

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

    bool latestMatches(Position& other);
    bool matchesHistory(const Position& other) const;
    int  historyDepth();

    bool isWon();
    bool isDrawn();
    bool isLost();

    double simpleQ();

    std::string toString();

    inline bool operator==(const Position& other) const {
        if (repetitionHistory.size() != other.repetitionHistory.size())
            return false;

        if (!matchesHistory(other))
            return false;

        for (int i = 0; i < 2; i++)
            if (colors[i] != other.colors[i])
                return false;

        for (int i = 0; i < int(PieceType::COUNT); i++)
            if (pieces[i] != other.pieces[i])
                return false;

        if (fiftyMoveRule != other.fiftyMoveRule)
            return false;

        if (sideToMove != other.sideToMove)
            return false;

        if (enPassantSquare != other.enPassantSquare)
            return false;

        if (castlingRights != other.castlingRights)
            return false;

        return true;
    }

    Position& operator=(const Position&) = default;

    inline bool operator!=(const Position& other) const {
        return !(*this == other);
    }

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

    std::vector<uint64_t> repetitionHistory;

    int fiftyMoveRule;

    uint64_t key;

    int legalMoves;

    void parsePieces(std::string piecesFen);
    void parseSideToMove(std::string stmFen);
    void parseCastling(std::string castlingFen);

    void  addPiece(Piece piece, Square square);
    void  removePiece(Piece piece, Square square);
    Piece getPieceOn(Square square) const;

    void initPinnedPieces();
    void initCheckers();

    bool hasRepeated();

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

inline bool Position::latestMatches(Position& other) {
    int depth = std::min(repetitionHistory.size(), other.repetitionHistory.size());

    return repetitionHistory[depth - 1] == other.repetitionHistory[depth - 1];
}

inline bool Position::matchesHistory(const Position& other) const {
    int depth = std::min(repetitionHistory.size(), other.repetitionHistory.size());

    for (int i = depth - 1; i >= 0; i--)
        if (repetitionHistory[i] != other.repetitionHistory[i])
            return false;

    return true;
}

inline int Position::historyDepth() {
    return repetitionHistory.size();
}

inline bool Position::isWon() {
    return false;
}

inline bool Position::isDrawn() {
    if (!legalMoves && !checkers)
        return true;

    if (fiftyMoveRule >= 100 && !isLost())
        return true;

    return hasRepeated();
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
    castlingRights.reset();

    repetitionHistory.clear();
    repetitionHistory.reserve(6000);

    enPassantSquare = Square::NONE;

    key = fiftyMoveRule = 0;

    legalMoves = -1;
}

inline int Position::indexOf(Move move) {
    Square from = sideToMove == Color::WHITE ? move.getFrom() : move.getFrom().mirrorVertical();
    Square to   = sideToMove == Color::WHITE ? move.getTo  () : move.getTo  ().mirrorVertical();

    return from * 64 + to;
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
#pragma once

#include <array>
#include <string>

#include "types.h"

class Position {
public:
    void setPosition(std::string fen);

    template<PieceType TYPE>
    Bitboard getPieces(Color color) const;
    Bitboard getPieces(PieceType pieceType) const;
    Bitboard getPieces(Color color) const;
    Bitboard getOccupied() const;

    Color  getSideToMove() const;
    Square getEPSquare() const;

    std::string toString();

    Position() {
        clear();
    }

private:
    std::array<Bitboard, 2                    > colors;
    std::array<Bitboard, int(PieceType::COUNT)> pieces;

    Color  sideToMove;
    Square enPassantSquare;

    void parsePieces(std::string piecesFen);

    void  addPiece(Piece piece, Square square);
    void  removePiece(Piece piece, Square square);
    Piece getPieceOn(Square square);

    void clear();
};

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

inline Color Position::getSideToMove() const {
    return sideToMove;
}

inline Square Position::getEPSquare() const {
    return enPassantSquare;
}

inline void Position::clear() {
    colors.fill(0);
    pieces.fill(0);
}
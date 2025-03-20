#pragma once

#include <array>
#include <string>

#include "types.h"

class Position {
public:
    void setPosition(std::string fen);

    Bitboard getPieces(PieceType pieceType) const;
    Bitboard getPieces(Color color) const;

    std::string toString();

    Position() {
        clear();
    }

private:
    std::array<Bitboard, 2                    > colors;
    std::array<Bitboard, int(PieceType::COUNT)> pieces;

    void parsePieces(std::string piecesFen);

    void  addPiece(Piece piece, Square square);
    void  removePiece(Piece piece, Square square);
    Piece getPieceOn(Square square);

    void clear();
};

inline Bitboard Position::getPieces(PieceType type) const {
    return pieces[int(type)];
}

inline Bitboard Position::getPieces(Color color) const {
    return colors[int(color)];
}

inline void Position::clear() {
    colors.fill(0);
    pieces.fill(0);
}
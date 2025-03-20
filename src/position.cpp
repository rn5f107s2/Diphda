#include <string>
#include <cctype>
#include <iostream>
#include <vector>

#include "position.h"
#include "uci.h"

void Position::setPosition(std::string fen) {
    std::vector<std::string> splitFEN = split(fen, ' ');

    clear();

    parsePieces(splitFEN.at(0));
}

void Position::parsePieces(std::string piecesFen) {
    Square currentSquare = Square::A8;

    for (char c : piecesFen) {
        Piece piece = Piece(c);

        if (std::isdigit(c)) {
            currentSquare -= c - '0';
            continue;
        }

        if (c == '/') {
            continue;
        }

        addPiece(piece, currentSquare);

        --currentSquare;
    }
}

void Position::addPiece(Piece piece, Square square) {
    Bitboard bb = Bitboard(square);

    pieces[int(piece.getType()) ] ^= bb;
    colors[int(piece.getColor())] ^= bb;
}

void Position::removePiece(Piece piece, Square square) {
    Bitboard bb = Bitboard(square);

    pieces[int(piece.getType()) ] ^= bb;
    colors[int(piece.getColor())] ^= bb;
}

Piece Position::getPieceOn(Square square) {
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

std::string Position::toString() {
    std::string position = "  +";

    for (int i = 0; i < 24; i++)
        position += "-";
        
    position += "+\n";

    for (Square sq = Square::H1; sq <= Square::A8; ++sq) {
        // The internal board representation is in the format h1 = 0, 
        // humans however expect a8 = 0 (top left), so mirror the square
        Square square = sq.mirrorHorizontal().mirrorVertical();

        if (square.getFile() == File::A_FILE)
            position += square.getRank().toString() + " |";

        position += " " + getPieceOn(square).toString() + " ";

        if (square.getFile() == File::H_FILE)
            position += "|\n";
    }

    position += "  +";

    for (int i = 0; i < 24; i++)
        position += "-";

    position += "+\n   ";

    for (File file = File::A_FILE; file >= File::H_FILE; --file)
        position += " " + file.toString() + " ";

    return position + "\n";
}
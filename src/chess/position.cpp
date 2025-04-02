#include <string>
#include <cctype>
#include <iostream>
#include <vector>

#include "position.h"
#include "attacks.h"
#include "../utility.h"

namespace Chess {

void Position::setPosition(std::string fen) {
    std::vector<std::string> splitFEN = split(fen, ' ');

    clear();

    parsePieces(splitFEN.at(0));
    parseSideToMove(splitFEN.at(1));
    parseCastling(splitFEN.at(2));
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

void Position::parseSideToMove(std::string stmFen) {
    sideToMove = stmFen.at(0) == 'b' ? Color::BLACK : Color::WHITE;
}

void Position::parseCastling(std::string castlingFen) {
    castlingRights.reset();

    for (char c : castlingFen) {
        switch (c) {
            case 'K': castlingRights.set(CastlingRights::WHITE_KINGSIDE); break;
            case 'Q': castlingRights.set(CastlingRights::WHITE_QUEENSIDE); break;
            case 'k': castlingRights.set(CastlingRights::BLACK_KINGSIDE); break;
            case 'q': castlingRights.set(CastlingRights::BLACK_QUEENSIDE); break;
        }
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

void Position::makeMove(Move move) {
    Square   from = move.getFrom();
    Square   to   = move.getTo();
    MoveType type = move.getType();

    Piece movedPiece = getPieceOn(from);
    Piece captured   = getPieceOn(to);

    castlingRights.updateCastlingRights(from);
    castlingRights.updateCastlingRights(to);

    removePiece(movedPiece, from);

    if (captured != Piece::NONE)
        removePiece(captured, to);

    if (type == MoveType::EN_PASSANT)
        removePiece(Piece(~sideToMove, PieceType::PAWN), Square(to.getFile(), from.getRank()));

    enPassantSquare = Square::NONE;

    if (   movedPiece.getType() == PieceType::PAWN
        && abs(from - to) == 16)
        enPassantSquare = Square((from + to) >> 1);

    if (type == MoveType::PROMO)
        movedPiece = Piece(sideToMove, move.getPromo().getType());

    if (type == MoveType::CASTLING) {
        Rank backRank = sideToMove == Color::WHITE ? Rank::RANK_1 : Rank::RANK_8;

        File rFrom = from > to ? File::H_FILE : File::A_FILE;
        File rTo   = from > to ? File::F_FILE : File::D_FILE;

        Piece rook = Piece(sideToMove, PieceType::ROOK);

        removePiece(rook, Square(rFrom, backRank));
        addPiece(rook, Square(rTo, backRank));
    }

    addPiece(movedPiece, to);

    sideToMove = ~sideToMove;
}

void Position::initPinnedPieces() {
    pinnedPieces = Bitboard(0);

    Color us   = sideToMove;
    Color them = ~us;

    Square kingSquare = getKingSquare(sideToMove);

    Bitboard opponent = getPieces(~sideToMove);
    Bitboard possiblePinners =   (getBishopAttacks(kingSquare, opponent) & (getPieces<PieceType::QUEEN>(them) | getPieces<PieceType::BISHOP>(them)))
                               | (getRookAttacks  (kingSquare, opponent) & (getPieces<PieceType::QUEEN>(them) | getPieces<PieceType::ROOK  >(them)));

    while (possiblePinners) {
        int pinnerSquare    = popLSB(possiblePinners);
        Bitboard pinnedLine = betweenBB(pinnerSquare, kingSquare) & getPieces(us);

        if (!multipleBits(pinnedLine))
            pinnedPieces |= pinnedLine;
    }
}

void Position::initCheckers() {
    checkers = attackersTo(getKingSquare(sideToMove));
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

} // Namespace Chess

#pragma once

#include <cstdint>
#include <string>

#include "types.h"

namespace Chess {

class MoveType {
public:
    enum Value {
        NORMAL, CASTLING, EN_PASSANT, PROMO
    };

    constexpr MoveType(Value v) : value(v) {}

    constexpr MoveType(int v) : value(Value(v)) {}

    constexpr operator Value() const { return value; }
private:
    Value value;
};

class PromotionPiece {
public:
    enum Value {
        KNIGHT, BISHOP, ROOK, QUEEN, NONE = 0
    };

    constexpr PromotionPiece(Value v) : value(v) {}

    constexpr PromotionPiece(int v) : value(Value(v)) {}

    constexpr operator Value() const { return value; }

    PieceType getType() {
        return PieceType(value + 1);
    }

    std::string toString() {
        switch (value) {
            case KNIGHT: return "n";
            case BISHOP: return "b";
            case ROOK  : return "r";
            case QUEEN : return "q";
            default    : return "";
        }
    }
private:
    Value value;
};

constexpr PromotionPiece& operator--(PromotionPiece &piece) { return piece = PromotionPiece(int(piece) - 1); }

class Move {
public:
    constexpr static int MAX_LEGAL = 256;

    Move() = default;

    Move(Square from, Square to, MoveType type = MoveType::NORMAL, PromotionPiece promo = PromotionPiece::NONE) {
        data =   (from  & SIX_BITS) << 0
               | (to    & SIX_BITS) << 6
               | (type  & TWO_BITS) << 12
               | (promo & TWO_BITS) << 14;
    }

    Square getFrom() {
        return Square(data & SIX_BITS);
    }

    Square getTo() {
        return Square((data >> 6) & SIX_BITS);
    }

    MoveType getType() {
        return MoveType((data >> 12) & TWO_BITS);
    }

    PromotionPiece getPromo() {
        return PromotionPiece((data >> 14) & TWO_BITS);
    }

    const Move& operator=(Move other) {
        data = other.data;

        return *this;
    }

    std::string toString() {
        std::string from  = getFrom().getFile().toString() + getFrom().getRank().toString();
        std::string to    = getTo  ().getFile().toString() + getTo  ().getRank().toString();
        std::string promo = getType() == MoveType::PROMO ? getPromo().toString() : "";

        return from + to + promo;
    }

private:
    static const int SIX_BITS = 0b111111;
    static const int TWO_BITS = 0b11;

    int16_t data;
};

} // Namespace Chess
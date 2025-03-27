#pragma once

#include <cstdint>
#include <string> 

using Bitboard = uint64_t;

enum class PieceType : int8_t {
    PAWN,
    KNIGHT,
    BISHOP,
    ROOK,
    QUEEN,
    KING,

    NO_TYPE,

    COUNT,
};

constexpr PieceType& operator++(PieceType& type) { return type = PieceType(int(type) + 1); };

enum class Color : bool {
    WHITE,
    BLACK
};

constexpr Color operator~(Color color) { return Color(!bool(color)); };

class File {
public:
    enum Value : int8_t {
        H_FILE,
        G_FILE,
        F_FILE,
        E_FILE,
        D_FILE,
        C_FILE,
        B_FILE,
        A_FILE
    };

    constexpr File(int file) : value(Value(file)) {}

    constexpr operator Value() const { return value; }

    constexpr explicit operator Bitboard() const { return 0x101010101010101 << int(value); }

    constexpr File& operator--() { 
        value = Value(int(value - 1));

        return *this;
    }; 

    std::string toString() {
        return std::string{ char('h' - value) };
    }

private:
    Value value;
};

class Rank {
    public:
        enum Value : int8_t {
            RANK_1,
            RANK_2,
            RANK_3,
            RANK_4,
            RANK_5,
            RANK_6,
            RANK_7,
            RANK_8
        };
    
        constexpr Rank(int rank) : value(Value(rank)) {}
    
        constexpr operator Value() const { return value; }

        constexpr explicit operator Bitboard() const { return 0xffULL << (int(value) * 8); }

        bool isBackrank() {
            return value == RANK_1 || value == RANK_8;
        }
    
        std::string toString() {
            return std::string{ char('1' + value) };
        }
    
    private:
        Value value;
};

class Square {
public:
    enum Value : int8_t {
        H1, G1, F1, E1, D1, C1, B1, A1,
        H2, G2, F2, E2, D2, C2, B2, A2,
        H3, G3, F3, E3, D3, C3, B3, A3,
        H4, G4, F4, E4, D4, C4, B4, A4,
        H5, G5, F5, E5, D5, C5, B5, A5,
        H6, G6, F6, E6, D6, C6, B6, A6,
        H7, G7, F7, E7, D7, C7, B7, A7,
        H8, G8, F8, E8, D8, C8, B8, A8,

        NONE,
        COUNT = 64
    };

    constexpr Square() : value(NONE) {}

    constexpr Square(Value v) : value(v) {}

    constexpr Square(int v) : value(Value(v)) {}

    constexpr operator Value() const { return value; }

    constexpr explicit operator Bitboard() const { return 1ULL << int(value); }

    constexpr Square operator-(int amount) const { return Square(Value(int(value) - amount)); };

    constexpr Square& operator--() { 
        value = Value(int(value - 1));

        return *this;
    }; 

    constexpr Square& operator++() { 
        value = Value(int(value + 1));

        return *this;
    };

    constexpr Square& operator-=(int amount) { 
        *this = *this - amount;

        return *this;
    }

    constexpr Square mirrorHorizontal() {
        return Square(Value(int(value) ^ 7));
    }

    constexpr Square mirrorVertical() {
        return Square(Value(int(value) ^ 56));
    }

    constexpr File getFile() {
        return File(value & 7);
    }

    constexpr Rank getRank() {
        return Rank(value / 8);
    }

private:
    Value value;
};

class Piece {
public:
    enum Value : int8_t {
        WHITE_PAWN,
        WHITE_KNIGHT,
        WHITE_BISHOP,
        WHITE_ROOK,
        WHITE_QUEEN,
        WHITE_KING,

        BLACK_PAWN,
        BLACK_KNIGHT,
        BLACK_BISHOP,
        BLACK_ROOK,
        BLACK_QUEEN,
        BLACK_KING,

        NO_PIECE
    };

    constexpr Piece(char c) : value(valueOf(c)) {}
    constexpr Piece(Value v) : value(v) {}
    constexpr Piece(Color c, PieceType pt) : value(Value(int(pt) + (c == Color::BLACK) * Piece::BLACK_PAWN)) {}

    constexpr Color getColor() const {
        return value <= Value::WHITE_KING ? Color::WHITE : Color::BLACK;
    }

    constexpr PieceType getType() const {
        return PieceType(value % 6);
    }

    std::string toString() const {
        return std::string{ value == NO_PIECE ? '-' : toChar() }; 
    }

private:
    Value value;

    constexpr char toChar() const {
        for (char c = 'A'; c <= 'z'; c++)
            if (valueOf(c) == value)
                return c;

        return '-';
    }

    constexpr Value valueOf(char c) const {
        switch (c) {
            case 'P': return Value::WHITE_PAWN;
            case 'N': return Value::WHITE_KNIGHT;
            case 'B': return Value::WHITE_BISHOP;
            case 'R': return Value::WHITE_ROOK;
            case 'Q': return Value::WHITE_QUEEN;
            case 'K': return Value::WHITE_KING;

            case 'p': return Value::BLACK_PAWN;
            case 'n': return Value::BLACK_KNIGHT;
            case 'b': return Value::BLACK_BISHOP;
            case 'r': return Value::BLACK_ROOK;
            case 'q': return Value::BLACK_QUEEN;
            case 'k': return Value::BLACK_KING;

            default : return Value::NO_PIECE;
        }
    }
};

inline Square popLSB(Bitboard &bb) {
    Square lsb = Square(__builtin_ctzll(bb));

    bb &= bb - 1;

    return lsb;
}

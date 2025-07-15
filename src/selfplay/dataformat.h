#pragma once

#include <array>
#include <vector>
#include <cstdint>

#include "../search/search.h"

enum Result : int8_t {
    WHITE_WIN, BOTH_DRAW, BLACK_WIN, UNDECIDED
};

#pragma pack(push, 1)

class MoveInfo {
    uint8_t nMoves;
    uint8_t playedIdx;

    float rootQ;

    // Only [nMoves] are written
    std::array<float   , 218> qValues;
    std::array<uint32_t, 218> visits;

public:
    MoveInfo(Node* root, uint8_t playedIdx);

    friend std::ostream& operator<<(std::ostream& stream, const MoveInfo& mi);
};

class GameRecord {
    Result result;
    int movecount;
    
    // Only [movecount] are written
    std::vector<MoveInfo> moves;

public:
    void pushBack(MoveInfo mi);
    void setResult(bool ww, bool d, bool wl);

    void clear();

    friend std::ostream& operator<<(std::ostream& stream, const GameRecord& mi);
};

#pragma pack(pop)
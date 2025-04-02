#pragma once

#include <cstdint>

#include "chess/position.h"
#include "chess/move.h"
#include "movelist.h"

template<bool ROOT>
uint64_t perft(Chess::Position &pos, int depth) {
    if (depth <= 0)
        return 1;

    MoveList ml; pos.generateMoves(ml);

    if (!ROOT && depth == 1)
        return ml.length();

    uint64_t nodeCount = 0;

    for (Chess::Move move : ml) {
        Chess::Position pos2 = pos;

        pos2.makeMove(move);

        uint64_t thisNodes = perft<false>(pos2, depth - 1);

        if constexpr (ROOT)
            std::cout << move.toString() << ": " << thisNodes << "\n";

        nodeCount += thisNodes;
    }

    if constexpr (ROOT)
        std::cout << "_____________\nNodes: " << nodeCount << std::endl;

    return nodeCount;
}
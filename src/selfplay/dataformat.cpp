#include "dataformat.h"

MoveInfo::MoveInfo(Node* n, int playIdx) {
    
}

void GameRecord::pushBack(MoveInfo mi) {
    moves.push_back(std::move(mi));
}

void GameRecord::setResult(bool ww, bool d, bool wl) {
    if (ww)
        result = WHITE_WIN;
    else if (d)
        result = BOTH_DRAW;
    else if (wl)
        result = BLACK_WIN;
    else
        result = UNDECIDED;
}

void GameRecord::clear() {
    result = UNDECIDED;
    movecount = 0;

    moves.clear();
}

std::ostream& operator<<(std::ostream& stream, const MoveInfo& mi) {
    stream << mi.nMoves << mi.playedIdx << mi.rootQ;
    
    for (int i = 0; i < mi.nMoves; i++)
        stream << mi.qValues[i] << std::endl;

    for (int i = 0; i < mi.nMoves; i++)
        stream << mi.visits[i] << std::endl;

    return stream;
}

std::ostream& operator<<(std::ostream& stream, const GameRecord& gr) {
    stream << gr.result << gr.movecount;
    
    for (int i = 0; i < gr.movecount; i++)
        stream << gr.moves[i];

    return stream;
}
#include "dataformat.h"

MoveInfo::MoveInfo(Node* n, uint8_t playIdx) {
    nMoves = n->childCount;

    playedIdx = playIdx;

    rootQ = n->getQ();

    for (int i = 0; i < n->childCount; i++) {
        qValues[i] = n->children[i].getQ();
        visits [i] = n->children[i].visits; 
    }
}

void GameRecord::pushBack(MoveInfo mi) {
    movecount++;
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
    stream.write(reinterpret_cast<const char*>(&mi.nMoves), sizeof(uint8_t));
    stream.write(reinterpret_cast<const char*>(&mi.playedIdx), sizeof(uint8_t));
    stream.write(reinterpret_cast<const char*>(&mi.rootQ), sizeof(float));

    stream.write(reinterpret_cast<const char*>(&mi.qValues[0]), sizeof(float) * mi.nMoves);

    stream.write(reinterpret_cast<const char*>(&mi.visits[0]), sizeof(uint32_t) * mi.nMoves);

    return stream;
}

std::ostream& operator<<(std::ostream& stream, const GameRecord& gr) {
    stream.write(reinterpret_cast<const char*>(&gr.result), sizeof(int8_t));
    stream.write(reinterpret_cast<const char*>(&gr.movecount), sizeof(int));
    
    for (int i = 0; i < gr.movecount; i++)
        stream << gr.moves[i];

    return stream;
}
#pragma once

#include "selfplayparams.h"
#include "selfplay.h"
#include "dataformat.h"

class SelfplayManager {
private:
    Evaluator* eval;

    SelfplayParmeters params;

    int miniBatchSize;
    int concurrentGames;

    bool activeHalf = false;

    uint64_t nodesSearched = 0, gamesPlayed = 0, positions = 0;
    std::chrono::steady_clock::time_point begin;

    std::array<std::vector<SelfplaySearcher>, 2> games;
    std::array<std::vector<GameRecord>, 2> gameRecords;

    void collectBatch();
    void collectNode(int gameIdx);

    std::ofstream outFile;

public:
    SelfplayManager();

    void run();
};
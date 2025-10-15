#pragma once

#include <chrono>

#include "network.h"
#include "../games/game.h"

void bench() {
    constexpr int batchSize = 128;

    Position pos; pos.setStartpos();
    Network* net = new Network(batchSize);
    net->loadWeights("Leela8x1.bin");

    int inputs[32 * batchSize];
    int outputIndices[218 * batchSize];

    memset(inputs, -1, sizeof(inputs));
    memset(outputIndices, -1, sizeof(outputIndices));

    MoveList ml; pos.generateMoves(ml);

    for (int i = 0; i < batchSize; i++) {
        pos.toChess768Dense(inputs + (32 * i));

        for (int j = 0; j < ml.length(); j++)
            outputIndices[i * 218 + j] = pos.indexOf(ml[j]);
    }

    auto begin = std::chrono::steady_clock::now();

    uint64_t nodes = 0;

    while (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count() < 1000) {
        net->forward(inputs, outputIndices);
        nodes += batchSize;
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

    std::cout << (nodes * 1000 / duration) << " nps " << std::endl;
}
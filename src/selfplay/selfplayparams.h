#pragma once

struct SelfplayParmeters {
    int maxConcurrentGames = 4096;
    int maxMiniBatchSize   = 2048;
    int samplesPerGame     = 16;

    int playouts = 800;

    int   tempDropMoveCount = 40;
    float initialTemp = 0.9f;
    float postDropTemp = 0.0f;

    float dirichletAlpha = 0.3;
    float dirichletEpsilon = 0.25;
};
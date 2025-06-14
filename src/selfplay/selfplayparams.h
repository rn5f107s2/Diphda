#pragma once

struct SelfplayParmeters {
    int maxConcurrentGames = 2;
    int maxMiniBatchSize   = 300;

    int playouts = 5000;

    int   tempDropMoveCount = 45;
    float initialTemp = 1.0f;
    float postDropTemp = 0.1f;

    float dirichletAlpha = 0.3;
    float dirichletEpsilon = 0.25;
};
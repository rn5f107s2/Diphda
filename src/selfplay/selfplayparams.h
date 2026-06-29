#pragma once

struct SelfplayParmeters {
    int maxConcurrentGames = 1000;
    int maxMiniBatchSize   = 500;

    int playouts = 800;

    int   tempDropMoveCount = 45;
    float initialTemp = 0.7f;
    float postDropTemp = 0.2f;

    float dirichletAlpha = 0.3;
    float dirichletEpsilon = 0.25;
};
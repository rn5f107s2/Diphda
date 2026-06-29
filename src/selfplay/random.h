#pragma once

#include <random>

static std::mt19937 gen;

inline double gammaDist(double alpha, double beta) {
    std::gamma_distribution<double> dist(alpha, beta);
    return dist(gen);
}

inline double uniformDistDouble(double low, double high) {
    std::uniform_real_distribution<double> dist(low, high);
    return dist(gen);
}
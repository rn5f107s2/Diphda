#pragma once

#include <random>
#include <atomic>

static std::mt19937 gen;
static std::mutex mtx;

inline double gammaDist(double alpha, double beta) {
    std::unique_lock lk(mtx);
    std::gamma_distribution<double> dist(alpha, beta);
    double val = dist(gen);
    lk.unlock();
    return val;
}

inline double uniformDistDouble(double low, double high) {
    std::unique_lock lk(mtx);
    std::uniform_real_distribution<double> dist(low, high);
    double val = dist(gen);
    lk.unlock();
    return val;
}
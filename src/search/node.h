#pragma once

#include <cstdint>
#include <atomic>

#include "parameters.h"
#include "../games/game.h"
#include "../movelist.h"

class Collector;

enum GameState : int8_t {
    ONGOING, LOSS, DRAW, WIN
};

class PackedInfo {
private:
    std::atomic<int8_t> raw = 31; // 1 bit waiting 2 bit w/d/l 5 bit ply, 31 = not waiting, not terminal, ply = 31 = maxply

public:
    bool waiting() {
        return raw.load(std::memory_order_consume) >> 7;
    }

    void waiting(bool newState) {
        int8_t r = raw.load(std::memory_order_relaxed);

        r &= 0b01111111;
        r |= 0b10000000 * newState;

        raw.store(r, std::memory_order_release);
    }

    GameState state() {
        return GameState((raw.load(std::memory_order_relaxed) & 0b01100000) >> 5);
    }

    void state(GameState newState) {
        int8_t r = raw.load(std::memory_order_relaxed);
        r &= 0b10011111;
        r |= newState << 5;
        raw.store(r, std::memory_order_relaxed);
    }

    int ply() {
        return raw.load(std::memory_order_relaxed) & 0b00011111;
    }

    void ply(int8_t newPly) {
        if (newPly > 31)
            newPly = 31;

        int8_t r = raw.load(std::memory_order_relaxed);

        r &= 0b11100000;
        r |= newPly;

        raw.store(r, std::memory_order_relaxed);
    }
};

struct Edge;

class Node {
public:
    std::atomic<double> q = 0; // 8

    Node* parent   = nullptr; // 8
    Node* children = nullptr; // 8
    Edge* edges    = nullptr; // 8

    std::atomic<int> visits = 0; // 4

    PackedInfo info; // 1

    uint8_t index = 0; // 1
    uint8_t childCount = 0; // 1

public:
    Node(Node* p, uint8_t idx): parent(p), index(idx) {}

    void search(Position& pos, Collector& eval, const SearchParameters& params, float c);

    Node* select(float c);
    void  expand(Position& pos, Collector& eval);
    void  backpropagate(double score);
    void  backpropagateMate(Node* child);
    void  virtualLoss(bool undo);

    double uct(uint64_t parentVisits, float c, double parentQ);
    double getQ();

    void labelPolicies(float* raw, float temperature);

    void deallocate();

    void updateVisits(int amount);
    void updateQ(double change);
    void createEdges(MoveList& ml);
    void createChildren();

    float getPolicy();
    Move  getMove();
};

struct Edge {
    std::atomic<float> policy;
    const Move  move;
    
public:
    Edge(Move m) : move(m) {}
};
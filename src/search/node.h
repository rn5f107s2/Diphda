#pragma once

#include <cstdint>
#include <atomic>

#include "../games/game.h"
#include "../movelist.h"

class Evaluator;

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

class Node {
public:
    std::atomic<int   > visits = 0;
    std::atomic<double> q      = 0;

    PackedInfo info;

    Move    move;
    Node*   parent     = nullptr;
    Node*   children   = nullptr;
    uint8_t childCount = 0;

    std::atomic<float> policy = 0;

public:
    Node(Move m, Node* p) : move(m), parent(p) {}

    void search(Position& pos, Evaluator& eval, float c);

    Node* select(float c);
    void  expand(Position& pos, Evaluator& eval);
    void  backpropagate(double score);
    void  backpropagateMate(Node* child);
    void  virtualLoss(bool undo);

    double uct(uint64_t parentVisits, float c, double parentQ);
    double getQ();

    void labelPolicies(float* raw, float temperature);

    void deallocate();

    void updateVisits(int amount);
    void updateQ(double change);
    void createChildren(MoveList& ml);
};
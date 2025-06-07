#pragma once

#include <cstdint>

#include "../games/game.h"

class Evaluator;

enum GameState : int8_t {
    ONGOING, LOSS, DRAW, WIN
};

class PackedInfo {
private:
    int8_t raw = 31; // 1 bit waiting 2 bit w/d/l 5 bit ply, 31 = not waiting, not terminal, ply = 31 = maxply

public:
    bool waiting() {
        return raw >> 7;
    }

    void waiting(bool newState) {
        raw &= 0b01111111;
        raw |= 0b10000000 * newState;
    }

    GameState state() {
        return GameState((raw & 0b01100000) >> 5);
    }

    void state(GameState newState) {
        raw &= 0b10011111;
        raw |= newState << 5;
    }

    int ply() {
        return raw & 0b00011111;
    }

    void ply(int8_t newPly) {
        if (newPly > 31)
            newPly = 31;

        raw &= 0b11100000;
        raw |= newPly;
    }
};

class Node {
public:
    uint64_t visits = 0;
    double   q      = 0;

    PackedInfo info;

    Move    move;
    float   policy     = 0;
    Node*   parent     = nullptr;
    Node*   children   = nullptr;
    uint8_t childCount = 0;

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
};
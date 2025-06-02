#pragma once

#include <cstdint>

#include "../games/game.h"
#include "../network/network.h"

void search(Position& pos);

class Node {
public:
    uint64_t visits = 0;
    double   q      = 0;

    bool terminal = false;

    Move    move;
    float   policy = 0;
    Node*   parent;
    Node*   children   = nullptr;
    uint8_t childCount = 0;

public:
    Node() = delete;

    Node(Move m, Node* p) : move(m), parent(p) {}

    void search(Position& pos);

    Node* select();
    void  expand(Position& pos);
    void  rollout(Position& pos);
    void  backpropagate(double score);

    double uct(uint64_t parentVisits);
    double getQ();

    void deallocate();
};
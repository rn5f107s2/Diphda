#pragma once

#include <cstdint>

#include "../games/game.h"
#include "../network/network.h"

struct NodePool;
struct Node;

static const double C = 1.414;

void search(Position& pos);

void writePolicyIndices(Position& pos, MoveList& ml, int* indices);

struct Evaluator {
    Network* net;

    int* inputs;
    int* policyIndices;
    Node* node;

    Evaluator() {
        net = new Network(1);

        net->loadWeights("testnet.bin");

        inputs        = (int*) malloc(sizeof(int) * 32);
        policyIndices = (int*) malloc(sizeof(int) * 218);
    }

    void add(Position &pos, MoveList& ml, Node* n) {
        pos.toChess768Dense(inputs);
        writePolicyIndices(pos, ml, policyIndices);

        net->forward(inputs, policyIndices);

        node = n;
    }

    Node* getNode() {
        return node;
    }

    float* getPolicies() {
        return net->getPolicy(0);
    }

    float* getValue() {
        return net->getValue(0);
    }
};

struct Node {
    Move move;
    float policy;

    uint64_t visits = 0;
    double   score  = 0.0;

    Node*   parent     = nullptr;
    Node*   children   = nullptr;
    uint8_t childCount = 0;

    void createChildren(MoveList& ml, NodePool& pool);
    void search(Position& pos, NodePool& pool);
    void labelPolicies(float* policies, double temperature);
    void backpropagate(double result);
    double puct(uint64_t parentVisits);

    Node* select();
};

struct NodePool {
    Node* memory;

    uint64_t sizeMB;
    uint64_t limit;
    uint64_t currIdx;

    NodePool(uint64_t size) : sizeMB(size) {
        uint64_t bytes = sizeMB * 1024 * 1024;
        uint64_t limit = bytes / sizeof(Node) * sizeof(Node);

        memory = (Node*) malloc(limit);

        currIdx = 0;
    }

    Node* allocate(int amount) {
        Node* ret = &memory[currIdx];
        currIdx += amount;
        return ret;
    }
};

inline void writePolicyIndices(Position& pos, MoveList& ml, int* indices) {
    int idx = 0;

    for (Move& m : ml)
        indices[idx++] = pos.indexOf(m);

    memset(indices + idx, -1, (218 - idx) * sizeof(int));
}
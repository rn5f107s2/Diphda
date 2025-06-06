#pragma once

#include <cstdint>

#include "../games/game.h"
#include "../network/network.h"

class Evaluator;

enum GameState : int8_t {
    ONGOING, LOSS, DRAW, WIN
};

class PackedInfo {
private:
    int8_t raw = 31; // 1 bit waiting 2 bit w/d/l 5 bit ply

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

    double uct(uint64_t parentVisits, float c);
    double getQ();

    void labelPolicies(float* raw, float temperature);

    void deallocate();
};

class Evaluator {
private:
    Network* net;

    const int batchSize;
    int* policyIndices;
    int* valueIndices;

    std::vector<Node*> nodes;

    void forward(float temperature = 1.0) {
        net->forward(valueIndices, policyIndices);

        for (int i = 0; i < nodes.size(); i++) {
            float* value = net->getValue(i);

            //float w = std::exp(value[2] / 2);
            //float d = std::exp(value[1] / 2);
            //float l = std::exp(value[0] / 2);

            //float sum = w + d + l;

            //float q = w / sum - l / sum;

            float q = std::tanh(value[0] / 2);

            nodes[i]->virtualLoss(true);
            nodes[i]->backpropagate(-q);
            nodes[i]->labelPolicies(net->getPolicy(i), temperature);

            nodes[i]->info.waiting(false);
        }

        nodes.clear();
    }

    void writeValueIndices(Position& pos) {
        pos.toChess768Dense(valueIndices + 32 * nodes.size());
    }

    void writePolicyIndices(Position& pos, MoveList& ml){
        int* indices = policyIndices + 218 * nodes.size();

        for (int i = 0; i < ml.length(); i++)
            indices[i] = pos.indexOf(ml[i]);

        memset(indices + ml.length(), -1, (218 - ml.length()) * sizeof(int));
    }

public:
    void addNode(Position& pos, MoveList& ml, Node* node) {
        writeValueIndices(pos);
        writePolicyIndices(pos, ml);

        nodes.push_back(node);

        if (nodes.size() == batchSize)
            forward();
    }

    void distribute(float temperature = 1.0) {
        forward(temperature);
    }

    Evaluator(int bs) : batchSize(bs) {
        policyIndices = (int*) malloc(218 * sizeof(int) * batchSize);
         valueIndices = (int*) malloc( 32 * sizeof(int) * batchSize);

        memset(policyIndices, -1, sizeof(218 * sizeof(int) * batchSize));

        net = new Network(batchSize);

        net->loadWeights("test256.bin");

        nodes.reserve(batchSize);
    }
};

class Searcher {
public:
    void search(Position& pos);
    void clear();

    Searcher() {
        evaluator = new Evaluator(100);
    }

private:
    bool     priorPosExists = false;
    Position priorPos;
    Node*    root = nullptr;

    Node* findNewRoot(Position& pos);
    Node* createNewRoot();

    Evaluator* evaluator;

    std::vector<Node*> disjunctSubtrees;
};
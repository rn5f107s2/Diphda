#include "search.h"

#include <cmath>
#include <chrono>

Evaluator* evaluator;

void search(Position& pos) {
    if (!evaluator)
        evaluator = new Evaluator();

    NodePool pool(256);

    Node root; root.parent = nullptr;

    for (int i = 0; i < 5000; i++) {
        Position p = pos;

        root.search(p, pool);

        Node* node = evaluator->getNode();
        node->labelPolicies(evaluator->getPolicies(), node == &root ? 5.0 : 1.0);

        float* value = evaluator->getValue();

        double w = std::exp(value[2]);
        double d = std::exp(value[1]);
        double l = std::exp(value[0]);

        double sum = w + d + l;
        double q   = w / sum - l / sum;

        node->backpropagate(-q);
    }

    double bestQ = -2;
    int    index = 0;

    for (int i = 0; i < root.childCount; i++) {
        double q = root.children[i].score / root.children[i].visits;
        std::cout << root.children[i].move.toString() << ": " << q << " " << root.children[i].visits << std::endl;

        if (q > bestQ) {
            bestQ = q;
            index = i;
        }
    }

    std::cout << "info depth 1 score cp " << std::round(bestQ * 100) << std::endl;
    std::cout << "bestmove " << root.children[index].move.toString() << std::endl;
}

void Node::createChildren(MoveList& ml, NodePool& pool) {
    children = pool.allocate(childCount = ml.length());

    for (int i = 0; i < childCount; i++) {
        children[i].move   = ml[i];
        children[i].parent = this;
    }
}

void Node::labelPolicies(float* policies, double temperature) {
    double sum = 0;

    for (int i = 0; i < childCount; i++)
        sum += std::exp(policies[i] / temperature);

    for (int i = 0; i < childCount; i++)
        children[i].policy = std::exp(policies[i] / temperature) / sum;
}

void Node::backpropagate(double result) {
    visits++;
    score += result;

    if (parent)
        parent->backpropagate(-result);
}

double Node::puct(uint64_t parentVisits) {
    double q                = !visits ? 1.0 : score / visits;
    double explorationScore = C * policy * std::sqrt(parentVisits) / (1 + visits);

    return q + explorationScore;
}

Node* Node::select() {
    double best = children[0].puct(visits);
    int bestIdx = 0;

    for (int i = 1; i < childCount; i++) {
        double value = children[i].puct(visits);

        if (value > best) {
            best    = value;
            bestIdx = i;
        }
    }

    return &children[bestIdx];
}

void Node::search(Position& pos, NodePool& pool) {
    if (!visits) {
        MoveList ml;
        pos.generateMoves(ml);
    
        createChildren(ml, pool);

        if (!childCount)
            return backpropagate(1.0);

        evaluator->add(pos, ml, this);

        return;
    }

    if (!childCount)
        return backpropagate(1.0);

    Node* selected = select();

    pos.makeMove(selected->move);

    selected->search(pos, pool);
}
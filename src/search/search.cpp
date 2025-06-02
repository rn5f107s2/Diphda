#include "search.h"

#include <cmath>
#include <chrono>

void search(Position& pos) {
    MoveList ml;
    pos.generateMoves(ml);

    Node root(Move(), nullptr);

    for (int i = 0; i < 5000; i++) {
        Position copy = pos;

        root.search(copy);
    }

    double bestQ = -1.0;
    Move bestMove;

    for (int i = 0; i < root.childCount; i++) {
        double q = root.children[i].getQ();

        if (q < bestQ)
            continue;

        bestQ = q;
        bestMove = root.children[i].move;
    }

    root.deallocate();

    std::cout << "info depth 1 score cp " << int(bestQ * 100) << std::endl;
    std::cout << "bestmove " << bestMove.toString() << std::endl;
}

void Node::search(Position& pos) {
    if (!visits) {
        expand(pos);
        return rollout(pos);
    }

    if (terminal)
        return backpropagate(std::abs(q) < 0.1 ? 0.0 : (q < 0 ? -1.0 : 1.0));

    Node* toSearch = select();

    pos.makeMove(toSearch->move);

    toSearch->search(pos);
}

double Node::uct(uint64_t parentVisits) {
    double Q = visits ? getQ() : 1.0;
    double U = 1.414 * std::sqrt(std::log(parentVisits) / std::max(visits, uint64_t(1)));

    return Q + U;
}

double Node::getQ() {
    return !visits ? -1.0 : q / visits;
}

Node* Node::select() {
    int    bestIndex = 0;
    double bestUCT   = children[0].uct(visits);

    for (int i = 1; i < childCount; i++) {
        double uct = children[i].uct(visits);

        if (uct < bestUCT)
            continue;

        bestUCT   = uct;
        bestIndex = i;
    }

    return children + bestIndex;
}

void Node::rollout(Position& pos) {
    bool won   = pos.isWon();
    bool lost  = pos.isLost();
    bool drawn = pos.isDrawn();

    terminal = won || lost || drawn;

    visits++;
    
    if (terminal)
        return backpropagate(won ? -1.0 : (drawn ? 0.0 : 1.0));

    return backpropagate((double(rand()) / double(RAND_MAX)) * 2. - 1.0);
}

void Node::expand(Position& pos) {
    MoveList ml; 
    pos.generateMoves(ml);

    childCount = ml.length();

    std::allocator<Node> allocator;
    children =  allocator.allocate(childCount);

    for (size_t i = 0; i < childCount; i++)
        new (children + i) Node(ml[i], this);
}

void Node::backpropagate(double score) {
    visits++;
    q += score;

    if (parent)
        parent->backpropagate(-score);
}

void Node::deallocate() {
    if (!children)
        return;

    for (int i = 0; i < childCount; i++)
        children[i].deallocate();

    std::allocator<Node> allocator;
    allocator.deallocate(children, childCount);
}
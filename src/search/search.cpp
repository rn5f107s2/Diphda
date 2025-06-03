#include "search.h"

#include <cmath>
#include <chrono>

void Searcher::search(Position& pos) {
    root = findNewRoot(pos);

    Node* rootParent = root->parent;
    root->parent = nullptr;

    for (int i = 0; i < 5000; i++) {
        Position copy = pos;

        root->search(copy);
    }

    root->parent = rootParent;

    double bestQ = -1.0;
    Move bestMove;

    for (int i = 0; i < root->childCount; i++) {
        double q = root->children[i].getQ();

        if (q < bestQ)
            continue;

        bestQ = q;
        bestMove = root->children[i].move;
    }

    std::cout << "info depth 1 score cp " << int(bestQ * 100) << std::endl;
    std::cout << "bestmove " << bestMove.toString() << std::endl;

    priorPos       = pos;
    priorPosExists = true;
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
    double U = 1.414 * policy * std::sqrt(parentVisits) / (1 + visits);

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

    backpropagate(-pos.simpleQ());
}

void Node::expand(Position& pos) {
    MoveList ml; 
    pos.generateMoves(ml);

    childCount = ml.length();

    std::allocator<Node> allocator;
    children =  allocator.allocate(childCount);

    for (size_t i = 0; i < childCount; i++) {
        new (children + i) Node(ml[i], this);

        children[i].policy = 1. / childCount;
    }
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

void Searcher::clear() {
    for (Node* disjunctRoot : disjunctSubtrees) {
        disjunctRoot->deallocate();
        delete disjunctRoot;
    }

    disjunctSubtrees.clear();

    priorPosExists = false;
}

Node* Searcher::createNewRoot() {
    Node* newRoot = new Node(Move(), nullptr);

    disjunctSubtrees.push_back(newRoot);

    return newRoot;
}

Node* Searcher::findNewRoot(Position& pos) {
    if (!priorPosExists || !priorPos.matchesHistory(pos))
        return createNewRoot();

    Node* rootCandidate = root;

    for (int i = 0; i < priorPos.historyDepth() - pos.historyDepth(); i++) {
        if (!rootCandidate->parent)
            return createNewRoot();

        rootCandidate = rootCandidate->parent;
    }

    Position origin = priorPos.historyDepth() <= pos.historyDepth() ? priorPos : pos;
    Position target = priorPos.historyDepth() <= pos.historyDepth() ? pos      : priorPos;

    while (origin != target) {
        bool found = false;

        if (origin.historyDepth() == target.historyDepth())
            return createNewRoot();

        if (!rootCandidate->children)
            return createNewRoot();
        
        for (int i = 0; i < rootCandidate->childCount; i++) {
            Position originCopy = origin;

            originCopy.makeMove(rootCandidate->children[i].move);

            found = originCopy.latestMatches(target);

            if (!found)
                continue;

            rootCandidate = &rootCandidate->children[i];
            break;
        }

        if (!found)
            return createNewRoot();

        origin.makeMove(rootCandidate->move);
    }

    for (int i = 0; i < priorPos.historyDepth() - pos.historyDepth(); i++) {
        if (!rootCandidate->parent)
            return createNewRoot();

        rootCandidate = rootCandidate->parent;
    }

    return rootCandidate;
}

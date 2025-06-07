#include "node.h"
#include "../games/game.h"
#include "evaluator.h"

void Node::search(Position& pos, Evaluator& eval, float c) {
    if (!visits)
        return expand(pos, eval);

    if (info.waiting())
        return eval.distribute();

    if (info.state() != ONGOING && !info.ply())
        return backpropagate(std::abs(q) < 0.1 ? 0.0 : (q < 0 ? -1.0 : 1.0));

    Node* toSearch = select(c);

    pos.makeMove(toSearch->move);

    toSearch->search(pos, eval, 1.414);
}

double Node::uct(uint64_t parentVisits, float c, double parentQ) {
    double Q = visits ? getQ() : -parentQ;
    double U = c * policy * std::sqrt(parentVisits) / (1 + visits);

    return Q + U;
}

double Node::getQ() {
    if (info.state() == ONGOING)
        return !visits ? -1.0 : q / visits;

    return info.state() == LOSS ? 1.0 : (info.state() == WIN ? -1.0 : 0.0);
}

Node* Node::select(float c) {
    int    bestIndex = 0;
    double bestUCT   = children[0].uct(visits, c, getQ());

    for (int i = 1; i < childCount; i++) {
        double uct = children[i].uct(visits, c, getQ());

        if (uct < bestUCT)
            continue;

        bestUCT   = uct;
        bestIndex = i;
    }

    return children + bestIndex;
}

void Node::expand(Position& pos, Evaluator& eval) {
    MoveList ml; 
    pos.generateMoves(ml);

    bool won   = pos.isWon();
    bool lost  = pos.isLost();
    bool drawn = pos.isDrawn();

    info.state(won ? WIN : (lost ? LOSS : (drawn ? DRAW : ONGOING)));
    
    if (info.state() != ONGOING) {
        info.ply(0);

        if (info.state() != DRAW)
            parent->backpropagateMate(this);

        return backpropagate(won ? -1.0 : (drawn ? 0.0 : 1.0));
    }

    std::allocator<Node> allocator;
    childCount = ml.length();
    children   = allocator.allocate(childCount);

    for (size_t i = 0; i < childCount; i++)
        new (children + i) Node(ml[i], this);

    info.waiting(true);

    eval.addNode(pos, ml, this);
    virtualLoss(false);
}

void Node::backpropagate(double score) {
    visits++;
    q += score;

    if (parent)
        parent->backpropagate(-score);
}

void Node::backpropagateMate(Node* child) {
    if (child->info.state() == LOSS) {
        info.state(WIN);
        info.ply(std::min(info.ply(), child->info.ply() + 1));

        if (parent)
            parent->backpropagateMate(this);
        
        return;
    }

    int maxPly = 0;

    for (int i = 0; i < childCount; i++) {
        if (children[i].info.state() != WIN)
            return;

        maxPly = std::max(maxPly, children[i].info.ply());
    }

    info.state(LOSS);
    info.ply(maxPly + 1);

    if (parent) 
        parent->backpropagateMate(this);
}

void Node::virtualLoss(bool undo) {
    visits += undo ? -1   :  1;
    q      += undo ?  1.0 : -1.0;

    if (parent)
        parent->virtualLoss(undo);
}

void Node::deallocate() {
    if (!children)
        return;

    for (int i = 0; i < childCount; i++)
        children[i].deallocate();

    std::allocator<Node> allocator;
    allocator.deallocate(children, childCount);
    children = nullptr;
}

void Node::labelPolicies(float* raw, float temperature) {
    float policies[256];
    float sum = 0.0;

    for (int i = 0; i < childCount; i++)
        sum += (policies[i] = std::exp(raw[i] / temperature));

    for (int i = 0; i < childCount; i++)
        children[i].policy = policies[i] / sum;
}
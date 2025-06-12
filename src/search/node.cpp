#include "node.h"
#include "../games/game.h"
#include "evaluator.h"
#include "searchparams.h"

void Node::search(Position& pos, Collector& collector, const SearchParameters& params, float c) {
    if (!visits.load(std::memory_order_relaxed))
        return expand(pos, collector);

    if (info.waiting())
        return collector.earlyFull();

    if (info.state() != ONGOING && !info.ply())
        return backpropagate(info.state() == WIN ? -1.0 : (info.state() == LOSS ? 1.0 : 0.0));

    if (!children)
        createChildren();

    Node* toSearch = select(c);

    pos.makeMove(toSearch->getMove());

    toSearch->search(pos, collector, params, params.cpuct);
}

double Node::uct(uint64_t parentVisits, float c, double parentQ) {
    int n = visits.load(std::memory_order_relaxed);
    
    double Q = n ? getQ() : -parentQ;
    double U = c * getPolicy() * std::sqrt(parentVisits) / (1 + n);

    return Q + U;
}

double Node::getQ() {
    int    n  = visits.load(std::memory_order_relaxed);
    double q2 = q.load(std::memory_order_relaxed); 

    if (info.state() == ONGOING)
        return !n ? -1.0 : q2 / n;

    return info.state() == LOSS ? 1.0 : (info.state() == WIN ? -1.0 : 0.0);
}

Node* Node::select(float c) {
    int    n  = visits.load(std::memory_order_relaxed);
    double q2 = getQ(); 

    int    bestIndex = 0;
    double bestUCT   = children[0].uct(n, c, q2);

    for (int i = 1; i < childCount; i++) {
        double uct = children[i].uct(n, c, q2);

        if (uct < bestUCT)
            continue;

        bestUCT   = uct;
        bestIndex = i;
    }

    return children + bestIndex;
}

void Node::expand(Position& pos, Collector& collector) {
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

    createEdges(ml);

    collector.addNode(this, pos, ml, 1.0f);
}

void Node::backpropagate(double score) {
    updateVisits(1);
    updateQ(score);

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
    updateVisits(undo ? -1 : 1);
    updateQ(undo ? 1.0 : -1.0);

    if (parent)
        parent->virtualLoss(undo);
}

void Node::deallocate() {
    if (!edges)
        return;

    std::allocator<Edge> eAllocator;
    eAllocator.deallocate(edges, childCount);

    edges = nullptr;

    if (!children)
        return;

    for (int i = 0; i < childCount; i++)
        children[i].deallocate();

    std::allocator<Node> nAllocator;
    nAllocator.deallocate(children, childCount);
    children = nullptr;
}

void Node::labelPolicies(float* raw, float temperature) {
    float policies[256];
    float sum = 0.0;

    for (int i = 0; i < childCount; i++)
        sum += (policies[i] = std::exp(raw[i] / temperature));

    for (int i = 0; i < childCount; i++)
        edges[i].policy.store(policies[i] / sum, std::memory_order_relaxed);
}

void Node::updateVisits(int amount) {
    visits.fetch_add(amount, std::memory_order_relaxed);
}

void Node::updateQ(double change) {
    q.fetch_add(change, std::memory_order_relaxed);
}

void Node::createEdges(MoveList& ml) {
    std::allocator<Edge> allocator;
    childCount = ml.length();
    edges      = allocator.allocate(childCount);

    for (size_t i = 0; i < childCount; i++)
        new (edges + i) Edge(ml[i]);
}

void Node::createChildren() {
    std::allocator<Node> allocator;
    children = allocator.allocate(childCount);

    for (int i = 0; i < childCount; i++)
        new (children + i) Node(this, i);
}

float Node::getPolicy() {
    return parent->edges[index].policy.load(std::memory_order_relaxed);
}

Move Node::getMove() {
    return parent->edges[index].move;
}

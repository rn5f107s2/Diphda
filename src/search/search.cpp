#include "search.h"

#include <cmath>
#include <chrono>

void Searcher::search(Position& pos) {
    root = findNewRoot(pos);

    Node* rootParent = root->parent;
    root->parent = nullptr;

    if (!root->visits) {
        root->expand(pos, *evaluator);
    } else {
        MoveList ml;
        pos.generateMoves(ml);
        evaluator->addNode(pos, ml, root);
    }

    evaluator->distribute(5.0f);

    for (int i = 0; i < 5000; i++) {
        Position copy = pos;

        root->search(copy, *evaluator);
    }

    evaluator->distribute();

    root->parent = rootParent;

    bool   win     = root->info.state() == WIN;
    double bestQ   = -1.0;
    int    bestPly = 32;
    Move bestMove;

    for (int i = 0; i < root->childCount; i++) {
        double q = root->children[i].getQ();

        std::cout << root->children[i].move.toString() << ": " << root->children[i].policy <<
                                                           " " << root->children[i].visits << 
                                                           " " << root->children[i].getQ() << std::endl;

        if (   (!win && q < bestQ) 
            || ( win && (root->children[i].info.state() != LOSS || root->children[i].info.ply() >= bestPly)))
            continue;

        bestQ    = q;
        bestPly  = root->children[i].info.ply();
        bestMove = root->children[i].move;
    }

    std::string value = !win ? std::to_string(int(std::round(std::atanh(bestQ) * 2 * 133))) : std::to_string(bestPly + 1);

    std::cout << "info depth 1 score " << (!win ? "cp " : "mate ") << value << std::endl;
    std::cout << "bestmove " << bestMove.toString() << std::endl;

    priorPos       = pos;
    priorPosExists = true;
}

void Node::search(Position& pos, Evaluator& eval) {
    if (!visits)
        return expand(pos, eval);

    if (info.waiting())
        return eval.distribute();

    if (info.state() != ONGOING && !info.ply())
        return backpropagate(std::abs(q) < 0.1 ? 0.0 : (q < 0 ? -1.0 : 1.0));

    Node* toSearch = select();

    pos.makeMove(toSearch->move);

    toSearch->search(pos, eval);
}

double Node::uct(uint64_t parentVisits) {
    double Q = visits ? getQ() : 1.0;
    double U = 1.414 * policy * std::sqrt(parentVisits) / (1 + visits);

    return Q + U;
}

double Node::getQ() {
    if (info.state() == ONGOING)
        return !visits ? -1.0 : q / visits;

    return info.state() == LOSS ? 1.0 : (info.state() == WIN ? -1.0 : 0.0);
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

        for (int i = 0; i < rootCandidate->parent->childCount; i++) 
            if (rootCandidate->parent->children[i].move != rootCandidate->move)
                rootCandidate->parent->children[i].deallocate();

        origin.makeMove(rootCandidate->move);
    }

    for (int i = 0; i < priorPos.historyDepth() - pos.historyDepth(); i++) {
        if (!rootCandidate->parent)
            return createNewRoot();

        rootCandidate = rootCandidate->parent;
    }

    return rootCandidate;
}

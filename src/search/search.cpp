#include "search.h"
#include "evaluator.h"

#include <cmath>
#include <chrono>

void Searcher::search(Position& pos, SearchTime& st) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    params.update();

    prepareNewRoot(pos);

    uint64_t nodes = 0;

    for (; !shouldStop(st, nodes); nodes++) {
        Position copy = pos;

        root->search(copy, evaluator->getCollector(), params, params.root_cpuct);

        if (evaluator->getCollector().getHalf(true).isFull())
            evaluator->forward();

        if (evaluator->getCollector().getHalf(true).forwardEarly)
            evaluator->forwardBlocking();
    }

    evaluator->forwardBlocking();

    bool win = root->info.state() == WIN;

    auto criterium = win ? [] (Node& n) { if (n.info.state() != LOSS) return -1.0; return double(31 - n.info.ply()); } 
                         : [] (Node& n) { return n.getQ(); };

    Node* best = selectBest(root, criterium);

    sendInfo(nodes, begin);             

    std::cout << "bestmove " << best->getMove().toString() << std::endl;

    priorPos       = pos;
    priorPosExists = true;
}


void Searcher::sendInfo(uint64_t nodes, std::chrono::steady_clock::time_point& begin) {
    bool win = root->info.state() == WIN;

    auto criterium = win ? [] (Node& n) { if (n.info.state() != LOSS) return -1.0; return double(31 - n.info.ply()); } 
                         : [] (Node& n) { return n.getQ(); };

    Node* best = selectBest(root, criterium);

    std::string value = win ? std::to_string(best->info.ply() + 1)
                            : std::to_string(int(std::round(std::atanh(best->getQ()) * 2 * 133)));

    std::chrono::steady_clock::time_point current = std::chrono::steady_clock::now();

    auto searchTime = std::chrono::duration_cast<std::chrono::milliseconds>(current - begin).count();

    std::cout << "info depth 1 score " << (!win ? "cp " : "mate ") << value 
              << " nodes " << nodes 
              << " time " << searchTime 
              << " nps " << (nodes * 1000 / (searchTime + 1)) 
              << " pv " << getPv(root, criterium) << std::endl;
}

std::string Searcher::getPv(Node* n, std::function<double(Node&)> func) {
    if (!n->children)
        return "";

    Node* best = selectBest(n, func);

    return best->getMove().toString() + " " + getPv(best, func);
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
    Node* newRoot = new Node(nullptr, 0);

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

            originCopy.makeMove(rootCandidate->children[i].getMove());

            found = originCopy.latestMatches(target);

            if (!found)
                continue;

            rootCandidate = &rootCandidate->children[i];
            break;
        }

        if (!found)
            return createNewRoot();

        for (int i = 0; i < rootCandidate->parent->childCount; i++) 
            if (rootCandidate->parent->children[i].getMove() != rootCandidate->getMove())
                rootCandidate->parent->children[i].deallocate();

        origin.makeMove(rootCandidate->getMove());
    }

    for (int i = 0; i < priorPos.historyDepth() - pos.historyDepth(); i++) {
        if (!rootCandidate->parent)
            return createNewRoot();

        rootCandidate = rootCandidate->parent;
    }

    return rootCandidate;
}

void Searcher::prepareNewRoot(Position& pos) {
    root = findNewRoot(pos);

    root->parent = nullptr;

    MoveList ml;
    pos.generateMoves(ml);

    if (!root->visits)
        root->createEdges(ml);
        
    evaluator->getCollector().addNode(root, pos, ml, params.root_pst);

    evaluator->forwardBlocking();
}

Node* Searcher::selectBest(Node* n, std::function<double(Node&)> func) {
    Node*  selected = nullptr;
    double best     = -std::numeric_limits<double>::infinity();

    for (int i = 0; i < n->childCount; i++) {
        double val = func(n->children[i]);

        if (val < best)
            continue;

        best     = val;
        selected = n->children + i;
    }

    return selected;
}

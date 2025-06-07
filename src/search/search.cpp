#include "search.h"
#include "evaluator.h"

#include <cmath>
#include <chrono>

void Searcher::search(Position& pos) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
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

    int nodes = 0;

    for (; nodes < 5000; nodes++) {
        Position copy = pos;

        root->search(copy, *evaluator, 5.0);
    }

    evaluator->distribute();

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    auto searchTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

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

    std::cout << "info depth 1 score " << (!win ? "cp " : "mate ") << value << " nps " << (nodes * 1000 / (searchTime + 1)) << std::endl;
    std::cout << "bestmove " << bestMove.toString() << std::endl;

    priorPos       = pos;
    priorPosExists = true;
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

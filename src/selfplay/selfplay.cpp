#include "selfplay.h"

void SelfplaySearcher::playMove(Move m) {
    nodes = 0;
    played = m;
    rootReady = false;

    pos.makeMove(m);

    Node* newRoot = nullptr;

    for (int i = 0; i < root->childCount; i++) {
        if (root->children[i].getMove().toString() == m.toString())
            newRoot = root->children + i;
        else
            root->children[i].deallocate();
    }

    root = newRoot;

    // generate moves for accurate terminal detection
    MoveList ml; pos.generateMoves(ml);
}

void SelfplaySearcher::prepareRoot() {
    if (!root)
        root = new Node(nullptr, 0);

    MoveList ml; pos.generateMoves(ml);

    if (!root->edges)
        root->createEdges(ml);

    evaluator->addNode(pos, ml, root, params.root_pst);

    rootReady = true;   
}

bool SelfplaySearcher::isTerminal() {
    return pos.isWon() || pos.isDrawn() || pos.isLost();
}

void SelfplaySearcher::startNewGame() {
    pos.setStartpos();

    cleanupRoot();
}

void SelfplaySearcher::cleanupRoot() {
    if (!root)
        return;

    while (root->parent) root = root->parent;

    root->deallocate();
    delete root;
}

void SelfplaySearcher::doPlayout() {
    Position copy = pos;

    root->search(copy, *evaluator, params, params.root_cpuct);
    nodes++;
}

void SelfplaySearcher::addSingle() {
    if (!rootReady)
        return prepareRoot();

    doPlayout();
}

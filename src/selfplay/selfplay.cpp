#include "selfplay.h"
#include "random.h"

#include <vector>
#include <algorithm>

void SelfplaySearcher::playMove(Move m) {
    nodes = 0;
    played = m;
    moveCount++;
    rootReady = dirichlet = false;

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

    evaluator->getCollector().addNode(root, pos, ml, params.root_pst);

    rootReady = true;   
}

bool SelfplaySearcher::isTerminal() {
    return pos.isWon() || pos.isDrawn() || pos.isLost();
}

void SelfplaySearcher::startNewGame() {
    pos.setStartpos();

    cleanupRoot();
    moveCount = nodes = rootReady = dirichlet = 0;
}

void SelfplaySearcher::cleanupRoot() {
    if (!root)
        return;

    while (root->parent) root = root->parent;

    root->deallocate();
    delete root;

    root = nullptr;
}

void SelfplaySearcher::doPlayout() {
    Position copy = pos;

    // Dont waste time propagating further up than needed
    Node* rp = root->parent;
    root->parent = nullptr;

    root->search(copy, evaluator->getCollector(), params, params.root_cpuct);
    nodes++;

    root->parent = rp;
}

void SelfplaySearcher::addSingle() {
    if (!rootReady)
        return prepareRoot();

    if (!dirichlet)
        applyDirichlet();

    doPlayout();
}

float SelfplaySearcher::getTemperature() {
    return moveCount > selfplayParams.tempDropMoveCount ? selfplayParams.postDropTemp : selfplayParams.initialTemp;
}

Node* SelfplaySearcher::chooseBestTerminal() {
    auto criterion = [] (Node& n) { 
        return n.info.state() == LOSS ? (32 - n.info.ply()) 
                                      : (n.info.state() == WIN ? -1 - n.info.ply()
                                                               : 0);  
    };

    Node* b = nullptr;
    int best = -100;

    for (int i = 0; i < root->childCount; i++) {
        int s = criterion(root->children[i]);

        if (s <= best)
            continue;

        best = s;
        b = root->children + i;
    }

    return b;
}

void SelfplaySearcher::applyDirichlet() {
    std::vector<float> noise; noise.reserve(root->childCount);
    float sum = 0.0f;

    for (int i = 0; i < root->childCount; i++) {
        noise.push_back(gammaDist(selfplayParams.dirichletAlpha, 1.0));
        sum += noise.back();
    }

    for (int i = 0; i < root->childCount; i++) {
        float newP = lerp(root->edges[i].policy.load(std::memory_order_relaxed), noise[i] / sum, selfplayParams.dirichletEpsilon);

        root->edges[i].policy.store(newP, std::memory_order_relaxed);
    }

    dirichlet = true;
}

Node* SelfplaySearcher::chooseAction() {
    if (root->info.state() != ONGOING)
        return chooseBestTerminal();

    float temperature = getTemperature();

    auto score = [&] (Node& n) { return std::pow(double(n.visits.load(std::memory_order_relaxed)), double(1. / temperature)); };

    double sum = 0;
    std::vector<double> scores; scores.reserve(root->childCount);

    for (int i = 0; i < root->childCount; i++) {
        scores.push_back(score(root->children[i]));
        sum += scores.back();
    }
     
    int i = 0;
    double rand = uniformDistDouble(0.0, 1.0);

    for (; i < root->childCount - 1; i++) {
        rand -= (scores[i] / sum);

        if (rand <= 0)
            break;
    }

    return root->children + i;
}

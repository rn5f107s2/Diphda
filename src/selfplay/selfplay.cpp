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

    cleanupRoot();

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

void SelfplaySearcher::addSingle(int samplesPerGame) {
    if (!rootReady)
        return prepareRoot();

    if (!dirichlet)
        applyDirichlet();

    for (int i = 0; i < samplesPerGame; i++)
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

    auto score = [&] (Node& n) { 
        uint64_t visits = n.visits.load(std::memory_order_relaxed);

        if (temperature <= 0)
            return double(visits);

        return visits ? std::log(double(visits)) / double(temperature) : -1.0;
    };

    double sum = 0;
    double max = -1;
    std::vector<double> logScores; logScores.reserve(root->childCount);
    std::vector<double> scores; scores.reserve(root->childCount);

    for (int i = 0; i < root->childCount; i++) {
        double logScore = score(root->children[i]);
    
        logScores.push_back(logScore);
        max = std::max(max, logScore);
    }

    for (int i = 0; i < root->childCount; i++) {
        bool dontSample = logScores[i] < 0 || (temperature <= 0 && logScores[i] != max);

        scores.push_back(dontSample ? 0.0 : std::exp(logScores[i] - max));
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

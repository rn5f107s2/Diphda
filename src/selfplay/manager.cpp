#include "manager.h"

SelfplayManager::SelfplayManager() {
    miniBatchSize   = std::min(params.maxConcurrentGames / 2, params.maxMiniBatchSize);
    concurrentGames = 2 * miniBatchSize;

    eval = new Evaluator(miniBatchSize);

    for (int i = 0; i < miniBatchSize; i++) {
        games[0].push_back(SelfplaySearcher(eval));
        games[1].push_back(SelfplaySearcher(eval));

        gameRecords[0].push_back(GameRecord());
        gameRecords[1].push_back(GameRecord());
    }
}

void SelfplayManager::collectBatch() {
    #pragma omp parallel for num_threads(4) schedule(static)
    for (int i = 0; i < miniBatchSize; i++)
        collectNode(i);

    eval->forward();

    activeHalf = !activeHalf;
}

void SelfplayManager::collectNode(int gameIdx) {
    SelfplaySearcher& game = games[activeHalf][gameIdx];
    GameRecord& record = gameRecords[activeHalf][gameIdx];

    if (game.shouldStop()) {
        Node* action = game.chooseAction();

        record.pushBack(MoveInfo(game.getRoot(), action->index));

        game.playMove(action->getMove());

        positions++;
    }

    if (game.isTerminal()) {
        record.setResult(game.ww(), game.d(), game.wl());

        std::unique_lock lk(mtx);

        outFile << record << std::flush;

        lk.unlock();

        record.clear();

        game.startNewGame();

        gamesPlayed++;
    }

    game.addSingle();

    nodesSearched++;
}

void SelfplayManager::run() {
    begin = std::chrono::steady_clock::now();

    outFile.open("testdata.bin", std::ios::binary);

    while (true) { 
        collectBatch();

        if (nodesSearched >= 10000000) {
            auto current = std::chrono::steady_clock::now();
            auto npms    = nodesSearched / std::chrono::duration_cast<std::chrono::milliseconds>(current - begin).count();

            std::cout << "\rPlayed " << gamesPlayed << " games containing " << positions << " positions at " << (npms * 1000) << " nps" << std::flush;

            if (gamesPlayed >= 5000)
                exit(0);

            begin = std::chrono::steady_clock::now();
            nodesSearched = 0;
        }
    }
}
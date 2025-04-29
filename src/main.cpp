#include <iostream>
#include <chrono>

#include "uci.h"
#include "games/chess/attacks.h"
#include "network/network.h"

int main(int argc, char** argv) {
    Chess::Attacks::init();

    Network* n = new Network();
    n->loadWeights("/dev/urandom");
    for (int i = 0; i < 10; i++) {
    std::cout << "Starting!" << std::endl;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    n->forward();
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Done!" << std::endl;
    std::cout << "Took: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    }

    UCIHandler* uci = new UCIHandler();
    uci->start(argc, argv);
    delete uci;
}
#include <iostream> 

#include "uci.h"
#include "utility.h"
#include "perft.h"
#include "games/game.h"
#include "search/search.h"

UCIHandler::UCIHandler() {
    internalBoard = new Position();

    internalBoard->setStartpos();

    commands["go"] = &UCIHandler::go;
    commands["uci"] = &UCIHandler::uci;
    commands["show"] = &UCIHandler::show;
    commands["perft"] = &UCIHandler::perft;
    commands["isready"] = &UCIHandler::isready;
    commands["position"] = &UCIHandler::position;
    commands["ucinewgame"] = &UCIHandler::ucinewgame;
} 

void UCIHandler::start(int argc, char** argv) {
    if (argc == 1)
        return loop();

    for (int i = 1; i < argc; i++) {
        std::string in = argv[i];
    
        if (in == "quit")
            break;
    
        handleInput(in);
    }
}

void UCIHandler::loop() {
    std::string input;

    while (true) {
        std::getline(std::cin, input);

        if (input == "quit")
            break;

        handleInput(input);
    }
}

void UCIHandler::handleInput(const std::string &in) {
    const std::string name = in.substr(0, in.find(' '));
    const std::string args = in.substr(std::min(in.length(), name.length() + 1));

    auto iter = commands.find(name);

    if (iter == commands.end()) {
        std::cout << "No such command: " << in << std::endl;
        return;
    }
    
    (this->*(iter->second))(args);
}

void UCIHandler::go(const std::string &arguments) {
    searcher.search(*internalBoard);
}

void UCIHandler::uci(const std::string &arguments) {
    std::cout << "uciok" << std::endl;
}

void UCIHandler::show(const std::string &arguments) {
    std::cout << internalBoard->toString() << std::endl;
}

void UCIHandler::perft(const std::string &arguments) {
    std::vector<std::string> splitArguments = split(arguments, ' ');

    int depth = std::stoi(splitArguments.at(0));

    ::perft<true>(*internalBoard, depth);
}

void UCIHandler::isready(const std::string &arguments) {
    std::cout << "readyok" << std::endl;
}

void UCIHandler::position(const std::string &arguments) {
    std::vector<std::string> splitArguments = split(arguments, ' ');

    bool fen = splitArguments.at(0) == "fen";
    int movesIndex = fen ? 6 : 2;

    if (fen)
        internalBoard->setPosition(splitArguments.at(1) + " " + splitArguments.at(2) + " " + splitArguments.at(3));
    else
        internalBoard->setStartpos();

    if (splitArguments.size() < movesIndex)
        return;

    for (int i = movesIndex; i < splitArguments.size(); i++) {
        std::string move = splitArguments.at(i);

        MoveList ml;
        internalBoard->generateMoves(ml);

        for (Move m : ml) {
            if (m.toString() == move) {
                internalBoard->makeMove(m);
                break;
            }
        }
    }
    
}

void UCIHandler::ucinewgame(const std::string& arguments) {
    searcher.clear();
}
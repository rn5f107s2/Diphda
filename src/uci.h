#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>

#include "ucioption.h"
#include "games/game.h"
#include "search/search.h"

static const std::string name       = "Diphda";
static const std::string version    = "0.1";

class UCIHandler {
public:
    void start(int argc, char** argv);

    UCIHandler();

private:
    void go(const std::string &arguments);
    void uci(const std::string &arguments);
    void show(const std::string &arguments);
    void perft(const std::string &argumetns);
    void isready(const std::string &arguments);
    void position(const std::string &arguments);
    void setoption(const std::string &arguments);
    void ucinewgame(const std::string &arguments);
    void backendbench(const std::string &arguments);

    void loop();
    void handleInput(const std::string &in);

    Searcher searcher;
    Position* internalBoard;
    OptionsContainer uciOptions;
    std::unordered_map<std::string, void(UCIHandler::*)(const std::string&)> commands;
};
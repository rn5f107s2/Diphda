#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>

#include "games/game.h"

static const std::string name       = "Diphda";
static const std::string version    = "0.1";

class UCIHandler {
public:
    void start(int argc, char** argv);

    UCIHandler();

private:
    void show(const std::string &arguments);
    void perft(const std::string &argumetns);
    void position(const std::string &arguments);

    void loop();
    void handleInput(const std::string &in);

    Position* internalBoard;
    std::unordered_map<std::string, void(UCIHandler::*)(const std::string&)> commands;
};
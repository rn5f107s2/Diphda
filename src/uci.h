#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>

class Position;

static const std::string name       = "Diphda";
static const std::string version    = "0.1";
static const std::string defaultFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

class UCIHandler {
public:
    void start(int argc, char** argv);

    UCIHandler();

private:
    void show(const std::string &arguments);
    void position(const std::string &arguments);

    void loop();
    void handleInput(const std::string &in);

    Position* internalBoard;
    std::unordered_map<std::string, void(UCIHandler::*)(const std::string&)> commands;
};

inline std::vector<std::string> split(std::string string, char delimiter) {
    std::stringstream stream(string);
    std::vector<std::string> split;
    std::string temp;

    while (std::getline(stream, temp, delimiter)) { split.push_back(temp); }

    return split;
}
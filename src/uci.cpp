#include <iostream> 

#include "uci.h"
#include "position.h"

UCIHandler::UCIHandler() {
    internalBoard = new Position();

    internalBoard->setPosition(defaultFEN);

    commands["show"] = &UCIHandler::show;
    commands["position"] = &UCIHandler::position;
} 

void UCIHandler::start(int argc, char** argv) {
    loop();
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

void UCIHandler::show(const std::string &arguments) {
    std::cout << internalBoard->toString() << std::endl;
}

void UCIHandler::position(const std::string &arguments) {
    std::vector<std::string> splitArguments = split(arguments, ' ');

    internalBoard->setPosition(splitArguments.at(1));
}
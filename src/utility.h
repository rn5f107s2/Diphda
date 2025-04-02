#pragma once

#include <vector>
#include <sstream>

inline std::vector<std::string> split(std::string string, char delimiter) {
    std::stringstream stream(string);
    std::vector<std::string> split;
    std::string temp;

    while (std::getline(stream, temp, delimiter)) { split.push_back(temp); }

    return split;
}
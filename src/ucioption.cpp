#include "ucioption.h"

#include <iostream>

UCIOption::UCIOption(std::string on, std::string tn) : optionName(on), typeName(tn) {}

std::string UCIOption::getName() {
    return optionName;
}

std::string UCIOption::toString() {
    return "option name " + getName() + " type " + typeName;
}

SpinOption::SpinOption(std::string name, int dflt, int min, int max) : UCIOption(name, "spin"),
                                                                       defaultValue(dflt),
                                                                       minValue(min),
                                                                       maxValue(max) 
{
    currentValue = defaultValue;
}

std::string SpinOption::toString() {
    return UCIOption::toString() + " default " + std::to_string(defaultValue) 
                                 + " min "     + std::to_string(minValue)
                                 + " max "     + std::to_string(maxValue);
}

bool SpinOption::set(const std::string& val) {
    int value = std::stoi(val);

    if (value > maxValue || value < minValue)
        return false;

    currentValue = value;

    return true;
}

FloatOption::FloatOption(std::string name, float dflt, float min, float max) : UCIOption(name, "string"),
                                                                               defaultValue(dflt),
                                                                               minValue(min),
                                                                               maxValue(max) 
{
    currentValue = defaultValue;
}

std::string FloatOption::toString() {
    return UCIOption::toString() + " default " + std::to_string(defaultValue);
}

bool FloatOption::set(const std::string& val) {
    float value = std::stof(val);

    if (value > maxValue || value < minValue)
        return false;

    currentValue = value;

    return true;
}

bool OptionsContainer::set(const std::string& name, const std::string &val) {
    for (UCIOption* opt : *this)
        if (opt->getName() == name)
            return opt->set(val);

    return false;
}

void OptionsContainer::print() {
    for (UCIOption* opt : *this)
        std::cout << opt->toString() << std::endl;
}
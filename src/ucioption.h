#pragma once

#include <string>
#include <vector>

struct UCIOption {
    std::string optionName;
    std::string typeName;

    UCIOption(std::string on, std::string tn);

    std::string getName();
    virtual std::string toString();
    virtual bool set(const std::string &val) = 0;
};

struct SpinOption : UCIOption {
    const int defaultValue, minValue, maxValue;

    int currentValue;

    SpinOption(std::string name, int dflt, int min, int max);

    std::string toString() override;

    bool set(const std::string &val) override;
};

struct FloatOption : UCIOption {
    const float defaultValue, minValue, maxValue;
    
    float currentValue;

    FloatOption(std::string name, float dflt, float min, float max);

    std::string toString() override;

    bool set(const std::string &val) override;
};

class OptionsContainer : std::vector<UCIOption*> {
public:
    using std::vector<UCIOption*>::push_back;
    using std::vector<UCIOption*>::insert;
    using std::vector<UCIOption*>::begin;
    using std::vector<UCIOption*>::end;

    bool set(const std::string& name, const std::string &val);

    void print();
};

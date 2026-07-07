#pragma once

#include "../ucioption.h"

#define OPTION_NAME(name) name##_opt

#define PARAMS                   \
    X(cpuct, 1.8, 0.1, 3)      \
    X(root_cpuct, 2.2, 1.0, 7.0) \
    X(root_pst, 1.4, 1.0, 7.0)

struct SearchParameters {
private:

#define X(name, dfault, min, max) FloatOption OPTION_NAME(name) = FloatOption(#name, dfault, min, max);
    PARAMS
#undef X

OptionsContainer parameters;

public:

#define X(name, dfault, min, max) float name = dfault;
    PARAMS
#undef X

    SearchParameters() {
#define X(name, dfault, min, max) parameters.push_back(&OPTION_NAME(name));
        PARAMS
#undef X
    }

    void update() {
#define X(name, dfault, min, max) name = OPTION_NAME(name).currentValue;
        PARAMS
#undef X
    }

    OptionsContainer& getOptions() {
        return parameters;
    }
};
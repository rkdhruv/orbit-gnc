#pragma once

#include <stdexcept>
#include <string>

// Precondition check that stays active in every build type (unlike assert, which NDEBUG
// removes). Throws std::invalid_argument naming the failed condition. `cond` is evaluated
// exactly once; the do/while(0) makes the macro a single statement inside an unbraced if/else.
// Web build: Emscripten disables exception catching by default, so there a failed check
// aborts instead. Recovering from it in simapp would need -sDISABLE_EXCEPTION_CATCHING=0.
#define SIMCORE_CHECK(cond, msg)                                                                   \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            throw std::invalid_argument(std::string("SIMCORE_CHECK(" #cond ") failed: ") + (msg)); \
        }                                                                                          \
    } while (0)

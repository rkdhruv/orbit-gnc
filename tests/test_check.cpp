// M1.1: SIMCORE_CHECK(cond, msg) — input validation that survives release builds.
//
// Unlike assert(), it must not depend on NDEBUG: these tests run under every preset, including
// release, and must pass in all of them.

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include "simcore/check.hpp"

TEST(Check, PassingConditionDoesNothing) {
    EXPECT_NO_THROW(SIMCORE_CHECK(1 + 1 == 2, "arithmetic works"));
}

TEST(Check, FailingConditionThrowsInvalidArgumentCarryingMessage) {
    EXPECT_THROW(SIMCORE_CHECK(false, "always fails"), std::invalid_argument);
    try {
        SIMCORE_CHECK(2 < 1, "dt must be finite");
        FAIL() << "SIMCORE_CHECK(false, ...) did not throw";
    } catch (const std::invalid_argument& e) {
        // The caller's message must reach whoever catches it (exact formatting is free).
        EXPECT_NE(std::string(e.what()).find("dt must be finite"), std::string::npos) << e.what();
    }
}

TEST(Check, ConditionIsEvaluatedExactlyOnce) {
    // Macro hygiene: a condition with side effects must not be evaluated twice (or zero times).
    int calls = 0;
    SIMCORE_CHECK(++calls > 0, "side effect");
    EXPECT_EQ(calls, 1);
}

TEST(Check, BehavesAsASingleStatement) {
    // Macro hygiene: usable as the body of an unbraced if/else (needs the do { } while (0) form or
    // equivalent); a bare `if (!(cond)) throw ...;` expansion would steal the else.
    bool took_else = false;
    const bool guard = false;
    if (guard)
        SIMCORE_CHECK(true, "unused");
    else
        took_else = true;
    EXPECT_TRUE(took_else);
}

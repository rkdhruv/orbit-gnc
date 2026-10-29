// M1.3: Brent's method for a bracketed root of a scalar function.
//
// Contract: g(a) and g(b) must not have the same sign; the result x satisfies |x - x*| <= xtol for
// the (or a) sign change x* inside [a, b]. Brent combines inverse quadratic interpolation and the
// secant step (superlinear near a simple root) with a bisection fallback, so it never does worse
// than bisection by more than a constant factor and keeps the bracket at every step.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>

#include "simcore/events/brent.hpp"

namespace {

using simcore::brent_root;

}  // namespace

// cos x = x has the single root x* = 0.7390851332151607 (the Dottie number) in [0, 1].
// Tolerance = the xtol passed in: that is the contract, not a fitted number.
TEST(Brent, FindsRootOfCosXMinusX) {
    constexpr double kRoot = 0.7390851332151607;
    for (const double xtol : {1e-6, 1e-10, 1e-14}) {
        SCOPED_TRACE(xtol);
        const double x = brent_root([](double v) { return std::cos(v) - v; }, 0.0, 1.0, xtol);
        EXPECT_NEAR(x, kRoot, xtol);
    }
}

// Wallis's cubic x^3 - 2x - 5 = 0, root 2.0945514815423265 in [2, 3] (Brent's own test case).
TEST(Brent, FindsRootOfWallisCubic) {
    const double x =
        brent_root([](double v) { return v * v * v - 2.0 * v - 5.0; }, 2.0, 3.0, 1e-13);
    EXPECT_NEAR(x, 2.0945514815423265, 1e-13);
}

// The bracket may be given in either order, and g may be decreasing through the root.
TEST(Brent, AcceptsReversedBracketAndDecreasingFunction) {
    const double x = brent_root([](double v) { return v - std::cos(v); }, 1.0, 0.0, 1e-12);
    EXPECT_NEAR(x, 0.7390851332151607, 1e-12);
    const double y = brent_root([](double v) { return 1.0 - v * v; }, 0.0, 3.0, 1e-12);  // falls
    EXPECT_NEAR(y, 1.0, 1e-12);
}

// A root exactly at an endpoint is returned exactly, without iterating.
TEST(Brent, RootAtEndpointIsReturnedExactly) {
    EXPECT_EQ(brent_root([](double v) { return v - 2.0; }, 2.0, 5.0, 1e-12), 2.0);
    EXPECT_EQ(brent_root([](double v) { return v - 5.0; }, 2.0, 5.0, 1e-12), 5.0);
}

// Superlinear convergence on a smooth simple root: bisection needs ceil(log2(1 / 1e-14)) = 47
// halvings of [0, 1] to reach xtol = 1e-14. Brent's interpolation steps should need well under
// half of that; we require <= 23 evaluations (strictly better than half of bisection, counting
// the two endpoint evaluations). Measured: see the summary.
TEST(Brent, ConvergesFasterThanBisectionOnSmoothRoot) {
    int calls = 0;
    const double x = brent_root(
        [&calls](double v) {
            ++calls;
            return std::cos(v) - v;
        },
        0.0, 1.0, 1e-14);
    EXPECT_NEAR(x, 0.7390851332151607, 1e-14);
    EXPECT_LE(calls, 23);
}

// A jump (no root, just a sign change) still converges to the jump location: Brent falls back to
// bisection when interpolation steps fail to shrink the bracket.
TEST(Brent, ConvergesToSignChangeOfStepFunction) {
    const double x = brent_root([](double v) { return v < 0.3 ? -1.0 : 1.0; }, 0.0, 1.0, 1e-10);
    EXPECT_NEAR(x, 0.3, 1e-10);
}

TEST(Brent, RejectsInvalidArguments) {
    const auto g = [](double v) { return v - 0.5; };
    const double kNaN = std::numeric_limits<double>::quiet_NaN();
    const double kInf = std::numeric_limits<double>::infinity();
    // Same sign at both ends: no bracketed root.
    EXPECT_THROW((void)brent_root(g, 0.6, 1.0, 1e-12), std::invalid_argument);
    EXPECT_THROW((void)brent_root(g, kNaN, 1.0, 1e-12), std::invalid_argument);
    EXPECT_THROW((void)brent_root(g, 0.0, kInf, 1e-12), std::invalid_argument);
    EXPECT_THROW((void)brent_root(g, 0.0, 1.0, 0.0), std::invalid_argument);
    EXPECT_THROW((void)brent_root(g, 0.0, 1.0, -1e-12), std::invalid_argument);
    EXPECT_THROW((void)brent_root(g, 0.0, 1.0, kNaN), std::invalid_argument);
    // g returning NaN at an endpoint can't establish a bracket.
    EXPECT_THROW((void)brent_root([](double) { return std::nan(""); }, 0.0, 1.0, 1e-12),
                 std::invalid_argument);
}

// Iteration cap: 2 iterations can't reach 1e-14 on [0, 1] (each step at best squares the error
// from ~0.1), so it must throw rather than return an unconverged value.
TEST(Brent, ThrowsWhenIterationCapIsExceeded) {
    EXPECT_THROW((void)brent_root([](double v) { return std::cos(v) - v; }, 0.0, 1.0, 1e-14, 2),
                 std::invalid_argument);
}

// M1.1: Epoch — absolute TT time stored as (int64 whole seconds since J2000.0, double frac in
// [0,1)).
//
// Why split: a bare double holding "seconds since J2000" at ~30 years (9.47e8 s) has an ulp of
// 2^(29-52) = 1.19e-7 s, so every addition is rounded to ~0.1 microsecond. The split form keeps the
// fraction in [0,1), where the ulp is <= 2^-53 = 1.1e-16 s, so arithmetic is accurate to ~1e-16 s
// regardless of how far the epoch is from J2000.
//
// Many values below (0.25, 0.5, 0.75, 1.5, ...) are dyadic rationals, exactly representable in
// binary, and every intermediate sum/difference of them is also exact. Those tests use EXPECT_EQ:
// any deviation is a logic bug, not roundoff.

#include <gtest/gtest.h>

#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "simcore/orbits/epoch.hpp"

namespace {

using simcore::Epoch;

// 30 Julian years after J2000.0: 30 * 365.25 d * 86400 s/d = 946,728,000 s (an exact integer).
constexpr std::int64_t kThirtyYears_s = 946'728'000;

// The class invariant every public operation must preserve.
void expect_normalised(const Epoch& t) {
    EXPECT_GE(t.fractional_seconds(), 0.0);
    EXPECT_LT(t.fractional_seconds(), 1.0);
}

void expect_epoch(const Epoch& t, std::int64_t whole_s, double frac_s) {
    EXPECT_EQ(t.whole_seconds(), whole_s);
    EXPECT_EQ(t.fractional_seconds(), frac_s);
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------------------------

TEST(Epoch, DefaultIsJ2000) {
    const Epoch t{};
    expect_epoch(t, 0, 0.0);
    expect_epoch(Epoch::j2000(), 0, 0.0);
    EXPECT_EQ(t, Epoch::j2000());
    EXPECT_EQ(Epoch::from_tt_seconds_since_j2000(0), Epoch::j2000());
}

TEST(Epoch, FactoryKeepsAlreadyNormalisedInput) {
    expect_epoch(Epoch::from_tt_seconds_since_j2000(kThirtyYears_s, 0.25), kThirtyYears_s, 0.25);
    expect_epoch(Epoch::from_tt_seconds_since_j2000(-5, 0.5), -5, 0.5);  // before J2000
}

// frac outside [0,1) must be carried into the whole part: value = whole + frac is preserved and
// 0 <= frac < 1 afterwards. All inputs are dyadic, so results are exact.
TEST(Epoch, FactoryNormalisesFractionAboveOne) {
    expect_epoch(Epoch::from_tt_seconds_since_j2000(10, 1.5), 11, 0.5);  // 10 + 1.5   = 11.5
    expect_epoch(Epoch::from_tt_seconds_since_j2000(0, 2.0), 2, 0.0);    // 0 + 2      = 2
    expect_epoch(Epoch::from_tt_seconds_since_j2000(10, 1.0), 11, 0.0);  // frac == 1 is not allowed
    expect_epoch(Epoch::from_tt_seconds_since_j2000(0, 1.0e6 + 0.75), 1'000'000, 0.75);
}

TEST(Epoch, FactoryNormalisesNegativeFraction) {
    expect_epoch(Epoch::from_tt_seconds_since_j2000(10, -0.25), 9, 0.75);  // 10 - 0.25  = 9.75
    expect_epoch(Epoch::from_tt_seconds_since_j2000(10, -3.5), 6, 0.5);    // 10 - 3.5   = 6.5
    expect_epoch(Epoch::from_tt_seconds_since_j2000(5, -1.0), 4, 0.0);     // exact integer carry
    expect_epoch(Epoch::from_tt_seconds_since_j2000(0, -0.25), -1, 0.75);  // crosses J2000: -0.25
}

// Classic normalisation bug: frac = -1e-20 -> floor gives -1 and frac + 1 rounds to exactly 1.0,
// violating frac < 1. The true value 10 - 1e-20 must be stored as the nearest representable epoch,
// i.e. within one ulp of 1.0 (2.2e-16 s) of exactly 10 s, with the invariant intact.
TEST(Epoch, FactoryTinyNegativeFractionStaysNormalised) {
    const Epoch t = Epoch::from_tt_seconds_since_j2000(10, -1e-20);
    expect_normalised(t);
    EXPECT_NEAR(t - Epoch::from_tt_seconds_since_j2000(10), 0.0,
                std::numeric_limits<double>::epsilon());
}

// ---------------------------------------------------------------------------------------------
// Arithmetic
// ---------------------------------------------------------------------------------------------

TEST(Epoch, AddSubtractDyadicOffsetsAreExact) {
    const Epoch t0 = Epoch::from_tt_seconds_since_j2000(100, 0.25);

    expect_epoch(t0 + 0.5, 100, 0.75);
    expect_epoch(t0 + 0.75, 101, 0.0);  // carry into whole seconds
    expect_epoch(t0 + 1234.5, 1334, 0.75);
    expect_epoch(t0 - 0.5, 99, 0.75);               // borrow from whole seconds
    expect_epoch(t0 - 200.5, -101, 0.75);           // 100.25 - 200.5 = -100.25 = -101 + 0.75
    expect_epoch(t0 + 3.0e9, 3'000'000'100, 0.25);  // ~95 years, integer-valued double

    // Epoch - dt is the same as Epoch + (-dt).
    EXPECT_EQ(t0 - 0.25, t0 + (-0.25));

    Epoch t = t0;
    t += 0.875;
    expect_epoch(t, 101, 0.125);
    t += -1.25;
    expect_epoch(t, 99, 0.875);
}

TEST(Epoch, TinyNegativeIncrementStaysNormalised) {
    // Same bug as the factory case, reached through operator+=.
    Epoch t = Epoch::from_tt_seconds_since_j2000(10);
    t += -1e-20;
    expect_normalised(t);
    EXPECT_NEAR(t - Epoch::from_tt_seconds_since_j2000(10), 0.0,
                std::numeric_limits<double>::epsilon());
}

TEST(Epoch, DifferenceIsSignedSeconds) {
    const Epoch a = Epoch::from_tt_seconds_since_j2000(5, 0.25);
    const Epoch b = Epoch::from_tt_seconds_since_j2000(7, 0.75);
    EXPECT_EQ(b - a, 2.5);  // 7.75 - 5.25
    EXPECT_EQ(a - b, -2.5);
    EXPECT_EQ(a - a, 0.0);
}

// (t + dt) - t == dt and (t + dt) - dt == t for a non-dyadic dt (0.1 s is not exact in binary).
// Tolerance: the fractional parts are < 1, so each add/subtract on them rounds by at most
// 2^-53 = 1.1e-16 s; a handful of such operations stays below 4 * epsilon(1.0) = 8.9e-16 s.
TEST(Epoch, AddThenDifferenceRoundTrips) {
    const Epoch t0 = Epoch::from_tt_seconds_since_j2000(kThirtyYears_s, 0.123456789);
    constexpr double kTol_s = 4.0 * std::numeric_limits<double>::epsilon();

    for (const double dt : {0.1, -0.1, 37.3, -86400.7, 1.0e7 + 0.3}) {
        SCOPED_TRACE(dt);
        const Epoch t1 = t0 + dt;
        expect_normalised(t1);
        // For |dt| ~ 1e7 the input double itself only carries ulp(1e7) = 1.9e-9 s of resolution,
        // but that is a property of dt, not of Epoch: the difference must reproduce dt's value.
        EXPECT_NEAR(t1 - t0, dt, kTol_s);
        EXPECT_NEAR((t1 - dt) - t0, 0.0, kTol_s);
    }
}

// The headline reason for the split representation.
//   Bare double:  t = 946728000.123456789 s; (t + 1e-3) - t = 1.0000467e-3 s,
//                 i.e. an error of 4.67e-8 s (= 0.39 ulp at 1.19e-7 s/ulp).
//   Split Epoch:  frac 0.123456789 + 0.001 rounds by <= ulp(0.12)/2 = 6.9e-18 s and the
//                 difference of two nearby fractions is exact (Sterbenz), so error ~1e-17 s.
// Tolerance 1e-15 s: a few ulps of 1.0 (covers any sane implementation of the fraction arithmetic)
// while being 7 orders of magnitude tighter than the 4.7e-8 s a bare double gets — and far
// "well under 1 ns".
TEST(Epoch, OneMillisecondAtThirtyYearsIsResolvedToSubNanosecond) {
    constexpr double kTol_s = 1e-15;
    const Epoch t0 = Epoch::from_tt_seconds_since_j2000(kThirtyYears_s, 0.123456789);
    const Epoch t1 = t0 + 1e-3;
    EXPECT_NEAR(t1 - t0, 1e-3, kTol_s);
    EXPECT_NEAR((t1 - 1e-3) - t0, 0.0, kTol_s);

    // Same claim when the 1 ms step carries across a whole-second boundary.
    const Epoch c0 = Epoch::from_tt_seconds_since_j2000(kThirtyYears_s, 0.9995);
    const Epoch c1 = c0 + 1e-3;
    EXPECT_EQ(c1.whole_seconds(), kThirtyYears_s + 1);
    expect_normalised(c1);
    EXPECT_NEAR(c1 - c0, 1e-3, kTol_s);
}

// 86,400 steps of exactly 1 s: integer-valued increments never touch the fraction (0.25 + 1.0 and
// 1.25 - 1.0 are exact), so after one day the epoch must be exactly whole + 86400 with frac intact.
TEST(Epoch, OneDayOfOneSecondStepsIsExact) {
    const Epoch t0 = Epoch::from_tt_seconds_since_j2000(kThirtyYears_s, 0.25);
    Epoch t = t0;
    for (int i = 0; i < 86'400; ++i) {
        t += 1.0;
    }
    expect_epoch(t, kThirtyYears_s + 86'400, 0.25);
}

// 1e6 steps of 1 ms starting 30 years after J2000 must land 1000 s later.
// Bound on the split representation's drift:
//   - each frac + 1e-3 (result in [0,1)) rounds by at most half an ulp of 1, 2^-54 = 5.6e-17 s;
//     wrapping (frac - 1.0 for frac in [1,2)) is exact.  Worst case: 1e6 * 5.6e-17 = 5.6e-11 s.
//   - double(1e-3) = 1e-3 + 2.1e-20, so 1e6 steps add a representation bias of 2.1e-14 s.
//   => |error| < 5.6e-11 s; tolerance 1e-10 s (~2x margin). Measured in IEEE double: 6.7e-13 s.
// Contrast: a bare double accumulator at 9.47e8 s rounds every 1e-3 step by the same 0.39 ulp in
// the same direction, drifting by 0.0467 s after 1e6 steps — 9 orders of magnitude worse.
TEST(Epoch, MillionMillisecondStepsDoNotDrift) {
    constexpr int kSteps = 1'000'000;
    constexpr double kTol_s = 1e-10;
    const Epoch t0 = Epoch::from_tt_seconds_since_j2000(kThirtyYears_s);

    Epoch t = t0;
    for (int i = 0; i < kSteps; ++i) {
        t += 1e-3;
    }
    expect_normalised(t);
    EXPECT_NEAR(t - t0, 1000.0, kTol_s);
    EXPECT_NEAR(t - Epoch::from_tt_seconds_since_j2000(kThirtyYears_s + 1000), 0.0, kTol_s);

    // Stepping back the same way returns to t0; the bound simply doubles.
    for (int i = 0; i < kSteps; ++i) {
        t += -1e-3;
    }
    expect_normalised(t);
    EXPECT_NEAR(t - t0, 0.0, 2.0 * kTol_s);
}

// ---------------------------------------------------------------------------------------------
// Ordering
// ---------------------------------------------------------------------------------------------

TEST(Epoch, OrderingComparesWholeThenFraction) {
    const Epoch a = Epoch::from_tt_seconds_since_j2000(5, 0.9);
    const Epoch b = Epoch::from_tt_seconds_since_j2000(6, 0.1);
    const Epoch c = Epoch::from_tt_seconds_since_j2000(6, 0.2);

    EXPECT_LT(a, b);  // whole second dominates a larger fraction
    EXPECT_LT(b, c);  // same whole second: fraction decides
    EXPECT_GT(c, a);
    EXPECT_LE(b, b);
    EXPECT_GE(b, b);
    EXPECT_NE(a, b);
    EXPECT_FALSE(b < b);

    // Before J2000: -0.5 s < 0 s.
    EXPECT_LT(Epoch::from_tt_seconds_since_j2000(-1, 0.5), Epoch::j2000());
}

TEST(Epoch, EqualityHoldsAcrossDifferentInputSplits) {
    // 4 + 1.5 and 5 + 0.5 are the same instant; normalisation makes the defaulted == agree.
    EXPECT_EQ(Epoch::from_tt_seconds_since_j2000(4, 1.5),
              Epoch::from_tt_seconds_since_j2000(5, 0.5));
    EXPECT_EQ(Epoch::from_tt_seconds_since_j2000(6, -0.5),
              Epoch::from_tt_seconds_since_j2000(5, 0.5));
    EXPECT_EQ(Epoch::from_tt_seconds_since_j2000(5) + 0.5,
              Epoch::from_tt_seconds_since_j2000(5, 0.5));
}

// ---------------------------------------------------------------------------------------------
// Input validation (SIMCORE_CHECK -> std::invalid_argument, active in every build type)
// ---------------------------------------------------------------------------------------------
//
// Why the 2^62 limit: an addend is split into integer and fractional parts (std::modf) and the
// integer part is cast to int64_t; converting a double outside the int64 range [-2^63, 2^63) is
// undefined behaviour. 2^62 s (~1.5e11 years)
// leaves a factor-2 headroom and is far beyond any physical epoch, so the check rejects only
// garbage (uninitialised or overflowed values), never a real time.
// The values used are exact doubles:
//   2^62                   = 4,611,686,018,427,387,904
//   nextafter(2^62, 0)     = 2^62 - 512 (ulp just below 2^62 is 2^(61-52) = 512)
//   2^61                   = 2,305,843,009,213,693,952
// None of the accepted cases below overflow whole_s (|result| < 2^62 < 2^63).

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kTwoTo62 = 0x1p62;
constexpr double kJustBelowTwoTo62 = 0x1.fffffffffffffp61;  // 2^62 - 512
constexpr double kTwoTo61 = 0x1p61;
constexpr std::int64_t kTwoTo62Minus512_i = (std::int64_t{1} << 62) - 512;
constexpr std::int64_t kTwoTo61_i = std::int64_t{1} << 61;

static_assert(kJustBelowTwoTo62 == kTwoTo62 - 512.0);
static_assert(kJustBelowTwoTo62 < kTwoTo62);

}  // namespace

TEST(Epoch, FactoryRejectsNonFiniteFraction) {
    EXPECT_THROW((void)Epoch::from_tt_seconds_since_j2000(0, kNaN), std::invalid_argument);
    EXPECT_THROW((void)Epoch::from_tt_seconds_since_j2000(0, kInf), std::invalid_argument);
    EXPECT_THROW((void)Epoch::from_tt_seconds_since_j2000(0, -kInf), std::invalid_argument);
    EXPECT_THROW((void)Epoch::from_tt_seconds_since_j2000(kThirtyYears_s, kNaN),
                 std::invalid_argument);
}

TEST(Epoch, FactoryRejectsFractionAtOrBeyondTwoTo62) {
    EXPECT_THROW((void)Epoch::from_tt_seconds_since_j2000(0, kTwoTo62), std::invalid_argument);
    EXPECT_THROW((void)Epoch::from_tt_seconds_since_j2000(0, -kTwoTo62), std::invalid_argument);
    EXPECT_THROW((void)Epoch::from_tt_seconds_since_j2000(0, 1e300), std::invalid_argument);
}

TEST(Epoch, FactoryAcceptsFractionJustBelowTwoTo62) {
    // Integer-valued doubles: the carry is exact, so EXPECT_EQ.
    Epoch t;
    EXPECT_NO_THROW(t = Epoch::from_tt_seconds_since_j2000(0, kJustBelowTwoTo62));
    expect_epoch(t, kTwoTo62Minus512_i, 0.0);
    EXPECT_NO_THROW(t = Epoch::from_tt_seconds_since_j2000(0, -kJustBelowTwoTo62));
    expect_epoch(t, -kTwoTo62Minus512_i, 0.0);
}

TEST(Epoch, AdditionRejectsNonFiniteOffset) {
    const Epoch t0 = Epoch::from_tt_seconds_since_j2000(kThirtyYears_s, 0.25);
    for (const double bad : {kNaN, kInf, -kInf}) {
        SCOPED_TRACE(bad);
        Epoch t = t0;
        EXPECT_THROW(t += bad, std::invalid_argument);
        EXPECT_THROW(t -= bad, std::invalid_argument);
        EXPECT_THROW((void)(t0 + bad), std::invalid_argument);
        EXPECT_THROW((void)(t0 - bad), std::invalid_argument);
    }
}

TEST(Epoch, AdditionRejectsOffsetAtOrBeyondTwoTo62) {
    const Epoch t0 = Epoch::j2000();
    for (const double bad : {kTwoTo62, -kTwoTo62, 1e300, -1e300}) {
        SCOPED_TRACE(bad);
        Epoch t = t0;
        EXPECT_THROW(t += bad, std::invalid_argument);
        EXPECT_THROW(t -= bad, std::invalid_argument);
        EXPECT_THROW((void)(t0 + bad), std::invalid_argument);
        EXPECT_THROW((void)(t0 - bad), std::invalid_argument);
    }
}

TEST(Epoch, AdditionAcceptsOffsetJustBelowTwoTo62) {
    // From J2000 none of these overflow whole_s; integer-valued, so results are exact.
    const Epoch t0 = Epoch::j2000();
    Epoch t;
    EXPECT_NO_THROW(t = t0 + kJustBelowTwoTo62);
    expect_epoch(t, kTwoTo62Minus512_i, 0.0);
    EXPECT_NO_THROW(t = t0 - kJustBelowTwoTo62);
    expect_epoch(t, -kTwoTo62Minus512_i, 0.0);
    EXPECT_NO_THROW(t = t0 + kTwoTo61);
    expect_epoch(t, kTwoTo61_i, 0.0);

    Epoch u = t0;
    EXPECT_NO_THROW(u += kTwoTo61);
    EXPECT_NO_THROW(u -= kTwoTo61);
    EXPECT_EQ(u, t0);
}

// Validation happens before any member is touched: a rejected += / -= leaves the epoch unchanged
// (strong exception guarantee), so a caller that catches the error still holds a valid time.
TEST(Epoch, RejectedIncrementLeavesEpochUnchanged) {
    const Epoch t0 = Epoch::from_tt_seconds_since_j2000(kThirtyYears_s, 0.25);
    for (const double bad : {kNaN, kInf, -kInf, kTwoTo62, -kTwoTo62}) {
        SCOPED_TRACE(bad);
        Epoch t = t0;
        EXPECT_THROW(t += bad, std::invalid_argument);
        EXPECT_EQ(t, t0);
        EXPECT_THROW(t -= bad, std::invalid_argument);
        EXPECT_EQ(t, t0);
    }
}

// ---------------------------------------------------------------------------------------------
// No silent truncation of a bare double into whole seconds
// ---------------------------------------------------------------------------------------------
//
// from_tt_seconds_since_j2000(1.5) would otherwise convert 1.5 -> int64 1 and silently drop half a
// second. The single-argument floating-point overload must be deleted, while integer literals
// (int, int64) with or without a fraction must still work.
//
// This has to be a concept (a requires-expression inside a template): a bare
// `static_assert(!requires { Epoch::from_tt_seconds_since_j2000(1.5); })` outside a template is a
// hard "call to deleted function" error rather than `false`, so it could never pass.
//
// Note for the implementation: a plain `static Epoch from_tt_seconds_since_j2000(double) = delete;`
// makes `from_tt_seconds_since_j2000(10)` AMBIGUOUS (int -> int64 and int -> double are both
// standard conversions of the same rank), which the int checks below catch. A constrained
// template, `template <std::floating_point T> static Epoch from_tt_seconds_since_j2000(T) =
// delete;`, rejects double/float/long double and stays out of overload resolution for integers.
namespace {

template <class T>
concept FactoryAcceptsSingle = requires(T x) { Epoch::from_tt_seconds_since_j2000(x); };

template <class W, class F>
concept FactoryAcceptsPair = requires(W w, F f) { Epoch::from_tt_seconds_since_j2000(w, f); };

}  // namespace

static_assert(!FactoryAcceptsSingle<double>, "single double would be truncated to whole seconds");
static_assert(!FactoryAcceptsSingle<float>, "single float would be truncated to whole seconds");
static_assert(!FactoryAcceptsSingle<long double>, "single long double would be truncated");
static_assert(FactoryAcceptsSingle<int>, "integer literal, e.g. from_tt_seconds_since_j2000(10)");
static_assert(FactoryAcceptsSingle<std::int64_t>);
static_assert(FactoryAcceptsPair<int, double>, "e.g. from_tt_seconds_since_j2000(10, 0.5)");
static_assert(FactoryAcceptsPair<std::int64_t, double>);

TEST(Epoch, IntegerLiteralFactoryStillWorks) {
    // Runtime counterpart of the static_asserts above (int literal first argument).
    expect_epoch(Epoch::from_tt_seconds_since_j2000(10), 10, 0.0);
    expect_epoch(Epoch::from_tt_seconds_since_j2000(10, 0.5), 10, 0.5);
}

// ---------------------------------------------------------------------------------------------
// int64 overflow of the whole-seconds counter
// ---------------------------------------------------------------------------------------------
//
// The 2^62 input limit makes the double -> int64 cast safe, but whole_s + carry can still leave
// the int64 range when whole_s is already near INT64_MAX / INT64_MIN. Signed overflow is UB, so it
// must be detected and rejected with std::invalid_argument. These epochs (~2.9e11 years) are
// unphysical; the point is that garbage in gives an exception, never UB or a wrapped-around time.
// All values are integers or dyadic fractions, so accepted results are exact (EXPECT_EQ).

namespace {

constexpr std::int64_t kI64Max = std::numeric_limits<std::int64_t>::max();  //  2^63 - 1
constexpr std::int64_t kI64Min = std::numeric_limits<std::int64_t>::min();  // -2^63

}  // namespace

TEST(Epoch, FactoryRejectsCarryThatOverflowsInt64) {
    // INT64_MAX + 1.0: the carry of 1 pushes whole past INT64_MAX.
    EXPECT_THROW((void)Epoch::from_tt_seconds_since_j2000(kI64Max, 1.0), std::invalid_argument);
    // INT64_MIN - 0.5 = (INT64_MIN - 1) + 0.5: the borrow of 1 pushes whole below INT64_MIN.
    EXPECT_THROW((void)Epoch::from_tt_seconds_since_j2000(kI64Min, -0.5), std::invalid_argument);
}

TEST(Epoch, FactoryAcceptsExtremeWholeSecondsWithoutCarry) {
    // Boundary: no carry is needed, so nothing overflows.
    Epoch t;
    EXPECT_NO_THROW(t = Epoch::from_tt_seconds_since_j2000(kI64Max, 0.5));
    expect_epoch(t, kI64Max, 0.5);
    EXPECT_NO_THROW(t = Epoch::from_tt_seconds_since_j2000(kI64Min, 0.5));
    expect_epoch(t, kI64Min, 0.5);
}

TEST(Epoch, IncrementThatOverflowsInt64ThrowsAndLeavesEpochUnchanged) {
    // 0.75 + 0.5 = 1.25 carries 1 into whole = INT64_MAX: overflow. The integer part of dt is 0,
    // so this passes the |dt| < 2^62 check and fails only at the final carry, i.e. it exercises
    // the strong guarantee on the late (normalisation) path, not just up-front validation.
    const Epoch t_max = Epoch::from_tt_seconds_since_j2000(kI64Max, 0.75);
    Epoch t = t_max;
    EXPECT_THROW(t += 0.5, std::invalid_argument);
    EXPECT_EQ(t, t_max);
    EXPECT_THROW((void)(t_max + 0.5), std::invalid_argument);

    // Mirror image at INT64_MIN: 0.25 - 0.5 = -0.25 borrows 1 from whole = INT64_MIN.
    const Epoch t_min = Epoch::from_tt_seconds_since_j2000(kI64Min, 0.25);
    Epoch u = t_min;
    EXPECT_THROW(u -= 0.5, std::invalid_argument);
    EXPECT_EQ(u, t_min);

    // A whole-second step that overflows (integer part of dt is nonzero) is also rejected.
    Epoch w = t_max;
    EXPECT_THROW(w += 1.0, std::invalid_argument);
    EXPECT_EQ(w, t_max);
}

TEST(Epoch, IncrementLandingExactlyOnInt64MaxIsAccepted) {
    // (INT64_MAX - 1) + 0.75 + 0.5 = INT64_MAX + 0.25: largest representable whole, no overflow.
    Epoch t = Epoch::from_tt_seconds_since_j2000(kI64Max - 1, 0.75);
    EXPECT_NO_THROW(t += 0.5);
    expect_epoch(t, kI64Max, 0.25);
}

TEST(Epoch, DifferenceThatOverflowsInt64Throws) {
    // INT64_MAX - INT64_MIN = 2^64 - 1 does not fit in int64.
    EXPECT_THROW((void)(Epoch::from_tt_seconds_since_j2000(kI64Max) -
                        Epoch::from_tt_seconds_since_j2000(kI64Min)),
                 std::invalid_argument);
    // -2 - INT64_MAX = -2^63 - 1: one below INT64_MIN.
    EXPECT_THROW((void)(Epoch::from_tt_seconds_since_j2000(-2) -
                        Epoch::from_tt_seconds_since_j2000(kI64Max)),
                 std::invalid_argument);
}

TEST(Epoch, DifferenceAtInt64BoundaryIsAccepted) {
    // -1 - INT64_MAX = -2^63 = INT64_MIN exactly: fits, and -2^63 is an exact double.
    double d = 0.0;
    EXPECT_NO_THROW(d = Epoch::from_tt_seconds_since_j2000(-1) -
                        Epoch::from_tt_seconds_since_j2000(kI64Max));
    EXPECT_EQ(d, -0x1p63);
}

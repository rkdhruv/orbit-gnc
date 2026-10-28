#include "simcore/orbits/epoch.hpp"

#include <cmath>

#include "simcore/check.hpp"

namespace simcore {

namespace {

// Largest magnitude accepted for a seconds argument. Below 2^62 the cast of its integer part
// to int64 is well defined with headroom; NaN/inf fail the comparison too.
constexpr double kMaxSeconds = 0x1p62;

void check_seconds(double s) {
    SIMCORE_CHECK(std::abs(s) < kMaxSeconds, "seconds must be finite with |s| < 2^62");
}

std::int64_t checked_add(std::int64_t a, std::int64_t b) {
    std::int64_t sum = 0;
    SIMCORE_CHECK(!__builtin_add_overflow(a, b, &sum), "Epoch whole seconds overflow int64");
    return sum;
}

// Move the integer part of frac_s into whole_s so that 0 <= frac_s < 1 afterwards.
// Requires |frac_s| < 2^62 (already checked).
void normalise(std::int64_t& whole_s, double& frac_s) {
    const double carry = std::floor(frac_s);
    whole_s = checked_add(whole_s, static_cast<std::int64_t>(carry));
    // Exact for frac_s >= 0. For frac_s < 0 the add of |carry| rounds, and a tiny negative
    // frac (e.g. -1e-20) gives -1e-20 + 1 == 1.0 exactly, breaking the invariant: carry again.
    frac_s -= carry;
    if (frac_s >= 1.0) {
        whole_s = checked_add(whole_s, 1);
        frac_s = 0.0;
    }
}

}  // namespace

Epoch::Epoch(std::int64_t whole_s, double frac_s) : whole_s_(whole_s), frac_s_(frac_s) {
    check_seconds(frac_s_);
    normalise(whole_s_, frac_s_);
}

Epoch Epoch::from_tt_seconds_since_j2000(std::int64_t whole_s, double frac_s) {
    return {whole_s, frac_s};
}

Epoch& Epoch::operator+=(double dt_s) {
    check_seconds(dt_s);
    // Split dt exactly into integer and fractional parts (modf is exact; both parts carry dt's
    // sign), so a large dt never meets the small fraction in one floating-point add.
    double dt_whole = 0.0;
    const double dt_frac = std::modf(dt_s, &dt_whole);  // in (-1, 1)
    // Work on copies so a throw leaves *this unchanged.
    std::int64_t whole = checked_add(whole_s_, static_cast<std::int64_t>(dt_whole));
    // In (-1, 2). This is the only rounding when the sum is >= 0; a negative sum rounds once
    // more when normalise() adds 1.
    double frac = frac_s_ + dt_frac;
    normalise(whole, frac);
    whole_s_ = whole;
    frac_s_ = frac;
    return *this;
}

double operator-(const Epoch& a, const Epoch& b) {
    std::int64_t dwhole = 0;
    SIMCORE_CHECK(!__builtin_sub_overflow(a.whole_s_, b.whole_s_, &dwhole),
                  "Epoch difference overflows int64");
    // Integer difference is exact; only the final small sum rounds.
    return static_cast<double>(dwhole) + (a.frac_s_ - b.frac_s_);
}

}  // namespace simcore

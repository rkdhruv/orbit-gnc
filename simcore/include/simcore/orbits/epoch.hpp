#pragma once

#include <compare>
#include <concepts>
#include <cstdint>

namespace simcore {

// Absolute time on the TT scale, stored as whole seconds since J2000.0 TT plus a fractional
// second in [0, 1). J2000.0 = JD 2451545.0 TT = 2000-01-01 12:00:00 TT.
//
// Why split: a single double holding ~1e9 s (30 years) has an ulp of ~1.2e-7 s, so every
// addition rounds to ~0.1 us. Keeping the fraction in [0, 1) gives an ulp <= 1.1e-16 s there,
// independent of how far the epoch is from J2000. Every operation renormalises so that the
// defaulted ordering (whole first, then frac) is correct.
//
// Valid range: whole seconds fit in int64 (+/- 2.9e11 years). Inputs must be finite with
// magnitude < 2^62 s; violations, and any result outside int64, throw std::invalid_argument.
// A throwing operation leaves the epoch unchanged.
class Epoch {
public:
    Epoch() = default;  // J2000.0 TT

    static Epoch j2000() { return {}; }

    // whole_s + frac_s seconds after J2000.0 TT. frac_s may be outside [0, 1) or negative;
    // it is carried into the whole part. Note whole_s is an integer: passing a double there
    // truncates (only -Wconversion catches it), so put sub-second time in frac_s.
    static Epoch from_tt_seconds_since_j2000(std::int64_t whole_s, double frac_s = 0.0);

    // A lone floating-point argument would be silently truncated to whole seconds. A template
    // (rather than a plain double overload) keeps integer arguments unambiguous.
    template <std::floating_point T>
    static Epoch from_tt_seconds_since_j2000(T) = delete;

    [[nodiscard]] std::int64_t whole_seconds() const { return whole_s_; }
    [[nodiscard]] double fractional_seconds() const { return frac_s_; }  // in [0, 1)

    // Advance by dt_s seconds (may be negative).
    Epoch& operator+=(double dt_s);
    Epoch& operator-=(double dt_s) { return *this += -dt_s; }

    friend Epoch operator+(Epoch t, double dt_s) { return t += dt_s; }
    friend Epoch operator-(Epoch t, double dt_s) { return t -= dt_s; }

    // Signed difference a - b in seconds.
    friend double operator-(const Epoch& a, const Epoch& b);

    friend auto operator<=>(const Epoch&, const Epoch&) = default;

private:
    Epoch(std::int64_t whole_s, double frac_s);  // validates and normalises

    std::int64_t whole_s_{0};
    double frac_s_{0.0};
};

}  // namespace simcore

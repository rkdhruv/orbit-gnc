#pragma once

// Test-only orbit fixtures shared by the M1.2 integrator tests. No Kepler solver exists yet
// (M1.4), so every reference here is analytic without one:
//   (a) circular orbit: uniform rotation, r(t) = r0 [cos(nt) p + sin(nt) q];
//   (b) eccentric orbit started at periapsis: after exactly one period T it is back at periapsis;
//   (c) specific energy E = v^2/2 - mu/r and angular momentum h = r x v are invariants.
//
// Both orbits lie in a plane inclined 51.6 deg (ISS-like) with RAAN 30 deg, so all three ECI
// components of r and v are non-zero and exercised (an equatorial orbit would keep z == 0 exactly
// and hide component-mixing bugs).

#include <cmath>
#include <limits>
#include <numbers>
#include <vector>

#include <Eigen/Dense>

#include "simcore/constants.hpp"
#include "simcore/dynamics/force_model.hpp"
#include "simcore/dynamics/point_mass_gravity.hpp"
#include "simcore/dynamics/state.hpp"
#include "simcore/orbits/epoch.hpp"

namespace orbit_fixtures {

using simcore::Epoch;
using simcore::State;
namespace earth = simcore::earth;

inline constexpr double kPi = std::numbers::pi;
inline constexpr double kEps = std::numeric_limits<double>::epsilon();  // 2.22e-16

// Start epoch ~25 years after J2000 with a non-zero fraction, so Epoch arithmetic is exercised
// away from zero. 0.25 is dyadic: adding integer or dyadic step sizes to it stays exact.
inline Epoch start_epoch() {
    return Epoch::from_tt_seconds_since_j2000(800'000'000, 0.25);
}

// Orbit-plane basis: p = periapsis / initial position direction, q = initial velocity direction.
//   p = (cos RAAN, sin RAAN, 0)
//   q = (-sin RAAN cos i, cos RAAN cos i, sin i)
// p and q are orthonormal (to roundoff) for any i, RAAN.
inline Eigen::Vector3d p_hat() {
    const double raan = 30.0 * kPi / 180.0;
    return {std::cos(raan), std::sin(raan), 0.0};
}
inline Eigen::Vector3d q_hat() {
    const double raan = 30.0 * kPi / 180.0;
    const double inc = 51.6 * kPi / 180.0;
    return {-std::sin(raan) * std::cos(inc), std::cos(raan) * std::cos(inc), std::sin(inc)};
}

// ---------------------------------------------------------------------------------------------
// (a) Circular 420 km orbit
// ---------------------------------------------------------------------------------------------
// r = R + 420 km = 6,378,137 + 420,000 = 6,798,137 m
// n = sqrt(mu / r^3) = sqrt(3.986004418e14 / 3.14168e20) = 1.126378e-3 rad/s
// T = 2 pi / n = 2 pi sqrt(r^3 / mu) = 5578.2227 s = 92.970 min
//   (CLAUDE.md's "~92.8 min" is a known error in the reference table; the derivation gives 92.97.)
// v = sqrt(mu / r) = 7657.27 m/s (CLAUDE.md: ~7.66 km/s, agrees)
inline constexpr double kR420_m = earth::radius + 420.0e3;
inline double circular_mean_motion_radps() {
    return std::sqrt(earth::mu / (kR420_m * kR420_m * kR420_m));
}
inline double circular_period_s() {
    return 2.0 * kPi / circular_mean_motion_radps();
}
inline double circular_speed_mps() {
    return std::sqrt(earth::mu / kR420_m);
}

inline State circular_initial_state() {
    return State{start_epoch(), kR420_m * p_hat(), circular_speed_mps() * q_hat()};
}

// Exact solution dt seconds after circular_initial_state(): uniform rotation by angle n dt.
//   r(t) = r0 [cos(n t) p + sin(n t) q]
//   v(t) = r0 n [-sin(n t) p + cos(n t) q]
inline State circular_analytic_state(double dt_s) {
    const double n = circular_mean_motion_radps();
    const double c = std::cos(n * dt_s);
    const double s = std::sin(n * dt_s);
    return State{start_epoch() + dt_s, kR420_m * (c * p_hat() + s * q_hat()),
                 kR420_m * n * (-s * p_hat() + c * q_hat())};
}

// ---------------------------------------------------------------------------------------------
// (b) Eccentric orbit, e = 0.7, periapsis at R + 420 km
// ---------------------------------------------------------------------------------------------
// r_p = 6,798,137 m
// a   = r_p / (1 - e) = 22,660,457 m
// r_a = a (1 + e) = 38,522,776 m
// T   = 2 pi sqrt(a^3 / mu) = 33,947.98 s (9.43 h)
// v_p = sqrt(mu (1 + e) / r_p)  (vis-viva: v^2 = mu (2/r - 1/a) at r = r_p) = 9983.86 m/s
// v_a = v_p r_p / r_a  (angular momentum conservation) = 1761.9 m/s
// E   = -mu / (2 a) = -8.795066e6 J/kg
inline constexpr double kEcc = 0.7;
inline constexpr double kPeriapsisRadius_m = kR420_m;
inline constexpr double kEccSemiMajorAxis_m = kPeriapsisRadius_m / (1.0 - kEcc);
inline constexpr double kApoapsisRadius_m = kEccSemiMajorAxis_m * (1.0 + kEcc);
inline double eccentric_period_s() {
    return 2.0 * kPi *
           std::sqrt(kEccSemiMajorAxis_m * kEccSemiMajorAxis_m * kEccSemiMajorAxis_m / earth::mu);
}
inline double eccentric_periapsis_speed_mps() {
    return std::sqrt(earth::mu * (1.0 + kEcc) / kPeriapsisRadius_m);
}

inline State eccentric_periapsis_state() {
    return State{start_epoch(), kPeriapsisRadius_m * p_hat(),
                 eccentric_periapsis_speed_mps() * q_hat()};
}

// Same orbit, started at apoapsis (opposite side, velocity antiparallel to q).
inline State eccentric_apoapsis_state() {
    const double v_a = eccentric_periapsis_speed_mps() * kPeriapsisRadius_m / kApoapsisRadius_m;
    return State{start_epoch(), -kApoapsisRadius_m * p_hat(), -v_a * q_hat()};
}

// ---------------------------------------------------------------------------------------------
// (c) Invariants of the two-body problem
// ---------------------------------------------------------------------------------------------
inline double specific_energy(const State& s) {
    return 0.5 * s.v_eci.squaredNorm() - earth::mu / s.r_eci.norm();
}
inline Eigen::Vector3d angular_momentum(const State& s) {
    return s.r_eci.cross(s.v_eci);
}

// ---------------------------------------------------------------------------------------------
// Instrumented force model
// ---------------------------------------------------------------------------------------------
// Point-mass gravity that also records every State it was evaluated at, so tests can check
// stage times, evaluation counts and that no evaluation happens outside the integration span.
class RecordingGravity final : public simcore::ForceModel {
public:
    [[nodiscard]] Eigen::Vector3d acceleration_eci(const State& s) const override {
        calls.push_back(s);
        return gravity_.acceleration_eci(s);
    }
    mutable std::vector<State> calls;

private:
    simcore::PointMassGravity gravity_;
};

}  // namespace orbit_fixtures

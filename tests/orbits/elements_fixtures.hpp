#pragma once

// Independent construction of states from classical elements for the M1.4 tests, so the tests
// don't rely on the implementation's own state_from_elements.
//
// Perifocal frame (P toward periapsis, Q 90 deg ahead in the orbit plane, W = P x Q):
//   p = a (1 - e^2)  (positive for ellipses and hyperbolas alike)
//   r_pf = p / (1 + e cos nu) [cos nu, sin nu, 0]
//   v_pf = sqrt(mu / p) [-sin nu, e + cos nu, 0]
// Rotation to ECI: R = Rz(raan) Rx(i) Rz(argp)  (active rotations; Vallado algorithm 10 "COE2RV",
// Curtis section 4.6).

#include <cmath>
#include <numbers>

#include <Eigen/Geometry>

#include "simcore/constants.hpp"
#include "simcore/dynamics/state.hpp"
#include "simcore/orbits/epoch.hpp"

namespace elements_fixtures {

inline constexpr double kPi = std::numbers::pi;
inline constexpr double kDeg = kPi / 180.0;

inline Eigen::Matrix3d perifocal_to_eci(double i, double raan, double argp) {
    return (Eigen::AngleAxisd(raan, Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(i, Eigen::Vector3d::UnitX()) *
            Eigen::AngleAxisd(argp, Eigen::Vector3d::UnitZ()))
        .toRotationMatrix();
}

inline simcore::State state_from(
    double a, double e, double i, double raan, double argp, double nu,
    simcore::Epoch t = simcore::Epoch::from_tt_seconds_since_j2000(800'000'000, 0.25)) {
    const double mu = simcore::earth::mu;
    const double p = a * (1.0 - e * e);
    const double r = p / (1.0 + e * std::cos(nu));
    const Eigen::Vector3d r_pf(r * std::cos(nu), r * std::sin(nu), 0.0);
    const Eigen::Vector3d v_pf(-std::sin(nu), e + std::cos(nu), 0.0);
    const Eigen::Matrix3d R = perifocal_to_eci(i, raan, argp);
    return simcore::State{t, R * r_pf, std::sqrt(mu / p) * (R * v_pf)};
}

// Difference of two angles wrapped to (-pi, pi], so 2 pi - 1e-15 and 0 compare as equal.
inline double angle_diff(double x, double y) {
    return std::remainder(x - y, 2.0 * kPi);
}

}  // namespace elements_fixtures

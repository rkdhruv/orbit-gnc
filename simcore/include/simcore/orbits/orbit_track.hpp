#pragma once

#include <vector>

#include <Eigen/Core>

#include "simcore/constants.hpp"
#include "simcore/dynamics/state.hpp"

namespace simcore {

// Points along the osculating two-body conic through s, in ECI, for drawing an orbit line.
// Built from r and v directly: h = r x v, p = |h|^2 / mu, e = v x h / mu - r_hat, perifocal axes
// P = e_hat (r_hat if e < 1e-12, a circle) and Q = h_hat x P, and
//   r(nu) = p / (1 + e cos nu) (cos nu P + sin nu Q).
// Closed orbits (e < 1): n_points at nu_k = nu0 + 2 pi k / n starting at the current position
// (equal true-anomaly steps put more points near periapsis, where the curve bends most); the ring
// closes from the last point to the first. r_max_m is ignored.
// Open orbits (e >= 1): n_points from nu = -nu_max to +nu_max, where r(nu_max) = r_max_m:
// cos nu_max = (p / r_max - 1) / e. Empty if periapsis p / (1 + e) is not inside r_max_m.
// Throws std::invalid_argument for n_points < 3, r_max_m not finite and > 0, or r = 0 or h = 0
// (no orbit plane).
struct OrbitTrack {
    std::vector<Eigen::Vector3d> points;
    bool closed;  // e < 1: draw a segment from the last point back to the first
};
OrbitTrack sample_orbit_eci(const State& s, int n_points, double r_max_m, double mu = earth::mu);

}  // namespace simcore

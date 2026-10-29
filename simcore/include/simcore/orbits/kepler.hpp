#pragma once

#include "simcore/constants.hpp"
#include "simcore/dynamics/state.hpp"
#include "simcore/orbits/epoch.hpp"

namespace simcore {

// Eccentric anomaly E solving Kepler's equation M = E - e sin E, for 0 <= e < 1 and any M.
// E is in the same revolution as M (|E - M| <= e). Newton's method from Danby's starting guess;
// f'(E) = 1 - e cos E >= 1 - e > 0, so the root is unique.
double solve_kepler_elliptic(double M_rad, double e);

// Hyperbolic anomaly H solving M = e sinh H - H, for e > 1 and any M (H has the sign of M).
double solve_kepler_hyperbolic(double M_rad, double e);

// Mean anomaly from true anomaly. Elliptic: result in [-pi, pi]; hyperbolic: requires
// 1 + e cos nu > 0 (inside the asymptotes).
double mean_from_true_anomaly(double nu_rad, double e);

// True anomaly from mean anomaly. Elliptic: result in [-pi, pi].
double true_from_mean_anomaly(double M_rad, double e);

// Analytic two-body propagation of s0 to epoch t (forward or backward), elliptic or hyperbolic:
// elements -> M = M0 + n (t - t0) -> solve Kepler -> nu -> state. n = sqrt(mu / |a|^3).
// Throws std::invalid_argument for the cases elements_from_state rejects (e.g. near-parabolic).
State kepler_propagate(const State& s0, const Epoch& t, double mu = earth::mu);

}  // namespace simcore

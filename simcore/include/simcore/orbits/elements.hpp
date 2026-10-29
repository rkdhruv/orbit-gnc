#pragma once

#include "simcore/constants.hpp"
#include "simcore/dynamics/state.hpp"
#include "simcore/orbits/epoch.hpp"

namespace simcore {

// Classical (Keplerian) orbital elements of an elliptic or hyperbolic two-body orbit.
// Ranges: i in [0, pi]; raan, argp, nu in [0, 2 pi) as returned by elements_from_state.
//
// Singular cases follow Vallado's conventions:
// - equatorial (sin i < 1e-11, node undefined): raan = 0 and argp is the longitude of periapsis,
//   measured from the x axis about the angular-momentum direction;
// - circular (e < 1e-11, periapsis undefined): argp = 0 and nu is the argument of latitude,
//   measured from the node (or from the x axis if also equatorial: the true longitude).
struct KeplerianElements {
    double a_m;       // semi-major axis [m]; negative for hyperbolas
    double e;         // eccentricity [-]
    double i_rad;     // inclination
    double raan_rad;  // right ascension of the ascending node
    double argp_rad;  // argument of periapsis
    double nu_rad;    // true anomaly
};

// Elements of the orbit through s about a body with gravitational parameter mu. Throws
// std::invalid_argument for non-finite input, r = 0, mu <= 0, a rectilinear orbit (h ~ 0) or a
// near-parabolic one (|1 - e| < 1e-9, where a is undefined).
KeplerianElements elements_from_state(const State& s, double mu = earth::mu);

// State at epoch t on the orbit given by el. Throws std::invalid_argument for non-finite input,
// e < 0, |1 - e| < 1e-9, a whose sign doesn't match e (ellipse: a > 0, hyperbola: a < 0), or a
// hyperbolic nu at or beyond the asymptote (1 + e cos nu <= 0).
State state_from_elements(const KeplerianElements& el, const Epoch& t, double mu = earth::mu);

}  // namespace simcore

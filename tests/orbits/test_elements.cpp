// M1.4: classical orbital elements <-> Cartesian state.
//
// From (r, v) and mu:
//   h = r x v,  n = z x h (node line),  e_vec = (v x h) / mu - r_hat,  e = |e_vec|
//   energy = v^2/2 - mu/r,  a = -mu / (2 energy)   (a < 0 for hyperbolas)
//   i = atan2(|h_xy|, h_z),  raan = atan2(n_y, n_x)
//   argp = angle from n_hat to e_hat about h_hat,  nu = angle from e_hat to r_hat about h_hat
// Angles as atan2(h_hat . (x cross y), x . y), which is well conditioned everywhere and handles
// retrograde orbits without special cases.
//
// Singular cases (Vallado's conventions): equatorial (i ~ 0 or pi: node undefined) -> raan = 0 and
// argp is the longitude of periapsis, measured from the x axis about h_hat; circular (e ~ 0:
// periapsis undefined) -> argp = 0 and nu is the argument of latitude (angle from the node, or
// from the x axis if also equatorial). Ranges: i in [0, pi]; raan, argp, nu in [0, 2 pi).
//
// Tolerances: each conversion is ~50 flops on O(1)-conditioned quantities (atan2 forms, no acos
// of near-1 arguments except i, whose test values are away from 0 and pi), so relative errors are
// ~50 eps = 1e-14. a = -mu/(2 energy) suffers cancellation by |v^2/2| / |energy|, at most ~10 for
// these orbits. Bounds: 1e-12 relative on a, 1e-12 on e and on angles (rad), leaving ~10-100x.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "elements_fixtures.hpp"
#include "simcore/constants.hpp"
#include "simcore/orbits/elements.hpp"

namespace {

using namespace elements_fixtures;
using simcore::KeplerianElements;
using simcore::State;
namespace earth = simcore::earth;

constexpr double kAngleTol = 1e-12;  // rad
constexpr double kRelTol = 1e-12;

void expect_elements(const KeplerianElements& el, double a, double e, double i, double raan,
                     double argp, double nu) {
    EXPECT_NEAR(el.a_m / a, 1.0, kRelTol);
    EXPECT_NEAR(el.e, e, kRelTol);
    EXPECT_NEAR(el.i_rad, i, kAngleTol);
    EXPECT_NEAR(angle_diff(el.raan_rad, raan), 0.0, kAngleTol);
    EXPECT_NEAR(angle_diff(el.argp_rad, argp), 0.0, kAngleTol);
    EXPECT_NEAR(angle_diff(el.nu_rad, nu), 0.0, kAngleTol);
    // Documented ranges.
    EXPECT_GE(el.i_rad, 0.0);
    EXPECT_LE(el.i_rad, kPi);
    for (const double ang : {el.raan_rad, el.argp_rad, el.nu_rad}) {
        EXPECT_GE(ang, 0.0);
        EXPECT_LT(ang, 2.0 * kPi);
    }
}

// The M1.2 e = 0.7 orbit (periapsis 420 km): a = 6,798,137 / 0.3 = 22,660,456.67 m.
constexpr double kA07 = (earth::radius + 420.0e3) / 0.3;

}  // namespace

// ---------------------------------------------------------------------------------------------
// Known orbits
// ---------------------------------------------------------------------------------------------

TEST(Elements, GeneralEllipticOrbit) {
    const double i = 51.6 * kDeg, raan = 30.0 * kDeg, argp = 75.0 * kDeg, nu = 2.0;
    const KeplerianElements el =
        simcore::elements_from_state(state_from(kA07, 0.7, i, raan, argp, nu));
    expect_elements(el, kA07, 0.7, i, raan, argp, nu);
}

// Retrograde hyperbola: e = 1.5, periapsis 500 km altitude -> a = r_p / (1 - e) = -13,756,274 m.
TEST(Elements, RetrogradeHyperbola) {
    const double a = (earth::radius + 500.0e3) / (1.0 - 1.5);
    EXPECT_NEAR(a, -13'756'274.0, 1e-6);
    const double i = 120.0 * kDeg, raan = 200.0 * kDeg, argp = 300.0 * kDeg, nu = 1.5;
    const KeplerianElements el =
        simcore::elements_from_state(state_from(a, 1.5, i, raan, argp, nu));
    EXPECT_LT(el.a_m, 0.0);
    expect_elements(el, a, 1.5, i, raan, argp, nu);
}

// Inbound leg of the hyperbola: nu = -1.5 is reported as 2 pi - 1.5.
TEST(Elements, HyperbolaInboundTrueAnomalyIsWrappedToZeroTwoPi) {
    const double a = (earth::radius + 500.0e3) / (1.0 - 1.5);
    const KeplerianElements el =
        simcore::elements_from_state(state_from(a, 1.5, 0.3, 1.0, 2.0, -1.5));
    EXPECT_NEAR(el.nu_rad, 2.0 * kPi - 1.5, kAngleTol);
}

// Circular, inclined: e_vec is pure roundoff (~1e-16), so argp = 0 and nu = argument of latitude.
// Built with argp = 0, nu = 1.2 rad -> argument of latitude 1.2 rad.
TEST(Elements, CircularOrbitReportsArgumentOfLatitude) {
    const double r = earth::radius + 420.0e3;
    const double i = 51.6 * kDeg, raan = 30.0 * kDeg;
    const KeplerianElements el =
        simcore::elements_from_state(state_from(r, 0.0, i, raan, 0.0, 1.2));
    EXPECT_LT(el.e, 1e-12);
    EXPECT_NEAR(el.a_m / r, 1.0, kRelTol);
    EXPECT_NEAR(el.i_rad, i, kAngleTol);
    EXPECT_NEAR(angle_diff(el.raan_rad, raan), 0.0, kAngleTol);
    EXPECT_EQ(el.argp_rad, 0.0);
    EXPECT_NEAR(angle_diff(el.nu_rad, 1.2), 0.0, kAngleTol);
}

// Equatorial prograde ellipse: node undefined -> raan = 0, argp = longitude of periapsis
// = raan_built + argp_built = 30 + 75 = 105 deg.
TEST(Elements, EquatorialOrbitReportsLongitudeOfPeriapsis) {
    const KeplerianElements el =
        simcore::elements_from_state(state_from(kA07, 0.7, 0.0, 30.0 * kDeg, 75.0 * kDeg, 2.0));
    EXPECT_EQ(el.i_rad, 0.0);
    EXPECT_EQ(el.raan_rad, 0.0);
    EXPECT_NEAR(angle_diff(el.argp_rad, 105.0 * kDeg), 0.0, kAngleTol);
    EXPECT_NEAR(angle_diff(el.nu_rad, 2.0), 0.0, kAngleTol);
}

// Retrograde equatorial (i = pi): the built orbit has its periapsis at ECI direction
// Rz(0) Rx(pi) [cos 75, sin 75, 0] = [cos 75, -sin 75, 0]. Measured about h_hat = -z from the x
// axis, that is +75 deg, so argp = 75 deg with raan = 0.
TEST(Elements, RetrogradeEquatorialOrbit) {
    const KeplerianElements el =
        simcore::elements_from_state(state_from(kA07, 0.7, kPi, 0.0, 75.0 * kDeg, 2.0));
    EXPECT_NEAR(el.i_rad, kPi, kAngleTol);
    EXPECT_EQ(el.raan_rad, 0.0);
    EXPECT_NEAR(angle_diff(el.argp_rad, 75.0 * kDeg), 0.0, kAngleTol);
    EXPECT_NEAR(angle_diff(el.nu_rad, 2.0), 0.0, kAngleTol);
}

// Circular equatorial: raan = argp = 0 and nu = true longitude (from the x axis) = 30 + 1.2 rad.
TEST(Elements, CircularEquatorialOrbitReportsTrueLongitude) {
    const double r = earth::radius + 420.0e3;
    const KeplerianElements el =
        simcore::elements_from_state(state_from(r, 0.0, 0.0, 30.0 * kDeg, 0.0, 1.2));
    EXPECT_EQ(el.raan_rad, 0.0);
    EXPECT_EQ(el.argp_rad, 0.0);
    EXPECT_NEAR(angle_diff(el.nu_rad, 30.0 * kDeg + 1.2), 0.0, kAngleTol);
}

// ---------------------------------------------------------------------------------------------
// Round trip state -> elements -> state over a grid, including near-singular cases
// ---------------------------------------------------------------------------------------------
// Compared as states, not elements: near e = 0 or i = 0 individual angles are ill-conditioned,
// but their errors cancel in the state (argp and nu both come from the same e_hat, so their
// errors are equal and opposite, and r depends on argp + nu). Bound 1e-12 relative on |r| and |v|.
TEST(Elements, StateRoundTripOverGrid) {
    struct Shape {
        double e;
        std::vector<double> nus;
    };
    // Hyperbolic true anomaly must satisfy 1 + e cos nu > 0, i.e. |nu| < acos(-1/e):
    // 2.30 rad for e = 1.5, 1.91 rad for e = 3.
    const std::vector<Shape> shapes{
        {0.0, {0.0, 1.0, 3.0, 5.5}},  {1e-6, {0.0, 1.0, 3.0, 5.5}}, {0.1, {0.0, 1.0, 3.0, 5.5}},
        {0.7, {0.0, 1.0, kPi, 5.5}},  {0.99, {0.0, 0.5, kPi, 6.0}}, {1.5, {0.0, 1.0, 2.2, -2.2}},
        {3.0, {0.0, 1.0, 1.8, -1.8}},
    };
    const std::vector<double> incs{0.0, 1e-6, 51.6 * kDeg, 90.0 * kDeg, 150.0 * kDeg, kPi};
    const double rp = earth::radius + 500.0e3;
    for (const Shape& sh : shapes) {
        const double a = rp / (1.0 - sh.e);
        for (const double i : incs) {
            for (const double nu : sh.nus) {
                SCOPED_TRACE(testing::Message() << "e=" << sh.e << " i=" << i << " nu=" << nu);
                const State s0 = state_from(a, sh.e, i, 200.0 * kDeg, 300.0 * kDeg, nu);
                const KeplerianElements el = simcore::elements_from_state(s0);
                const State s1 = simcore::state_from_elements(el, s0.t);
                EXPECT_EQ(s1.t, s0.t);
                EXPECT_LE((s1.r_eci - s0.r_eci).norm(), kRelTol * s0.r_eci.norm());
                EXPECT_LE((s1.v_eci - s0.v_eci).norm(), kRelTol * s0.v_eci.norm());
            }
        }
    }
}

// state_from_elements matches the independent perifocal construction directly.
TEST(Elements, StateFromElementsMatchesPerifocalConstruction) {
    const KeplerianElements el{kA07, 0.7, 51.6 * kDeg, 30.0 * kDeg, 75.0 * kDeg, 2.0};
    const State ref = state_from(kA07, 0.7, el.i_rad, el.raan_rad, el.argp_rad, el.nu_rad);
    const State s = simcore::state_from_elements(el, ref.t);
    EXPECT_LE((s.r_eci - ref.r_eci).norm(), kRelTol * ref.r_eci.norm());
    EXPECT_LE((s.v_eci - ref.v_eci).norm(), kRelTol * ref.v_eci.norm());
}

// ---------------------------------------------------------------------------------------------
// Invalid input
// ---------------------------------------------------------------------------------------------

TEST(Elements, RejectsInvalidInput) {
    const State good = state_from(kA07, 0.7, 0.9, 0.5, 1.0, 2.0);
    const double kNaN = std::numeric_limits<double>::quiet_NaN();
    // Parabolic: speed exactly escape speed sqrt(2 mu / r) -> energy 0, a undefined.
    const double r = earth::radius + 500.0e3;
    const State parabolic{good.t, Eigen::Vector3d(r, 0.0, 0.0),
                          Eigen::Vector3d(0.0, std::sqrt(2.0 * earth::mu / r), 0.0)};
    EXPECT_THROW((void)simcore::elements_from_state(parabolic), std::invalid_argument);
    // Rectilinear (h = 0): the orbit plane is undefined.
    const State radial{good.t, Eigen::Vector3d(r, 0.0, 0.0), Eigen::Vector3d(100.0, 0.0, 0.0)};
    EXPECT_THROW((void)simcore::elements_from_state(radial), std::invalid_argument);
    EXPECT_THROW(
        (void)simcore::elements_from_state(State{good.t, Eigen::Vector3d::Zero(), good.v_eci}),
        std::invalid_argument);
    EXPECT_THROW((void)simcore::elements_from_state(
                     State{good.t, Eigen::Vector3d(kNaN, 0.0, 0.0), good.v_eci}),
                 std::invalid_argument);
    EXPECT_THROW((void)simcore::elements_from_state(good, 0.0), std::invalid_argument);
    EXPECT_THROW((void)simcore::elements_from_state(good, -earth::mu), std::invalid_argument);

    const simcore::Epoch t = good.t;
    // e < 0; a with the wrong sign for e (ellipse needs a > 0, hyperbola a < 0); parabolic e.
    EXPECT_THROW((void)simcore::state_from_elements({kA07, -0.1, 0.0, 0.0, 0.0, 0.0}, t),
                 std::invalid_argument);
    EXPECT_THROW((void)simcore::state_from_elements({-kA07, 0.7, 0.0, 0.0, 0.0, 0.0}, t),
                 std::invalid_argument);
    EXPECT_THROW((void)simcore::state_from_elements({kA07, 1.5, 0.0, 0.0, 0.0, 0.0}, t),
                 std::invalid_argument);
    EXPECT_THROW((void)simcore::state_from_elements({kA07, 1.0, 0.0, 0.0, 0.0, 0.0}, t),
                 std::invalid_argument);
    // Hyperbolic nu beyond the asymptote (1 + e cos nu <= 0): acos(-1/1.5) = 2.30 rad.
    EXPECT_THROW((void)simcore::state_from_elements({-1.4e7, 1.5, 0.0, 0.0, 0.0, 2.5}, t),
                 std::invalid_argument);
    EXPECT_THROW((void)simcore::state_from_elements({kA07, 0.7, kNaN, 0.0, 0.0, 0.0}, t),
                 std::invalid_argument);
}

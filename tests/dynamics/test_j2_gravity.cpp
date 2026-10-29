// M2.1: J2 (Earth oblateness) perturbation as a ForceModel, plus ForceSum to combine models.
//
// Potential energy per unit mass of the J2 term (z = Earth's spin axis, ECI z):
//   V_J2 = (mu J2 R^2 / (2 r^3)) (3 z^2 / r^2 - 1),  a_J2 = -grad V_J2
//   a_J2 = -(3/2) J2 mu R^2 / r^5 [x (1 - 5 z^2/r^2), y (1 - 5 z^2/r^2), z (3 - 5 z^2/r^2)]
// (Vallado eq. 8-30; Curtis eq. 12.30.) The model returns the perturbation only; total gravity is
// ForceSum{PointMassGravity, J2Gravity}. Assumes ECI z is the pole (no precession/nutation).
//
// Reference values at r = R + 420 km = 6,798,137 m:
//   equator: |a_J2| = 1.5 J2 mu R^2 / r^4 = 1.2329e-2 m/s^2, inward;
//   pole:    |a_J2| = 3 J2 mu R^2 / r^4   = 2.4659e-2 m/s^2, outward.
// Roundoff tolerance: ~30 flops at 0.5 ulp -> kRel = 1e-13.
//
// Secular nodal regression (first-order J2 theory, Vallado eq. 9-41):
//   dRAAN/dt = -1.5 n J2 (R/p)^2 cos i
//   420 km circular, i = 51.6 deg -> -1.00013e-6 rad/s = -4.951 deg/day
//   700 km circular sun-synchronous needs dRAAN/dt = 360 deg / 365.2422 d = +0.98565 deg/day,
//   i.e. cos i = -rate / (1.5 n J2 (R/a)^2) -> i = 98.188 deg.
// Measured by sampling RAAN at every ascending node (u = 0), where the short-period J2 terms in
// RAAN (~sin 2u) vanish, over one day. Tolerance 1%: the formula is first order in J2 and uses
// mean elements; our start is osculating, which differs in a by ~J2 (R/a)^2 ~ 1e-3 relative and
// moves the rate (~a^-3.5) by ~0.35%. O(J2^2) terms are ~1e-6.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <vector>

#include <Eigen/Geometry>

#include "../orbits/elements_fixtures.hpp"
#include "simcore/constants.hpp"
#include "simcore/dynamics/force_sum.hpp"
#include "simcore/dynamics/j2_gravity.hpp"
#include "simcore/dynamics/point_mass_gravity.hpp"
#include "simcore/events/events.hpp"
#include "simcore/integrators/dormand_prince.hpp"
#include "simcore/orbits/elements.hpp"

namespace {

using elements_fixtures::kDeg;
using elements_fixtures::state_from;
using simcore::Epoch;
using simcore::J2Gravity;
using simcore::State;
namespace earth = simcore::earth;

constexpr double kRel = 1e-13;
constexpr double kPi = std::numbers::pi;
const Epoch kT0 = Epoch::from_tt_seconds_since_j2000(800'000'000, 0.25);

State at(const Eigen::Vector3d& r) {
    return State{kT0, r, Eigen::Vector3d::Zero()};
}

double v_j2(const Eigen::Vector3d& r) {
    const double rn = r.norm();
    return earth::mu * earth::j2 * earth::radius * earth::radius / (2.0 * rn * rn * rn) *
           (3.0 * r.z() * r.z() / (rn * rn) - 1.0);
}

// Mean RAAN rate [rad/s] over `days`, sampled at ascending nodes (z rising through 0).
double measured_raan_rate(const State& s0, double days) {
    auto sum = std::make_shared<simcore::ForceSum>();
    sum->add(std::make_shared<simcore::PointMassGravity>());
    sum->add(std::make_shared<J2Gravity>());
    simcore::DpOptions o;
    o.rtol = 1e-12;
    o.atol_pos_m = 1e-6;
    o.atol_vel_mps = 1e-9;
    simcore::DormandPrince54 dp(o);
    const simcore::EventSpec node{"ascending node", [](const State& s) { return s.r_eci.z(); },
                                  simcore::EventDirection::kRising, false};
    const simcore::EventRun run =
        simcore::propagate_with_events(dp, *sum, s0, s0.t + days * 86400.0, {node});
    EXPECT_GE(run.hits.size(), 10u);
    const State& first = run.hits.front().state;
    const State& last = run.hits.back().state;
    const double d_raan = std::remainder(
        simcore::elements_from_state(last).raan_rad - simcore::elements_from_state(first).raan_rad,
        2.0 * kPi);
    return d_raan / (last.t - first.t);
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Acceleration
// ---------------------------------------------------------------------------------------------

TEST(J2Gravity, EquatorialPointIsExtraInwardPull) {
    const double r = earth::radius + 420.0e3;
    const Eigen::Vector3d a = J2Gravity{}.acceleration_eci(at({r, 0.0, 0.0}));
    const double expected =
        1.5 * earth::j2 * earth::mu * earth::radius * earth::radius / (r * r * r * r);
    EXPECT_NEAR(expected, 1.2329e-2, 1e-6);  // hand value
    EXPECT_NEAR(a.x() / -expected, 1.0, kRel);
    EXPECT_EQ(a.y(), 0.0);
    EXPECT_EQ(a.z(), 0.0);
    // Relative to point-mass gravity: 1.5 J2 (R/r)^2 = 1.4295e-3.
    const double pm = earth::mu / (r * r);
    EXPECT_NEAR(-a.x() / pm, 1.4295e-3, 1e-7);
}

TEST(J2Gravity, PolarPointIsOutward) {
    const double r = earth::radius + 420.0e3;
    const Eigen::Vector3d a = J2Gravity{}.acceleration_eci(at({0.0, 0.0, r}));
    const double expected =
        3.0 * earth::j2 * earth::mu * earth::radius * earth::radius / (r * r * r * r);
    EXPECT_NEAR(expected, 2.4659e-2, 1e-6);
    EXPECT_NEAR(a.z() / expected, 1.0, kRel);
    EXPECT_EQ(a.x(), 0.0);
    EXPECT_EQ(a.y(), 0.0);
}

// a = -grad V_J2, checked by central differences at an oblique point (independent of the closed
// form). Here |V_J2| ~ 2.5e4 J/kg and |a_J2| ~ 1e-2 m/s^2 at |r| = 7.1e6 m. With h = 50 m:
// roundoff ~eps |V_J2| / h = 1.1e-13 m/s^2 (1e-11 relative); truncation ~(h/r)^2 times a factor
// ~10 for the J2 term's higher derivatives (it varies on scale r/5) -> ~5e-10 relative.
// Bound 1e-8 relative (20x margin).
TEST(J2Gravity, IsNegativeGradientOfPotential) {
    const Eigen::Vector3d r(4100.0e3, -3300.0e3, 4200.0e3);
    const Eigen::Vector3d a = J2Gravity{}.acceleration_eci(at(r));
    constexpr double h = 50.0;
    Eigen::Vector3d grad;
    for (int k = 0; k < 3; ++k) {
        Eigen::Vector3d dp = r, dm = r;
        dp(k) += h;
        dm(k) -= h;
        grad(k) = (v_j2(dp) - v_j2(dm)) / (2.0 * h);
    }
    EXPECT_LE((a + grad).norm(), 1e-8 * a.norm());
}

// Axial symmetry: rotating r about z rotates a the same way; mirroring z -> -z flips a_z.
TEST(J2Gravity, IsAxiallySymmetricAndEquatoriallyMirrored) {
    const J2Gravity j2;
    const Eigen::Vector3d r(5.0e6, 2.0e6, 3.5e6);
    const Eigen::Matrix3d Rz =
        Eigen::AngleAxisd(1.234, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    const Eigen::Vector3d a = j2.acceleration_eci(at(r));
    EXPECT_LE((j2.acceleration_eci(at(Rz * r)) - Rz * a).norm(), kRel * 10.0 * a.norm());
    const Eigen::Vector3d rm(r.x(), r.y(), -r.z());
    const Eigen::Vector3d am = j2.acceleration_eci(at(rm));
    EXPECT_NEAR(am.x(), a.x(), kRel * a.norm());
    EXPECT_NEAR(am.y(), a.y(), kRel * a.norm());
    EXPECT_NEAR(am.z(), -a.z(), kRel * a.norm());
}

TEST(J2Gravity, RejectsInvalidParametersAndOrigin) {
    const double kNaN = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((void)J2Gravity(0.0), std::invalid_argument);
    EXPECT_THROW((void)J2Gravity(earth::mu, -1.0), std::invalid_argument);
    EXPECT_THROW((void)J2Gravity(earth::mu, earth::radius, kNaN), std::invalid_argument);
    EXPECT_THROW((void)J2Gravity{}.acceleration_eci(at(Eigen::Vector3d::Zero())),
                 std::invalid_argument);
}

// ---------------------------------------------------------------------------------------------
// ForceSum
// ---------------------------------------------------------------------------------------------

// The sum is evaluated in insertion order, so it equals the hand-written sum bit for bit.
TEST(ForceSum, SumsModelsInInsertionOrder) {
    const State s = at({5.0e6, 2.0e6, 3.5e6});
    const simcore::PointMassGravity pm;
    const J2Gravity j2;
    simcore::ForceSum sum;
    EXPECT_EQ(sum.size(), 0u);
    EXPECT_EQ(sum.acceleration_eci(s), Eigen::Vector3d::Zero());
    sum.add(std::make_shared<simcore::PointMassGravity>()).add(std::make_shared<J2Gravity>());
    EXPECT_EQ(sum.size(), 2u);
    const Eigen::Vector3d expected =
        Eigen::Vector3d::Zero() + pm.acceleration_eci(s) + j2.acceleration_eci(s);
    EXPECT_EQ(sum.acceleration_eci(s), expected);
}

TEST(ForceSum, RejectsNullModel) {
    simcore::ForceSum sum;
    EXPECT_THROW(sum.add(nullptr), std::invalid_argument);
}

// ---------------------------------------------------------------------------------------------
// Secular effects (the M2 done-when check)
// ---------------------------------------------------------------------------------------------

TEST(J2Gravity, NodalRegressionMatchesFirstOrderTheoryForIssOrbit) {
    const double a = earth::radius + 420.0e3;
    const double i = 51.6 * kDeg;
    const State s0 = state_from(a, 0.0, i, 30.0 * kDeg, 0.0, 0.0);  // at the ascending node
    const double n = std::sqrt(earth::mu / (a * a * a));
    const double expected =
        -1.5 * n * earth::j2 * (earth::radius / a) * (earth::radius / a) * std::cos(i);
    EXPECT_NEAR(expected * 86400.0 / kDeg, -4.951, 1e-3);  // deg/day, hand value
    EXPECT_NEAR(measured_raan_rate(s0, 1.0) / expected, 1.0, 0.01);
}

TEST(J2Gravity, SunSynchronousInclinationGivesOneTurnPerYear) {
    const double a = earth::radius + 700.0e3;
    const double n = std::sqrt(earth::mu / (a * a * a));
    const double sso_rate = 2.0 * kPi / (365.2422 * 86400.0);
    const double i =
        std::acos(-sso_rate / (1.5 * n * earth::j2 * (earth::radius / a) * (earth::radius / a)));
    EXPECT_NEAR(i / kDeg, 98.188, 1e-3);  // hand value
    const State s0 = state_from(a, 0.0, i, 30.0 * kDeg, 0.0, 0.0);
    EXPECT_NEAR(measured_raan_rate(s0, 1.0) / sso_rate, 1.0, 0.01);
}

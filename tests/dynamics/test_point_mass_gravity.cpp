// M1.1: point-mass (two-body) gravity, a = -mu * r / |r|^3, behind the ForceModel interface.
//
// Physics: Newtonian gravity of a spherically symmetric body acts like a point mass at its
// centre. The potential is U(r) = -mu/|r| and the acceleration is a = -grad U = -mu r / |r|^3:
// magnitude mu/|r|^2, pointing at the centre.
//
// Reference radius: 420 km circular orbit (ISS-like), r = R + h = 6,378,137 + 420,000 = 6,798,137
// m.
//
// Roundoff tolerance used throughout (kRoundoffRel = 1e-14 relative): the implementation needs
// ~10 flops (squared norm: 3 mul + 2 add, sqrt, cube, divide, scale) and the test another ~10 to
// form norms/ratios; each rounds by <= 0.5 ulp = 1.1e-16 relative, so the worst case is ~2e-15.
// 1e-14 leaves margin for e.g. std::pow, but any modelling error (wrong exponent, missing factor)
// is O(1) and fails by 13+ orders of magnitude.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>

#include <Eigen/Dense>

#include "simcore/constants.hpp"
#include "simcore/dynamics/force_model.hpp"
#include "simcore/dynamics/point_mass_gravity.hpp"
#include "simcore/dynamics/state.hpp"
#include "simcore/orbits/epoch.hpp"

namespace {

using simcore::Epoch;
using simcore::ForceModel;
using simcore::PointMassGravity;
using simcore::State;
namespace earth = simcore::earth;

constexpr double kRoundoffRel = 1e-14;
constexpr double kR420_m = earth::radius + 420.0e3;  // 6,798,137 m

const Eigen::Vector3d kZeroV = Eigen::Vector3d::Zero();

// A non-axis-aligned point at |r| = R + 420 km, so all three components are exercised.
Eigen::Vector3d r420_oblique() {
    return Eigen::Vector3d(1.0, -2.0, 3.0).normalized() * kR420_m;
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Interface
// ---------------------------------------------------------------------------------------------

static_assert(std::is_abstract_v<ForceModel>, "ForceModel is an interface");
static_assert(std::has_virtual_destructor_v<ForceModel>,
              "deleted through base pointers by the integrator");
static_assert(std::is_base_of_v<ForceModel, PointMassGravity>);

// Slicing prevention: ForceModel's copy/move assignment are protected, so a derived model can't be
// partially overwritten through a base reference (`ForceModel& a = g1; a = g2;` would copy only the
// base subobject and keep g1's mu). is_copy_constructible is meaningless here (an abstract class is
// never constructible), but is_*_assignable on an abstract class does test the access check.
static_assert(!std::is_copy_assignable_v<ForceModel>, "base copy-assignment must be protected");
static_assert(!std::is_move_assignable_v<ForceModel>, "base move-assignment must be protected");

// ...while concrete, final models stay ordinary value types (they reach the protected base
// members).
static_assert(std::is_copy_constructible_v<PointMassGravity>);
static_assert(std::is_copy_assignable_v<PointMassGravity>);
static_assert(std::is_nothrow_move_constructible_v<PointMassGravity>);
static_assert(std::is_nothrow_move_assignable_v<PointMassGravity>);

TEST(PointMassGravity, CopyPreservesMuAndAcceleration) {
    constexpr double kMuMoon = 4.9048695e12;
    const PointMassGravity original(kMuMoon);
    const PointMassGravity copy = original;  // NOLINT(performance-unnecessary-copy-initialization)
    PointMassGravity assigned;               // Earth mu
    assigned = original;

    const State s{Epoch::j2000(), r420_oblique(), kZeroV};
    EXPECT_EQ(copy.mu(), kMuMoon);
    EXPECT_EQ(assigned.mu(), kMuMoon);
    EXPECT_EQ(copy.acceleration_eci(s), original.acceleration_eci(s));
    EXPECT_EQ(assigned.acceleration_eci(s), original.acceleration_eci(s));
}

TEST(PointMassGravity, DefaultMuIsEarth) {
    EXPECT_EQ(PointMassGravity{}.mu(), earth::mu);
}

TEST(PointMassGravity, UsableThroughForceModelInterface) {
    // Integrators only see ForceModel; dispatch through the base must give the same answer.
    const PointMassGravity direct;
    const std::unique_ptr<ForceModel> model = std::make_unique<PointMassGravity>();
    const Eigen::Vector3d r = r420_oblique();
    EXPECT_EQ(model->acceleration_eci(State{Epoch::j2000(), r, kZeroV}),
              direct.acceleration_eci(State{Epoch::j2000(), r, kZeroV}));
}

// ---------------------------------------------------------------------------------------------
// Magnitude and direction at 420 km
// ---------------------------------------------------------------------------------------------

// |a| = mu / r^2 = 3.986004418e14 / 6,798,137^2 = 3.986004418e14 / 4.62146667e13 = 8.624977 m/s^2
// (vs. g0 = 9.807 at the surface: (R/r)^2 = 0.8802, and 9.798 * 0.8802 = 8.625 — consistent).
TEST(PointMassGravity, MagnitudeAt420kmIsMuOverRSquared) {
    const PointMassGravity grav;
    const Eigen::Vector3d a = grav.acceleration_eci(State{Epoch::j2000(), r420_oblique(), kZeroV});

    const double expected = earth::mu / (kR420_m * kR420_m);
    EXPECT_NEAR(a.norm(), expected, kRoundoffRel * expected);

    // Cross-check against the hand-derived number above; 1e-6 m/s^2 = rounding of the 7 printed
    // digits.
    EXPECT_NEAR(a.norm(), 8.624977, 1e-6);
}

TEST(PointMassGravity, PointsExactlyAtCentreOnAxes) {
    // r along one axis: the other two components are 0 * (finite scalar) = 0 exactly, so any
    // nonzero off-axis component is a bug, not roundoff.
    const PointMassGravity grav;
    const double expected = earth::mu / (kR420_m * kR420_m);

    for (int axis = 0; axis < 3; ++axis) {
        SCOPED_TRACE(axis);
        Eigen::Vector3d r = Eigen::Vector3d::Zero();
        r[axis] = kR420_m;
        const Eigen::Vector3d a = grav.acceleration_eci(State{Epoch::j2000(), r, kZeroV});
        EXPECT_NEAR(a[axis], -expected, kRoundoffRel * expected);  // towards the origin
        EXPECT_EQ(a[(axis + 1) % 3], 0.0);
        EXPECT_EQ(a[(axis + 2) % 3], 0.0);
    }
}

TEST(PointMassGravity, DirectionIsMinusRHat) {
    // a_hat must equal -r_hat. Unit-vector components are O(1), so a few ulps of 1.0 (~1e-15)
    // bounds the roundoff of the two normalisations; also a and r must be anti-parallel (dot < 0).
    const PointMassGravity grav;
    const Eigen::Vector3d r = r420_oblique();
    const Eigen::Vector3d a = grav.acceleration_eci(State{Epoch::j2000(), r, kZeroV});

    EXPECT_LT(a.dot(r), 0.0);
    EXPECT_LT((a.normalized() + r.normalized()).norm(), 1e-15);
}

// ---------------------------------------------------------------------------------------------
// Inverse-square law and independence from v, t
// ---------------------------------------------------------------------------------------------

// a(k r) = a(r) / k^2 for any k > 0 (same direction, magnitude scales as 1/k^2).
TEST(PointMassGravity, InverseSquareScaling) {
    const PointMassGravity grav;
    const Eigen::Vector3d r = r420_oblique();
    const Eigen::Vector3d a1 = grav.acceleration_eci(State{Epoch::j2000(), r, kZeroV});

    for (const double k : {2.0, 0.5, 3.0, 6.2}) {  // 6.2 * 6798 km ~ GEO radius
        SCOPED_TRACE(k);
        const Eigen::Vector3d ak = grav.acceleration_eci(State{Epoch::j2000(), k * r, kZeroV});
        EXPECT_LT((ak - a1 / (k * k)).norm(), kRoundoffRel * a1.norm() / (k * k));
    }
}

TEST(PointMassGravity, IndependentOfVelocityAndTime) {
    // Point-mass gravity depends on position only. Same r must give a bit-identical result
    // regardless of v or t (pure function of r; also a determinism requirement).
    const PointMassGravity grav;
    const Eigen::Vector3d r = r420_oblique();
    const Eigen::Vector3d a_ref = grav.acceleration_eci(State{Epoch::j2000(), r, kZeroV});

    const Eigen::Vector3d v_orbit(-5000.0, 1000.0, 5700.0);  // |v| ~ 7.65 km/s
    const Eigen::Vector3d v_huge(3.0e5, -2.0e5, 1.0e5);
    const Epoch t_far = Epoch::from_tt_seconds_since_j2000(946'728'000, 0.5);  // ~30 years later

    EXPECT_EQ(grav.acceleration_eci(State{Epoch::j2000(), r, v_orbit}), a_ref);
    EXPECT_EQ(grav.acceleration_eci(State{Epoch::j2000(), r, v_huge}), a_ref);
    EXPECT_EQ(grav.acceleration_eci(State{t_far, r, kZeroV}), a_ref);
    EXPECT_EQ(grav.acceleration_eci(State{t_far, r, v_orbit}), a_ref);
}

// ---------------------------------------------------------------------------------------------
// Conservative field: a = -grad U, U = -mu / |r|
// ---------------------------------------------------------------------------------------------

// Central finite difference: dU/dx_i ~ [U(r + h e_i) - U(r - h e_i)] / (2h).
//   Truncation error: (h^2/6) U''' ~ (h^2/6) * 6 mu / r^4  ->  relative to |a| = mu/r^2: (h/r)^2.
//   Roundoff error:   U ~ mu/r = 5.9e7 m^2/s^2 carries ~eps*|U| per evaluation, divided by 2h ->
//                     relative to |a|: eps * r / h.
//   Optimum h = r * eps^(1/3) ~ 6.7e6 * 6e-6 ~ 40 m.  With h = 50 m and |r| = 6.73e6 m:
//     truncation (50/6.73e6)^2 = 5.5e-11, roundoff 2.2e-16 * 6.73e6 / 50 = 3.0e-11 -> ~9e-11 total.
//   Tolerance 1e-9 relative (~10x the bound; measured 2.4e-11). A wrong exponent or sign is O(1).
// Point (4100, -3300, 4200) km: non-axis-aligned, |r| = 6,733.5 km (~355 km altitude), integer
// metres so r +/- h is exact.
TEST(PointMassGravity, AccelerationIsMinusGradientOfPotential) {
    const PointMassGravity grav;
    const Eigen::Vector3d r0(4'100'000.0, -3'300'000.0, 4'200'000.0);
    constexpr double h = 50.0;  // m
    constexpr double kFdRel = 1e-9;

    const auto potential = [](const Eigen::Vector3d& r) { return -earth::mu / r.norm(); };

    Eigen::Vector3d grad_u;
    for (int i = 0; i < 3; ++i) {
        const Eigen::Vector3d dr = h * Eigen::Vector3d::Unit(i);
        grad_u[i] = (potential(r0 + dr) - potential(r0 - dr)) / (2.0 * h);
    }

    const Eigen::Vector3d a = grav.acceleration_eci(State{Epoch::j2000(), r0, kZeroV});
    EXPECT_LT((a + grad_u).norm(), kFdRel * a.norm());
}

// ---------------------------------------------------------------------------------------------
// Circular orbit consistency (CLAUDE.md reference: 420 km circular orbit, speed ~7.66 km/s)
// ---------------------------------------------------------------------------------------------

// Circular orbit: centripetal acceleration v^2/r must be supplied entirely by gravity mu/r^2,
//   v = sqrt(mu / r) = sqrt(3.986004418e14 / 6,798,137) = sqrt(5.863378e7) = 7657.27 m/s.
// Reference says ~7.66 km/s: tolerance 5 m/s = half a unit in the reference's last digit (0.01
// km/s).
TEST(PointMassGravity, CircularSpeedAt420kmBalancesGravity) {
    const PointMassGravity grav;
    const double v_circ = std::sqrt(earth::mu / kR420_m);
    EXPECT_NEAR(v_circ, 7657.27, 0.01);  // hand derivation above, rounded to 0.01 m/s
    EXPECT_NEAR(v_circ, 7660.0, 5.0);    // CLAUDE.md reference value

    const Eigen::Vector3d r = r420_oblique();
    const double a_mag = grav.acceleration_eci(State{Epoch::j2000(), r, kZeroV}).norm();
    const double centripetal = v_circ * v_circ / kR420_m;
    EXPECT_NEAR(a_mag, centripetal, kRoundoffRel * centripetal);
}

// ---------------------------------------------------------------------------------------------
// Custom mu
// ---------------------------------------------------------------------------------------------

// Moon: GM = 4.9048695e12 m^3/s^2, 100 km circular orbit, r = 1,737.4 km + 100 km = 1,837,400 m.
//   |a| = 4.9048695e12 / 1,837,400^2 = 4.9048695e12 / 3.37603876e12 = 1.452848 m/s^2.
TEST(PointMassGravity, CustomMuIsHonoured) {
    constexpr double kMuMoon = 4.9048695e12;
    constexpr double kRMoon100_m = 1'837'400.0;
    const PointMassGravity moon(kMuMoon);
    EXPECT_EQ(moon.mu(), kMuMoon);

    const Eigen::Vector3d r = Eigen::Vector3d(-2.0, 1.0, 2.0).normalized() * kRMoon100_m;
    const Eigen::Vector3d a = moon.acceleration_eci(State{Epoch::j2000(), r, kZeroV});

    const double expected = kMuMoon / (kRMoon100_m * kRMoon100_m);
    EXPECT_NEAR(a.norm(), expected, kRoundoffRel * expected);
    EXPECT_NEAR(a.norm(), 1.452848, 1e-6);  // hand derivation above, rounding of printed digits
    EXPECT_LT((a.normalized() + r.normalized()).norm(), 1e-15);
}

// ---------------------------------------------------------------------------------------------
// Constructor validation
// ---------------------------------------------------------------------------------------------

// mu = G*M must be finite and strictly positive: mu = 0 is no gravity at all (almost certainly a
// default-initialised value), mu < 0 is repulsive, and NaN/inf poison every step downstream.
TEST(PointMassGravity, ConstructorRejectsNonPositiveOrNonFiniteMu) {
    constexpr double kInf = std::numeric_limits<double>::infinity();
    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    for (const double bad : {kNaN, kInf, -kInf, 0.0, -0.0, -earth::mu}) {
        SCOPED_TRACE(bad);
        EXPECT_THROW((void)PointMassGravity{bad}, std::invalid_argument);
    }
}

TEST(PointMassGravity, ConstructorAcceptsFinitePositiveMu) {
    // Any finite positive mu is physical, however small (e.g. a 1 kg-scale body, mu ~ 6.7e-11).
    for (const double ok : {earth::mu, 4.9048695e12, 1.0, 6.6743e-11}) {
        SCOPED_TRACE(ok);
        EXPECT_NO_THROW((void)PointMassGravity{ok});
    }
}

// ---------------------------------------------------------------------------------------------
// Singularity at the origin and non-finite results
// ---------------------------------------------------------------------------------------------

// At r = 0 the formula is 0 * (mu / 0) = 0 * inf = NaN, which would silently poison an
// integration. It must be rejected with std::invalid_argument (active in every build type).
// -0.0 components also give |r|^2 == 0.
TEST(PointMassGravity, ThrowsAtOrigin) {
    const PointMassGravity grav;
    const State at_origin{Epoch::j2000(), Eigen::Vector3d::Zero(), kZeroV};
    const State at_neg_zero{Epoch::j2000(), Eigen::Vector3d(-0.0, 0.0, -0.0), kZeroV};
    EXPECT_THROW((void)grav.acceleration_eci(at_origin), std::invalid_argument);
    EXPECT_THROW((void)grav.acceleration_eci(at_neg_zero), std::invalid_argument);

    // Also through the interface the integrator uses.
    const ForceModel& model = grav;
    EXPECT_THROW((void)model.acceleration_eci(at_origin), std::invalid_argument);
}

// The rule is "reject when |r|^2 or k = mu/|r|^3 is not finite", not "reject only r = 0": a small
// but representable r, however unphysical (inside the Earth), is still a valid input and must obey
// |a| = mu / r^2 with the same roundoff tolerance.
//   r = 1 m:    |a| = 3.986004418e14 m/s^2
//   r = 1 mm:   |a| = 3.986004418e20 m/s^2
// Both well inside double range (|r|^3 = 1e-9 m^3 does not underflow), so kRoundoffRel holds.
TEST(PointMassGravity, TinyNonzeroRadiusDoesNotThrowAndObeysInverseSquare) {
    const PointMassGravity grav;
    for (const double r_mag : {1.0, 1e-3}) {
        SCOPED_TRACE(r_mag);
        for (const Eigen::Vector3d& dir :
             {Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Vector3d(1.0, -2.0, 3.0).normalized()}) {
            const Eigen::Vector3d r = r_mag * dir;
            Eigen::Vector3d a = Eigen::Vector3d::Zero();
            EXPECT_NO_THROW(a = grav.acceleration_eci(State{Epoch::j2000(), r, kZeroV}));
            const double expected = earth::mu / (r_mag * r_mag);
            EXPECT_NEAR(a.norm(), expected, kRoundoffRel * expected);
            EXPECT_LT((a.normalized() + r.normalized()).norm(), 1e-15);
        }
    }
}

// Rejection rule: |r|^2 must be finite and k = mu / |r|^3 must be finite. Both conditions are
// needed because "r != 0" is not enough in floating point:
//   r = (1e-100, 0, 0):  |r|^2 = 1e-200 and |r|^3 = 1e-300 are normal, nonzero doubles, but
//                         k = 3.986e14 / 1e-300 = 3.986e314 > DBL_MAX (1.8e308) -> +inf, and
//                         a = -k * r = (-inf, -inf * 0, -inf * 0) = (-inf, NaN, NaN).
// When k and r are finite, every component |k * r_i| <= k |r| = mu / |r|^2 is finite too
// (|r| < 1 gives k|r| < k; |r| >= 1 gives k|r| <= mu), so the two checks guarantee a finite a.
TEST(PointMassGravity, ThrowsWhenMuOverRCubedOverflows) {
    const PointMassGravity grav;
    const State tiny{Epoch::j2000(), Eigen::Vector3d(1e-100, 0.0, 0.0), kZeroV};
    EXPECT_THROW((void)grav.acceleration_eci(tiny), std::invalid_argument);
}

// Non-finite positions (an earlier step that blew up) must be rejected rather than propagated:
// an inf component makes |r|^2 = inf, a NaN component makes |r|^2 = NaN.
TEST(PointMassGravity, ThrowsForNonFinitePosition) {
    constexpr double kInf = std::numeric_limits<double>::infinity();
    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    const PointMassGravity grav;
    for (const Eigen::Vector3d& r :
         {Eigen::Vector3d(kInf, 0.0, 0.0), Eigen::Vector3d(kR420_m, -kInf, 0.0),
          Eigen::Vector3d(kNaN, 0.0, 0.0), Eigen::Vector3d(kR420_m, 0.0, kNaN)}) {
        SCOPED_TRACE(r.transpose());
        EXPECT_THROW((void)grav.acceleration_eci(State{Epoch::j2000(), r, kZeroV}),
                     std::invalid_argument);
    }
}

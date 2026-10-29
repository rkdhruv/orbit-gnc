// M1.4: Kepler's equation and analytic two-body propagation (elliptic and hyperbolic).
//
// Elliptic:   M = E - e sin E,          tan(nu/2) = sqrt((1+e)/(1-e)) tan(E/2),  n = sqrt(mu/a^3)
// Hyperbolic: M = e sinh H - H,         tanh(H/2) = sqrt((e-1)/(e+1)) tan(nu/2), n =
// sqrt(mu/(-a)^3) M grows linearly in time, M = M0 + n dt, so propagation is: state -> elements ->
// M0 -> M -> solve Kepler for E or H -> nu -> state. Kepler's equation is solved by Newton's
// method: f(E) = E - e sin E - M has f' = 1 - e cos E >= 1 - e > 0, so the root is unique.
//
// Reference values computed independently (Python, 100 Newton iterations) and quoted to 17
// digits; tolerances are converged-Newton accuracy, a few ulp of the result.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <vector>

#include "../integrators/orbit_fixtures.hpp"
#include "elements_fixtures.hpp"
#include "simcore/dynamics/point_mass_gravity.hpp"
#include "simcore/integrators/dormand_prince.hpp"
#include "simcore/orbits/kepler.hpp"

namespace {

using elements_fixtures::kDeg;
using elements_fixtures::state_from;
using simcore::State;
namespace earth = simcore::earth;
constexpr double kPi = std::numbers::pi;
constexpr double kEps = std::numeric_limits<double>::epsilon();

// Hyperbola used throughout: e = 1.5, periapsis altitude 500 km, a = -13,756,274 m,
// n = sqrt(mu / (-a)^3) = 3.913070401e-4 rad/s, asymptote nu_inf = acos(-1/e) = 2.3005 rad.
constexpr double kHypE = 1.5;
constexpr double kHypA = (earth::radius + 500.0e3) / (1.0 - kHypE);

double rel_err(const Eigen::Vector3d& x, const Eigen::Vector3d& ref) {
    return (x - ref).norm() / ref.norm();
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Kepler's equation
// ---------------------------------------------------------------------------------------------

TEST(KeplerEquation, EllipticKnownValues) {
    EXPECT_NEAR(simcore::solve_kepler_elliptic(1.0, 0.5), 1.4987011335178484, 4e-15);
    EXPECT_NEAR(simcore::solve_kepler_elliptic(0.1, 0.9), 0.6308435275631538, 4e-15);
    EXPECT_NEAR(simcore::solve_kepler_elliptic(3.0, 0.99), 3.0704106691175017, 4e-15);
    EXPECT_EQ(simcore::solve_kepler_elliptic(0.0, 0.7), 0.0);      // periapsis
    EXPECT_EQ(simcore::solve_kepler_elliptic(1.234, 0.0), 1.234);  // circle: E = M
}

// Over a grid of e (up to 0.999999) and M (including many revolutions, as in a 100-orbit run):
// the residual E - e sin E - M is at the level of M's own rounding, |r| <= 4 eps (|M| + 1), and E
// stays in the same revolution as M: |E - M| = e |sin E| <= e.
TEST(KeplerEquation, EllipticResidualOverGrid) {
    for (const double e : {0.0, 0.1, 0.5, 0.9, 0.99, 0.999999}) {
        for (int k = -50; k <= 50; ++k) {
            const double M = 0.0631 * k + (k % 7) * 2.0 * kPi * 15.0;  // up to 90 revolutions
            SCOPED_TRACE(testing::Message() << "e=" << e << " M=" << M);
            const double E = simcore::solve_kepler_elliptic(M, e);
            EXPECT_LE(std::abs(E - e * std::sin(E) - M), 4.0 * kEps * (std::abs(M) + 1.0));
            EXPECT_LE(std::abs(E - M), e + 4.0 * kEps * (std::abs(M) + 1.0));
        }
    }
}

// e = 1.5, nu = 1.5 rad: H = 2 atanh(sqrt(0.5/2.5) tan 0.75) = 0.8871965544549957,
// M = e sinh H - H = 0.6251813249152555.
TEST(KeplerEquation, HyperbolicKnownValues) {
    EXPECT_NEAR(simcore::solve_kepler_hyperbolic(0.6251813249152555, 1.5), 0.8871965544549957,
                4e-15);
    EXPECT_EQ(simcore::solve_kepler_hyperbolic(0.0, 2.0), 0.0);
}

// Residual of e sinh H - H - M over e in [1.0001, 10] and |M| up to 1e6 (far out on the
// asymptote). Newton converges to H within a few ulp, so the residual is bounded by the slope
// times that: |r| <= 8 eps (e cosh H) max(1, |H|), plus M's own rounding 4 eps |M|.
TEST(KeplerEquation, HyperbolicResidualOverGrid) {
    for (const double e : {1.0001, 1.1, 1.5, 3.0, 10.0}) {
        for (const double M : {0.0, 1e-3, 0.1, 1.0, 10.0, 1e3, 1e6}) {
            for (const double sgn : {1.0, -1.0}) {
                SCOPED_TRACE(testing::Message() << "e=" << e << " M=" << sgn * M);
                const double H = simcore::solve_kepler_hyperbolic(sgn * M, e);
                const double resid = e * std::sinh(H) - H - sgn * M;
                EXPECT_LE(
                    std::abs(resid),
                    8.0 * kEps * e * std::cosh(H) * std::max(1.0, std::abs(H)) + 4.0 * kEps * M);
                EXPECT_GE(sgn * H, 0.0);  // H has the sign of M
            }
        }
    }
}

TEST(KeplerEquation, RejectsInvalidInput) {
    const double kNaN = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((void)simcore::solve_kepler_elliptic(1.0, -0.1), std::invalid_argument);
    EXPECT_THROW((void)simcore::solve_kepler_elliptic(1.0, 1.0), std::invalid_argument);
    EXPECT_THROW((void)simcore::solve_kepler_elliptic(kNaN, 0.5), std::invalid_argument);
    EXPECT_THROW((void)simcore::solve_kepler_hyperbolic(1.0, 1.0), std::invalid_argument);
    EXPECT_THROW((void)simcore::solve_kepler_hyperbolic(1.0, 0.5), std::invalid_argument);
    EXPECT_THROW((void)simcore::solve_kepler_hyperbolic(kNaN, 1.5), std::invalid_argument);
}

// ---------------------------------------------------------------------------------------------
// Anomaly conversions
// ---------------------------------------------------------------------------------------------

// e = 0.7, nu = 2 rad: E = 1.158703181269719, M = 0.517304054353014 (the M1.3 event test's
// "2794.988 s since periapsis" = M / n).
TEST(Anomalies, KnownValues) {
    EXPECT_NEAR(simcore::mean_from_true_anomaly(2.0, 0.7), 0.517304054353014, 4e-15);
    EXPECT_NEAR(simcore::mean_from_true_anomaly(1.5, kHypE), 0.6251813249152555, 4e-15);
    EXPECT_NEAR(simcore::mean_from_true_anomaly(-1.5, kHypE), -0.6251813249152555, 4e-15);
    EXPECT_EQ(simcore::mean_from_true_anomaly(0.0, 0.7), 0.0);
}

// true -> mean -> true over (-pi, pi] (elliptic) and inside the asymptotes (hyperbolic). The
// inverse is most sensitive at periapsis: dnu/dM = (1 + e cos nu)^2 / (1 - e^2)^1.5, = 1414 at
// e = 0.99, so a 1e-16 error in M becomes ~1.4e-13 in nu. Bound 1e-12 rad.
TEST(Anomalies, RoundTripOverGrid) {
    for (const double e : {0.0, 0.1, 0.7, 0.99}) {
        for (int k = -15; k <= 16; ++k) {
            const double nu = kPi * k / 16.0;  // (-pi, pi]
            SCOPED_TRACE(testing::Message() << "e=" << e << " nu=" << nu);
            const double M = simcore::mean_from_true_anomaly(nu, e);
            EXPECT_GT(M, -kPi - 1e-15);
            EXPECT_LE(M, kPi + 1e-15);
            EXPECT_NEAR(std::remainder(simcore::true_from_mean_anomaly(M, e) - nu, 2.0 * kPi), 0.0,
                        1e-12);
        }
    }
    for (const double e : {1.1, 1.5, 3.0}) {
        const double nu_inf = std::acos(-1.0 / e);
        for (int k = -9; k <= 9; ++k) {
            const double nu = 0.99 * nu_inf * k / 9.0;
            SCOPED_TRACE(testing::Message() << "e=" << e << " nu=" << nu);
            const double M = simcore::mean_from_true_anomaly(nu, e);
            EXPECT_NEAR(simcore::true_from_mean_anomaly(M, e), nu, 1e-12);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Analytic propagation
// ---------------------------------------------------------------------------------------------

TEST(KeplerPropagate, ZeroDurationReturnsStateExactly) {
    const State s0 = orbit_fixtures::eccentric_apoapsis_state();
    const State s1 = simcore::kepler_propagate(s0, s0.t);
    EXPECT_EQ(s1.t, s0.t);
    EXPECT_EQ(s1.r_eci, s0.r_eci);
    EXPECT_EQ(s1.v_eci, s0.v_eci);
}

// Apoapsis + T/2 = periapsis; apoapsis + T = apoapsis. Errors: elements round trip (1e-14) plus
// rounding of M0 + n dt (~4 eps * pi) -> bound 1e-12 relative.
TEST(KeplerPropagate, HalfAndFullPeriodOfEccentricOrbit) {
    const State apo = orbit_fixtures::eccentric_apoapsis_state();
    const State peri = orbit_fixtures::eccentric_periapsis_state();
    const double T = orbit_fixtures::eccentric_period_s();
    const State half = simcore::kepler_propagate(apo, apo.t + 0.5 * T);
    EXPECT_EQ(half.t, apo.t + 0.5 * T);
    EXPECT_LE(rel_err(half.r_eci, peri.r_eci), 1e-12);
    EXPECT_LE(rel_err(half.v_eci, peri.v_eci), 1e-12);
    const State full = simcore::kepler_propagate(apo, apo.t + T);
    EXPECT_LE(rel_err(full.r_eci, apo.r_eci), 1e-12);
    EXPECT_LE(rel_err(full.v_eci, apo.v_eci), 1e-12);
}

// The circular 420 km orbit against the exact rotation r0 [cos nt, sin nt] at 100 T + 1234 s.
// Both sides carry the phase n t = 631 rad; its rounding is ~eps n t, so the bound is
// 1e-12 (1 + n t) relative = 6.3e-10 (4 mm).
TEST(KeplerPropagate, MatchesCircularRotationOverHundredOrbits) {
    const State s0 = orbit_fixtures::circular_initial_state();
    const double dt = 100.0 * orbit_fixtures::circular_period_s() + 1234.0;
    const State ref = orbit_fixtures::circular_analytic_state(dt);
    const State s = simcore::kepler_propagate(s0, s0.t + dt);
    const double bound = 1e-12 * (1.0 + orbit_fixtures::circular_mean_motion_radps() * dt);
    EXPECT_LE(rel_err(s.r_eci, ref.r_eci), bound);
    EXPECT_LE(rel_err(s.v_eci, ref.v_eci), bound);
}

// Hyperbola from periapsis to nu = 1.5 rad: t = M / n = 0.62518 / 3.91307e-4 = 1597.6746 s.
// Backward by the same time reaches nu = -1.5. Bound 1e-12 relative (as above).
TEST(KeplerPropagate, HyperbolicFlybyMatchesClosedForm) {
    const double i = 120.0 * kDeg, raan = 200.0 * kDeg, argp = 300.0 * kDeg;
    const State peri = state_from(kHypA, kHypE, i, raan, argp, 0.0);
    const double n = std::sqrt(earth::mu / (-kHypA * -kHypA * -kHypA));
    const double t15 = 0.6251813249152555 / n;
    EXPECT_NEAR(t15, 1597.6746156898553, 1e-9);
    const State out = state_from(kHypA, kHypE, i, raan, argp, 1.5);
    const State in = state_from(kHypA, kHypE, i, raan, argp, -1.5);
    const State fwd = simcore::kepler_propagate(peri, peri.t + t15);
    const State bwd = simcore::kepler_propagate(peri, peri.t - t15);
    EXPECT_LE(rel_err(fwd.r_eci, out.r_eci), 1e-12);
    EXPECT_LE(rel_err(fwd.v_eci, out.v_eci), 1e-12);
    EXPECT_LE(rel_err(bwd.r_eci, in.r_eci), 1e-12);
    EXPECT_LE(rel_err(bwd.v_eci, in.v_eci), 1e-12);
}

// Energy v^2/2 - mu/r and angular momentum r x v are invariants of the two-body problem.
// Propagated states rebuild them from elements, so they hold to roundoff: 1e-12 relative.
TEST(KeplerPropagate, ConservesEnergyAndAngularMomentum) {
    const std::vector<State> starts{
        state_from(2.0e7, 0.99, 0.4, 1.0, 2.0, 0.3),
        state_from(kHypA, kHypE, 2.0, 3.0, 4.0, -2.0),
        orbit_fixtures::circular_initial_state(),
    };
    for (const State& s0 : starts) {
        const double E0 = 0.5 * s0.v_eci.squaredNorm() - earth::mu / s0.r_eci.norm();
        const Eigen::Vector3d h0 = s0.r_eci.cross(s0.v_eci);
        for (const double dt : {-5000.0, 100.0, 7777.7, 50000.0}) {
            SCOPED_TRACE(dt);
            const State s = simcore::kepler_propagate(s0, s0.t + dt);
            const double E = 0.5 * s.v_eci.squaredNorm() - earth::mu / s.r_eci.norm();
            EXPECT_NEAR(E / E0, 1.0, 1e-12);
            EXPECT_LE(rel_err(s.r_eci.cross(s.v_eci), h0), 1e-12);
        }
    }
}

TEST(KeplerPropagate, RejectsParabolicAndInvalidInput) {
    const double r = earth::radius + 500.0e3;
    const State parabolic{orbit_fixtures::start_epoch(), Eigen::Vector3d(r, 0.0, 0.0),
                          Eigen::Vector3d(0.0, std::sqrt(2.0 * earth::mu / r), 0.0)};
    EXPECT_THROW((void)simcore::kepler_propagate(parabolic, parabolic.t + 10.0),
                 std::invalid_argument);
    const State s0 = orbit_fixtures::eccentric_apoapsis_state();
    EXPECT_THROW((void)simcore::kepler_propagate(s0, s0.t + 10.0, 0.0), std::invalid_argument);
}

// ---------------------------------------------------------------------------------------------
// Milestone M1: analytic and numerical propagation agree over 100 orbits
// ---------------------------------------------------------------------------------------------
// Dormand-Prince 5(4) at rtol 1e-12, atol 1e-6 m / 1e-9 m/s against kepler_propagate.
//
// Bound for the e = 0.7 orbit (100 T = 3.39e6 s): M1.2's approved one-orbit bound at default
// tolerances (rtol 1e-10, atol 1e-3 m / 1e-6 m/s) is 21.9 m, and M1.2 verified error proportional
// to tolerance. These tolerances are >= 100x tighter, so <= 0.219 m per orbit; the along-track
// error grows at most linearly with orbits (a secular energy error is a fixed period error), so
// <= 21.9 m over 100 orbits. Bound 25 m; velocity scaled by v_p / r_p: 25 m * 1.47e-3 /s = 0.037
// m/s.
TEST(Milestone, NumericalMatchesKeplerOverHundredEllipticOrbits) {
    simcore::PointMassGravity g;
    simcore::DpOptions o;
    o.rtol = 1e-12;
    o.atol_pos_m = 1e-6;
    o.atol_vel_mps = 1e-9;
    simcore::DormandPrince54 dp(o);
    const State s0 = state_from(orbit_fixtures::kEccSemiMajorAxis_m, 0.7, 51.6 * kDeg, 30.0 * kDeg,
                                75.0 * kDeg, 2.0);
    const simcore::Epoch t_end = s0.t + 100.0 * orbit_fixtures::eccentric_period_s();
    const State num = dp.propagate(g, s0, t_end);
    const State ana = simcore::kepler_propagate(s0, t_end);
    EXPECT_LE((num.r_eci - ana.r_eci).norm(), 25.0);
    EXPECT_LE((num.v_eci - ana.v_eci).norm(), 0.037);
}

// Hyperbolic flyby from nu = -2 to +2 rad (both well inside nu_inf = 2.30): t = 2 M(2) / n.
// Per-step error is held at sc_r = atol + rtol |r| <= 1e-6 + 1e-12 * 4.6e7 = 4.7e-5 m (|r| peaks
// at p / (1 + e cos 2) = 4.6e7 m); with N accepted steps the worst-case linear sum is N sc_r.
// We assert N < 2000 (so the sum is < 0.094 m) and the position error < 0.1 m; velocity < 1e-4 m/s
// (sc_v sum: N (1e-9 + 1e-12 * 1.2e4) < 2.6e-5 m/s, plus the r error times the local rate).
TEST(Milestone, NumericalMatchesKeplerOnHyperbolicFlyby) {
    simcore::PointMassGravity g;
    simcore::DpOptions o;
    o.rtol = 1e-12;
    o.atol_pos_m = 1e-6;
    o.atol_vel_mps = 1e-9;
    simcore::DormandPrince54 dp(o);
    const State s0 = state_from(kHypA, kHypE, 120.0 * kDeg, 200.0 * kDeg, 300.0 * kDeg, -2.0);
    const double n = std::sqrt(earth::mu / (-kHypA * -kHypA * -kHypA));
    const double dt = 2.0 * simcore::mean_from_true_anomaly(2.0, kHypE) / n;
    const State num = dp.propagate(g, s0, s0.t + dt);
    const State ana = simcore::kepler_propagate(s0, s0.t + dt);
    EXPECT_LT(dp.stats().accepted, 2000);
    EXPECT_LE((num.r_eci - ana.r_eci).norm(), 0.1);
    EXPECT_LE((num.v_eci - ana.v_eci).norm(), 1e-4);
}

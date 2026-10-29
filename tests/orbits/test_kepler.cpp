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

// Small M near e = 1, where f' = 1 - e cos E is tiny and an absolute stopping test is too loose:
// the solution must be accurate RELATIVE to E. The absolute rounding of f is a few eps times the
// size of its terms, so |E - e sin E - M| <= 8 eps (|E| + e |sin E| + |M|).
// Scale of the answers: for M << (1-e)^1.5, E ~ M / (1 - e) (e = 0.999999, M = 1e-14 -> 1e-8);
// for (1 - e) << E, E ~ (6 M)^(1/3).
TEST(KeplerEquation, EllipticSmallMeanAnomalyNearParabolicIsRelativelyAccurate) {
    for (const double e : {0.9999, 0.999999}) {
        for (const double M : {1e-14, 1e-10, 1e-6, 1e-3}) {
            for (const double sgn : {1.0, -1.0}) {
                SCOPED_TRACE(testing::Message() << "e=" << e << " M=" << sgn * M);
                const double E = simcore::solve_kepler_elliptic(sgn * M, e);
                const double scale = std::abs(E) + e * std::abs(std::sin(E)) + M;
                EXPECT_LE(std::abs(E - e * std::sin(E) - sgn * M), 8.0 * kEps * scale);
                EXPECT_GT(sgn * E, 0.0);
            }
        }
    }
}

// Hyperbolic counterpart: e = 1.0001 with small M, where H ~ M / (e - 1).
TEST(KeplerEquation, HyperbolicSmallMeanAnomalyNearParabolicIsRelativelyAccurate) {
    for (const double M : {1e-14, 1e-10, 1e-6, 1e-3}) {
        SCOPED_TRACE(M);
        const double e = 1.0001;
        const double H = simcore::solve_kepler_hyperbolic(M, e);
        const double scale = e * std::abs(std::sinh(H)) + std::abs(H) + M;
        EXPECT_LE(std::abs(e * std::sinh(H) - H - M), 8.0 * kEps * scale);
        EXPECT_GT(H, 0.0);
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

// Timing at high eccentricity, against an independent closed form (the energy/momentum test above
// can't catch a wrong time -> anomaly map: the state is rebuilt from a, e, i, raan, argp, so E
// and h are conserved for any nu). From nu0 = -2.5 rad (inbound) through periapsis to nu1 = +2.5:
// t = (M(nu1) - M(nu0)) / n with M from the test's own formula (E = 2 atan2(sqrt(1-e) sin(nu/2),
// sqrt(1+e) cos(nu/2)), M = E - e sin E).
// Sensitivity: near periapsis a mean-anomaly error dM moves the state by v_p dM / n, a relative
// error kappa dM with kappa = v_p / (n r_p) = sqrt((1+e) / (1-e)^3) (1411 at e = 0.99, 1.4e6 at
// 0.9999). dM is M's rounding (~eps) plus the vis-viva cancellation in a (~v^2/(2|E|) eps ~ 200 eps
// at e = 0.99). Bound 1e-13 kappa relative.
// The same bound applies to propagating in two pieces vs one (composition), measured 5e-13 at
// e = 0.99 and 4.5e-11 at e = 0.9999 by the M1 review.
TEST(KeplerPropagate, HighEccentricityTimingAndCompositionThroughPeriapsis) {
    const double rp = earth::radius + 500.0e3;
    for (const double e : {0.99, 0.9999}) {
        SCOPED_TRACE(e);
        const double a = rp / (1.0 - e);
        const double n = std::sqrt(earth::mu / (a * a * a));
        const auto M_of = [e](double nu) {
            const double E = 2.0 * std::atan2(std::sqrt(1.0 - e) * std::sin(0.5 * nu),
                                              std::sqrt(1.0 + e) * std::cos(0.5 * nu));
            return E - e * std::sin(E);
        };
        const double i = 0.7, raan = 1.1, argp = 2.3;
        const State s0 = state_from(a, e, i, raan, argp, -2.5);
        const State ref = state_from(a, e, i, raan, argp, 2.5);
        const double dt = (M_of(2.5) - M_of(-2.5)) / n;
        const double kappa = std::sqrt((1.0 + e) / ((1.0 - e) * (1.0 - e) * (1.0 - e)));
        const double bound = 1e-13 * kappa;

        const State one = simcore::kepler_propagate(s0, s0.t + dt);
        EXPECT_LE(rel_err(one.r_eci, ref.r_eci), bound);
        EXPECT_LE(rel_err(one.v_eci, ref.v_eci), bound);

        const State mid = simcore::kepler_propagate(s0, s0.t + 0.37 * dt);
        const State two = simcore::kepler_propagate(mid, s0.t + dt);
        EXPECT_LE(rel_err(two.r_eci, one.r_eci), bound);
        EXPECT_LE(rel_err(two.v_eci, one.v_eci), bound);
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
// Bound for the e = 0.7 orbit (100 T = 3.39e6 s). This is an empirical regression bound, not a
// worst case. The energy error is secular (it grows ~linearly with orbits: measured dE/E =
// -2.5e-12, -2.5e-11, -2.5e-10 after 1, 10, 100 orbits), so the semi-major axis drifts, the period
// error grows with it and the along-track error grows ~quadratically (dn/n = -1.5 da/a,
// integrated over time). Measured |dr| = 4.1e-5 m, 0.031 m, 3.49 m after 1, 10, 100 orbits
// (~110x per decade, i.e. quadratic). A worst-case derivation from the per-step tolerance gives
// kilometres, which would catch nothing. 25 m is ~7x the measured 3.49 m: a regression in step
// control or the tableau (km-level) still fails it. Velocity bound scaled by v_p / r_p:
// 25 m * 1.47e-3 /s = 0.037 m/s.
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

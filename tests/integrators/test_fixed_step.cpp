// M1.2: fixed-step integrators — velocity Verlet (2nd order, symplectic) and classical RK4
// (4th order), plus propagate_fixed (equal steps landing exactly on t_end).
//
// Both integrate the second-order ODE r'' = a(t, r, v), written as the first-order system
// y = [r; v], y' = [v; a].
//
// Velocity Verlet (kick-drift-kick):
//   v_half = v + h/2 a(t, r, v);  r1 = r + h v_half;  v1 = v_half + h/2 a(t+h, r1, v_half)
// For a = a(r) it is the composition of two exact flows (a "kick" that changes only v and a
// "drift" that changes only r), each of which is a symplectic map, so the composition is
// symplectic. Backward error analysis: a symplectic method exactly conserves a modified
// Hamiltonian H~ = H + h^2 H2 + O(h^4), so the true energy error stays bounded (oscillates) for
// exponentially long times instead of drifting. It is also symmetric: Phi_{-h} o Phi_{h} = id.
//
// RK4 matches the Taylor series to h^4: local error O(h^5), global error O(h^4). It is neither
// symplectic nor symmetric, so its energy error drifts secularly.
//
// Roundoff vocabulary used below: eps = 2.22e-16 = ulp(1.0) (machine epsilon); one floating-point
// operation rounds by <= eps/2 relative.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>

#include "orbit_fixtures.hpp"
#include "simcore/constants.hpp"
#include "simcore/dynamics/force_model.hpp"
#include "simcore/dynamics/point_mass_gravity.hpp"
#include "simcore/dynamics/state.hpp"
#include "simcore/integrators/fixed_step.hpp"
#include "simcore/orbits/epoch.hpp"

namespace {

using simcore::Epoch;
using simcore::FixedStepMethod;
using simcore::ForceModel;
using simcore::PointMassGravity;
using simcore::State;
using namespace orbit_fixtures;

State step(FixedStepMethod m, const ForceModel& f, const State& s, double h_s) {
    return m == FixedStepMethod::kRk4 ? simcore::rk4_step(f, s, h_s)
                                      : simcore::verlet_step(f, s, h_s);
}

// N steps of size h with the raw step function.
State run_steps(FixedStepMethod m, const ForceModel& f, State s, double h_s, int n_steps) {
    for (int k = 0; k < n_steps; ++k) {
        s = step(m, f, s, h_s);
    }
    return s;
}

// Position error after exactly one circular period using N equal steps of T/N.
double one_orbit_error_m(FixedStepMethod m, int n_steps) {
    const PointMassGravity g;
    const double T = circular_period_s();
    const State s = run_steps(m, g, circular_initial_state(), T / n_steps, n_steps);
    return (s.r_eci - circular_initial_state().r_eci).norm();
}

// A force that depends on t, r AND v, so the one-step formula tests can see whether every stage
// receives the right time and velocity: a damped harmonic oscillator plus a thrust ramp,
//   a = -w^2 r - gamma v + b (t - t_ref) x_hat.
class DampedRampForce final : public ForceModel {
public:
    [[nodiscard]] Eigen::Vector3d acceleration_eci(const State& s) const override {
        return eval(s.t, s.r_eci, s.v_eci);
    }
    static Eigen::Vector3d eval(const Epoch& t, const Eigen::Vector3d& r,
                                const Eigen::Vector3d& v) {
        return -kW2 * r - kGamma * v + kB * (t - start_epoch()) * Eigen::Vector3d::UnitX();
    }
    static constexpr double kW2 = 1e-6;     // w = 1e-3 rad/s [1/s^2]
    static constexpr double kGamma = 1e-4;  // [1/s]
    static constexpr double kB = 1e-3;      // [m/s^3]
};

// ~20 flops per output component in both the implementation and the hand formula below, each
// rounding by <= eps/2 relative to quantities no larger than |r| (or |v|): worst case ~2e-15
// relative. 1e-14 leaves margin for a different (equally valid) evaluation order; any wrong
// coefficient, stage time or stage velocity is an O(h^2 gamma |a|)-or-larger change, i.e. >= 1e-9
// relative here, five orders above the tolerance.
constexpr double kRoundoffRel = 1e-14;

}  // namespace

// ---------------------------------------------------------------------------------------------
// One-step formulas (time- and velocity-dependent force)
// ---------------------------------------------------------------------------------------------

TEST(VelocityVerlet, StepMatchesKickDriftKickFormula) {
    const DampedRampForce f;
    const State s{start_epoch() + 100.0, Eigen::Vector3d(7.0e6, -2.0e6, 3.0e6),
                  Eigen::Vector3d(1.0e3, 7.0e3, -2.0e3)};
    for (const double h : {60.0, -60.0}) {
        SCOPED_TRACE(h);
        // Documented convention: the second evaluation uses (t + h, r1, v_half).
        const Eigen::Vector3d a0 = DampedRampForce::eval(s.t, s.r_eci, s.v_eci);
        const Eigen::Vector3d v_half = s.v_eci + 0.5 * h * a0;
        const Eigen::Vector3d r1 = s.r_eci + h * v_half;
        const Eigen::Vector3d a1 = DampedRampForce::eval(s.t + h, r1, v_half);
        const Eigen::Vector3d v1 = v_half + 0.5 * h * a1;

        const State out = simcore::verlet_step(f, s, h);
        EXPECT_EQ(out.t, s.t + h);  // 100.25 +/- 60: dyadic, exact
        EXPECT_LT((out.r_eci - r1).norm(), kRoundoffRel * r1.norm());
        EXPECT_LT((out.v_eci - v1).norm(), kRoundoffRel * v1.norm());
    }
}

TEST(Rk4, StepMatchesClassicalTableau) {
    // y' = F(t, y) with y = [r; v], F = [v; a]. Classical RK4:
    //   k1 = F(t, y), k2 = F(t + h/2, y + h/2 k1), k3 = F(t + h/2, y + h/2 k2),
    //   k4 = F(t + h, y + h k3),  y1 = y + h/6 (k1 + 2 k2 + 2 k3 + k4).
    const DampedRampForce f;
    const State s{start_epoch() + 100.0, Eigen::Vector3d(7.0e6, -2.0e6, 3.0e6),
                  Eigen::Vector3d(1.0e3, 7.0e3, -2.0e3)};
    for (const double h : {60.0, -60.0}) {
        SCOPED_TRACE(h);
        const Epoch tm = s.t + 0.5 * h;
        const Epoch t1 = s.t + h;
        const Eigen::Vector3d k1r = s.v_eci;
        const Eigen::Vector3d k1v = DampedRampForce::eval(s.t, s.r_eci, s.v_eci);
        const Eigen::Vector3d k2r = s.v_eci + 0.5 * h * k1v;
        const Eigen::Vector3d k2v = DampedRampForce::eval(tm, s.r_eci + 0.5 * h * k1r, k2r);
        const Eigen::Vector3d k3r = s.v_eci + 0.5 * h * k2v;
        const Eigen::Vector3d k3v = DampedRampForce::eval(tm, s.r_eci + 0.5 * h * k2r, k3r);
        const Eigen::Vector3d k4r = s.v_eci + h * k3v;
        const Eigen::Vector3d k4v = DampedRampForce::eval(t1, s.r_eci + h * k3r, k4r);
        const Eigen::Vector3d r1 = s.r_eci + h / 6.0 * (k1r + 2.0 * k2r + 2.0 * k3r + k4r);
        const Eigen::Vector3d v1 = s.v_eci + h / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);

        const State out = simcore::rk4_step(f, s, h);
        EXPECT_EQ(out.t, t1);
        EXPECT_LT((out.r_eci - r1).norm(), kRoundoffRel * r1.norm());
        EXPECT_LT((out.v_eci - v1).norm(), kRoundoffRel * v1.norm());
    }
}

TEST(FixedStep, StepsEvaluateForceAtStageTimes) {
    // t = start + 100 s, h = 60 s: every stage time (t, t + 30, t + 60) is dyadic -> exact.
    const State s0{start_epoch() + 100.0, circular_initial_state().r_eci,
                   circular_initial_state().v_eci};
    for (const double h : {60.0, -60.0}) {
        SCOPED_TRACE(h);
        RecordingGravity vv;
        (void)simcore::verlet_step(vv, s0, h);
        ASSERT_EQ(vv.calls.size(), 2u);
        EXPECT_EQ(vv.calls[0].t, s0.t);
        EXPECT_EQ(vv.calls[1].t, s0.t + h);

        RecordingGravity rk;
        (void)simcore::rk4_step(rk, s0, h);
        ASSERT_EQ(rk.calls.size(), 4u);
        EXPECT_EQ(rk.calls[0].t, s0.t);
        EXPECT_EQ(rk.calls[1].t, s0.t + 0.5 * h);
        EXPECT_EQ(rk.calls[2].t, s0.t + 0.5 * h);
        EXPECT_EQ(rk.calls[3].t, s0.t + h);
    }
}

// ---------------------------------------------------------------------------------------------
// Convergence order on the circular 420 km orbit (1 period, analytic reference)
// ---------------------------------------------------------------------------------------------
// For an order-p method the global error is e(h) = C h^p (1 + D h + O(h^2)), so halving h gives
// e(h) / e(h/2) = 2^p (1 + D h / 2 + ...), and the observed order is p_obs = log2 of that ratio.
//
// Step sizes: N = 400 and 800 steps per orbit (h = 13.95 s and 6.97 s, n h = 2 pi / N = 0.016 and
// 0.008 rad per step) — in the asymptotic range, where the O(h) correction is a few percent.
// Roundoff floor: each step injects ~eps |r| = 1.5e-9 m; along-track growth over one orbit is at
// most 1 + 3 n t = 1 + 6 pi ~ 20, so even a worst-case linear sum over 800 steps is
// 800 * 1.5e-9 * 20 = 2.4e-5 m — under 1% of the smallest error measured here (RK4, N = 800,
// ~4e-3 m), so it cannot move p_obs by more than ~0.01.
//
// Band: |p_obs - p| <= 0.2 (ratio within [2^(p-0.2), 2^(p+0.2)]). It allows the pre-asymptotic
// O(h) correction (expected ~0.05 in p at these steps) with 4x margin, and still excludes the
// neighbouring orders p +/- 1 (ratio off by a factor 2) by a wide margin.
// Measured with a scratch reference implementation: RK4 ratio 16.69 (p_obs 4.06), errors
// 6.49e-2 m -> 3.89e-3 m; Verlet ratio 3.999 (p_obs 2.00), errors 3513 m -> 878 m.

TEST(Rk4, ConvergesAtFourthOrderOnCircularOrbit) {
    const double e400 = one_orbit_error_m(FixedStepMethod::kRk4, 400);
    const double e800 = one_orbit_error_m(FixedStepMethod::kRk4, 800);
    const double p_obs = std::log2(e400 / e800);
    EXPECT_NEAR(p_obs, 4.0, 0.2) << "errors " << e400 << " m, " << e800 << " m";
}

TEST(VelocityVerlet, ConvergesAtSecondOrderOnCircularOrbit) {
    const double e400 = one_orbit_error_m(FixedStepMethod::kVelocityVerlet, 400);
    const double e800 = one_orbit_error_m(FixedStepMethod::kVelocityVerlet, 800);
    const double p_obs = std::log2(e400 / e800);
    EXPECT_NEAR(p_obs, 2.0, 0.2) << "errors " << e400 << " m, " << e800 << " m";
}

// ---------------------------------------------------------------------------------------------
// Symplectic energy behaviour over 100 orbits
// ---------------------------------------------------------------------------------------------
// Orbit: e = 0.7 (periapsis 420 km), so the energy error has a strong once-per-orbit spike at
// periapsis — the bounded oscillation is visible. h = T / 1000 = 33.9 s for BOTH methods
// (h / (r_p / v_p) = 33.9 / 681 = 0.05 at periapsis). 100 orbits = 100,000 steps each.
//
// Verlet: |E(t) - E0| = |h^2 (H2(y0) - H2(y(t)))| + O(h^4): a bounded function of where the
//   satellite is on its (slowly precessing) orbit. The maximum over orbits 91-100 therefore equals
//   the maximum over orbits 1-10, up to (i) the sampling grid hitting the periapsis spike at a
//   slightly different phase, relative change ~(h / (r_p/v_p))^2 = 0.25%, and (ii) roundoff drift
//   ~sqrt(1e5) eps |E| (1e-13 relative), both negligible. Bound: ratio <= 1.1.
// RK4: loses a fixed amount delta of energy per orbit (secular), plus a bounded periapsis
//   oscillation A. max over orbits 91-100 ~ 100 delta + A, over 1-10 ~ 10 delta + A. Pure drift
//   gives a ratio of 10; we require > 5, i.e. the drift dominates (holds whenever A < 12.5 delta).
//   Also the energy sampled at the end of every orbit must move in the same direction every orbit
//   (monotone drift): per-orbit change ~delta = 0.23 J/kg vs roundoff ~1e-9 J/kg.
// Measured (scratch reference): Verlet ratio 1.0000 (max |dE| 8988 J/kg, 1.0e-3 relative);
//   RK4 ratio 8.05 (2.99 -> 24.0 J/kg), monotone.
TEST(VelocityVerlet, EnergyErrorStaysBoundedOver100OrbitsWhileRk4Drifts) {
    const PointMassGravity g;
    constexpr int kStepsPerOrbit = 1000;
    constexpr int kOrbits = 100;
    const double h = eccentric_period_s() / kStepsPerOrbit;

    struct Result {
        double max_first10 = 0.0;
        double max_last10 = 0.0;
        std::vector<double> end_of_orbit_dE;
    };
    const auto run = [&](FixedStepMethod m) {
        Result res;
        State s = eccentric_periapsis_state();
        const double e0 = specific_energy(s);
        for (int orbit = 0; orbit < kOrbits; ++orbit) {
            for (int k = 0; k < kStepsPerOrbit; ++k) {
                s = step(m, g, s, h);
                const double de = std::abs(specific_energy(s) - e0);
                if (orbit < 10) {
                    res.max_first10 = std::max(res.max_first10, de);
                }
                if (orbit >= kOrbits - 10) {
                    res.max_last10 = std::max(res.max_last10, de);
                }
            }
            res.end_of_orbit_dE.push_back(specific_energy(s) - e0);
        }
        return res;
    };

    const Result vv = run(FixedStepMethod::kVelocityVerlet);
    EXPECT_GT(vv.max_first10, 0.0);
    EXPECT_LE(vv.max_last10, 1.1 * vv.max_first10)
        << "Verlet energy error grew: " << vv.max_first10 << " -> " << vv.max_last10 << " J/kg";

    const Result rk = run(FixedStepMethod::kRk4);
    EXPECT_GT(rk.max_last10, 5.0 * rk.max_first10)
        << "RK4 energy error did not drift: " << rk.max_first10 << " -> " << rk.max_last10;
    const double first_change = rk.end_of_orbit_dE[1] - rk.end_of_orbit_dE[0];
    ASSERT_NE(first_change, 0.0);
    for (std::size_t k = 1; k < rk.end_of_orbit_dE.size(); ++k) {
        SCOPED_TRACE(k);
        const double change = rk.end_of_orbit_dE[k] - rk.end_of_orbit_dE[k - 1];
        EXPECT_GT(change * first_change, 0.0) << "RK4 drift is not monotone";
    }
}

// ---------------------------------------------------------------------------------------------
// Angular momentum (central force)
// ---------------------------------------------------------------------------------------------
// In exact arithmetic Verlet conserves h = r x v exactly for a central force a = -k(r) r:
//   kick:  r x (v + c r) = r x v                      (r x r = 0)
//   drift: (r + h v_half) x v_half = r x v_half       (v_half x v_half = 0)
//   kick:  r1 x (v_half + c' r1) = r1 x v_half
// So only roundoff changes h. Per step: r rounds by <= ~eps |r| (norm over 3 components, one
// multiply-add), v by <= ~2 eps |v| (two kicks), and the computed a is parallel to r only to
// ~eps; |d(r x v)| <= |dr||v| + |r||dv| ~ (3 sqrt(3) eps) |r||v| ~ 5.2 eps |r||v|. On the e = 0.7
// orbit |r||v| <= |h| / cos(gamma_max) = 1.40 |h| (tan gamma_max = e / sqrt(1 - e^2)), so the
// per-step change is <= ~7.3 eps |h|; round up to 8 eps |h|. Worst case (all errors aligned) over
// N = 10,000 steps: 8e4 eps = 1.8e-11 relative. Realistic (random walk) ~sqrt(N) eps ~ 1e-14.
// Measured (scratch reference): Verlet 1.2e-14. RK4 at the same h is not angular-momentum
// conserving (truncation error O(h^4)); it must exceed the roundoff bound, proving the test can
// tell the difference.
TEST(VelocityVerlet, ConservesAngularMomentumToRoundoffForCentralForce) {
    const PointMassGravity g;
    constexpr int kSteps = 10'000;  // 10 orbits of e = 0.7 at T / 1000
    const double h = eccentric_period_s() / 1000.0;
    const State s0 = eccentric_periapsis_state();
    const Eigen::Vector3d h0 = angular_momentum(s0);
    const double bound_rel = kSteps * 8.0 * kEps;  // 1.8e-11

    const State vv = run_steps(FixedStepMethod::kVelocityVerlet, g, s0, h, kSteps);
    EXPECT_LT((angular_momentum(vv) - h0).norm(), bound_rel * h0.norm());

    const State rk = run_steps(FixedStepMethod::kRk4, g, s0, h, kSteps);
    EXPECT_GT((angular_momentum(rk) - h0).norm(), bound_rel * h0.norm());
}

// ---------------------------------------------------------------------------------------------
// Time reversibility
// ---------------------------------------------------------------------------------------------
// Verlet is symmetric: verlet_step(-h) exactly undoes verlet_step(h) in exact arithmetic (replay
// the three sub-steps backwards: the last kick is undone by the first kick of the -h step, etc.).
// So N steps forward + N back returns to the start up to roundoff only.
// Bound: 2N = 200 steps, each injecting <= ~4 eps |r| of position-equivalent roundoff (r update
// ~1 ulp per component, the two kicks ~1 ulp of v each, times r/v; summed in norm). An injected
// error grows at most like the along-track drift it causes: a radial/energy perturbation changes
// a by da/a ~ 2 dr/r, the mean motion by -3/2 da/a, giving along-track growth 1 + 3 n t; the net
// time any error is carried is <= N h = 6000 s, so G <= 1 + 3 * 1.126e-3 * 6000 = 21.3.
//   |dr| <= 200 * 4 * eps * 6.798e6 m * 21.3 = 2.6e-5 m;   |dv| <= n |dr| = 2.9e-8 m/s.
// Epoch: 60 s steps from a .25 fraction are dyadic, so time returns exactly.
// Measured (scratch reference): Verlet 3.2e-8 m, 3.7e-11 m/s. RK4 (not symmetric): 18.5 m, which
// must fail the bound.
TEST(VelocityVerlet, IsTimeReversible) {
    const PointMassGravity g;
    constexpr int kSteps = 100;
    constexpr double h = 60.0;
    const double n = circular_mean_motion_radps();
    const double tol_r = 2.0 * kSteps * 4.0 * kEps * kR420_m * (1.0 + 3.0 * n * kSteps * h);
    const double tol_v = n * tol_r;
    const State s0 = circular_initial_state();

    State vv = run_steps(FixedStepMethod::kVelocityVerlet, g, s0, h, kSteps);
    vv = run_steps(FixedStepMethod::kVelocityVerlet, g, vv, -h, kSteps);
    EXPECT_EQ(vv.t, s0.t);
    EXPECT_LT((vv.r_eci - s0.r_eci).norm(), tol_r);
    EXPECT_LT((vv.v_eci - s0.v_eci).norm(), tol_v);

    State rk = run_steps(FixedStepMethod::kRk4, g, s0, h, kSteps);
    rk = run_steps(FixedStepMethod::kRk4, g, rk, -h, kSteps);
    EXPECT_GT((rk.r_eci - s0.r_eci).norm(), tol_r) << "RK4 should not be reversible";
}

// ---------------------------------------------------------------------------------------------
// propagate_fixed
// ---------------------------------------------------------------------------------------------
// n = max(1, ceil(|T| / h_max)) equal steps of T / n; step k starts at s0.t + k T/n; result.t is
// exactly t_end. Checked through the times the force model is called at.

TEST(PropagateFixed, LandsExactlyOnTEndWithEqualStepsDyadic) {
    // T = 1000 s, h_max = 300 s -> n = ceil(3.33) = 4 steps of exactly 250 s (no 100 s stub).
    // All times are start (.25 fraction) + multiples of 125 s: exact, so EXPECT_EQ.
    const State s0 = circular_initial_state();
    const Epoch t_end = s0.t + 1000.0;

    RecordingGravity vv;
    const State out_vv =
        simcore::propagate_fixed(vv, s0, t_end, 300.0, FixedStepMethod::kVelocityVerlet);
    EXPECT_EQ(out_vv.t, t_end);
    ASSERT_EQ(vv.calls.size(), 8u);  // 4 steps x 2 evaluations
    for (std::size_t k = 0; k < 4; ++k) {
        SCOPED_TRACE(k);
        const double tk = 250.0 * static_cast<double>(k);
        EXPECT_EQ(vv.calls[2 * k].t, s0.t + tk);
        EXPECT_EQ(vv.calls[2 * k + 1].t, s0.t + (tk + 250.0));
    }

    RecordingGravity rk;
    const State out_rk = simcore::propagate_fixed(rk, s0, t_end, 300.0, FixedStepMethod::kRk4);
    EXPECT_EQ(out_rk.t, t_end);
    ASSERT_EQ(rk.calls.size(), 16u);  // 4 steps x 4 evaluations
    for (std::size_t k = 0; k < 4; ++k) {
        SCOPED_TRACE(k);
        const double tk = 250.0 * static_cast<double>(k);
        EXPECT_EQ(rk.calls[4 * k].t, s0.t + tk);
        EXPECT_EQ(rk.calls[4 * k + 1].t, s0.t + (tk + 125.0));
        EXPECT_EQ(rk.calls[4 * k + 2].t, s0.t + (tk + 125.0));
        EXPECT_EQ(rk.calls[4 * k + 3].t, s0.t + (tk + 250.0));
    }
}

TEST(PropagateFixed, HMaxThatDividesSpanExactlyGivesNoExtraStep) {
    // T / h_max = 1000 / 250 = 4 exactly -> ceil = 4, not 5.
    const State s0 = circular_initial_state();
    RecordingGravity vv;
    (void)simcore::propagate_fixed(vv, s0, s0.t + 1000.0, 250.0, FixedStepMethod::kVelocityVerlet);
    EXPECT_EQ(vv.calls.size(), 8u);
}

TEST(PropagateFixed, LandsExactlyOnTEndWithEqualStepsNonDyadic) {
    // T = 1000 s, h_max = 7 s -> n = ceil(142.857) = 143 steps of T/143 = 6.993007 s.
    // Expected Verlet evaluation times: s0.t + k T/n and s0.t + (k+1) T/n. These are not exact in
    // binary; each stage time is Epoch + double, where k T/n <= 1000 s rounds to ulp(1000) =
    // 1.1e-13 s and the fraction add to 1.1e-16 s. Two such roundings < 3e-13 s; tolerance 1e-12 s.
    // An accumulated or stub-step schedule would be off by a whole fraction of a step (>= 1e-3 s).
    constexpr double kTimeTol_s = 1e-12;
    const State s0 = circular_initial_state();
    const Epoch t_end = s0.t + 1000.0;
    const double h = 1000.0 / 143.0;

    RecordingGravity vv;
    const State out =
        simcore::propagate_fixed(vv, s0, t_end, 7.0, FixedStepMethod::kVelocityVerlet);
    EXPECT_EQ(out.t, t_end);
    ASSERT_EQ(vv.calls.size(), 2u * 143u);
    for (std::size_t k = 0; k < 143; ++k) {
        SCOPED_TRACE(k);
        const double kd = static_cast<double>(k);
        EXPECT_NEAR(vv.calls[2 * k].t - s0.t, kd * h, kTimeTol_s);
        EXPECT_NEAR(vv.calls[2 * k + 1].t - s0.t, (kd + 1.0) * h, kTimeTol_s);
    }
}

TEST(PropagateFixed, WorksBackwardInTime) {
    // T = -1000 s, h_max = 300 -> 4 steps of -250 s; times descend from s0.t. Dyadic -> exact.
    const State s0 = circular_initial_state();
    const Epoch t_end = s0.t - 1000.0;

    RecordingGravity rk;
    const State out = simcore::propagate_fixed(rk, s0, t_end, 300.0, FixedStepMethod::kRk4);
    EXPECT_EQ(out.t, t_end);
    ASSERT_EQ(rk.calls.size(), 16u);
    for (std::size_t k = 0; k < 4; ++k) {
        SCOPED_TRACE(k);
        const double tk = -250.0 * static_cast<double>(k);
        EXPECT_EQ(rk.calls[4 * k].t, s0.t + tk);
        EXPECT_EQ(rk.calls[4 * k + 1].t, s0.t + (tk - 125.0));
        EXPECT_EQ(rk.calls[4 * k + 3].t, s0.t + (tk - 250.0));
    }
}

TEST(PropagateFixed, VerletForwardThenBackwardReturnsToStart) {
    // Same reversibility claim and bound as VelocityVerlet.IsTimeReversible, through the
    // propagate_fixed API: 6000 s / 60 s = 100 steps each way, back-span is exactly -T so each
    // backward step is exactly -h.
    const PointMassGravity g;
    const double n = circular_mean_motion_radps();
    const double tol_r = 2.0 * 100.0 * 4.0 * kEps * kR420_m * (1.0 + 3.0 * n * 6000.0);
    const State s0 = circular_initial_state();

    const State fwd =
        simcore::propagate_fixed(g, s0, s0.t + 6000.0, 60.0, FixedStepMethod::kVelocityVerlet);
    const State back =
        simcore::propagate_fixed(g, fwd, s0.t, 60.0, FixedStepMethod::kVelocityVerlet);
    EXPECT_EQ(back.t, s0.t);
    EXPECT_LT((back.r_eci - s0.r_eci).norm(), tol_r);
    EXPECT_LT((back.v_eci - s0.v_eci).norm(), n * tol_r);
}

TEST(PropagateFixed, Rk4BackwardIsMirrorOfForward) {
    // The Kepler problem is time-reversible: the backward flow from (r, v) equals the forward flow
    // from (r, -v) with the velocity flipped back, Phi_{-t}(r, v) = F(Phi_t(r, -v)), F(r,v) =
    // (r,-v). RK4's formulas are sign-symmetric under (h, v) -> (-h, -v), so the two runs agree to
    // roundoff. Bound: two independent runs of 100 steps (quarter orbit), each with the
    // reversibility-style roundoff bound 100 * 4 eps |r| * (1 + 3 n t), n t = pi/2:
    //   2 * 100 * 4 * eps * 6.798e6 * (1 + 3 pi / 2) = 6.9e-6 m.
    // A backward run that silently went forward (e.g. |h| used) is off by ~r sqrt(2) ~ 9.6e6 m.
    const PointMassGravity g;
    const double quarter = circular_period_s() / 4.0;
    const double h_max = quarter / 99.5;  // ceil(99.5) = 100 steps
    const double tol_r = 2.0 * 100.0 * 4.0 * kEps * kR420_m * (1.0 + 3.0 * kPi / 2.0);
    const double tol_v = circular_mean_motion_radps() * tol_r;
    const State s0 = circular_initial_state();
    const State mirrored0{s0.t, s0.r_eci, -s0.v_eci};

    const State back =
        simcore::propagate_fixed(g, s0, s0.t - quarter, h_max, FixedStepMethod::kRk4);
    const State fwd =
        simcore::propagate_fixed(g, mirrored0, s0.t + quarter, h_max, FixedStepMethod::kRk4);
    EXPECT_LT((back.r_eci - fwd.r_eci).norm(), tol_r);
    EXPECT_LT((back.v_eci + fwd.v_eci).norm(), tol_v);

    // Sanity: it actually went a quarter orbit backward (to r0 rotated by -90 deg), to within RK4
    // truncation at h = 14 s (~1e-2 m, from the convergence test); 1 km just rules out direction
    // bugs.
    EXPECT_LT((back.r_eci - circular_analytic_state(-quarter).r_eci).norm(), 1.0e3);
}

TEST(PropagateFixed, ZeroDurationReturnsInitialState) {
    // T = 0 -> n = max(1, 0) = 1 step of h = 0: r + 0 v = r and v + 0 a = v exactly.
    const PointMassGravity g;
    const State s0 = circular_initial_state();
    for (const FixedStepMethod m : {FixedStepMethod::kVelocityVerlet, FixedStepMethod::kRk4}) {
        const State out = simcore::propagate_fixed(g, s0, s0.t, 60.0, m);
        EXPECT_EQ(out.t, s0.t);
        EXPECT_EQ(out.r_eci, s0.r_eci);
        EXPECT_EQ(out.v_eci, s0.v_eci);
    }
}

TEST(PropagateFixed, RejectsNonPositiveOrNonFiniteHMax) {
    constexpr double kInf = std::numeric_limits<double>::infinity();
    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    const PointMassGravity g;
    const State s0 = circular_initial_state();
    for (const FixedStepMethod m : {FixedStepMethod::kVelocityVerlet, FixedStepMethod::kRk4}) {
        for (const double bad : {0.0, -0.0, -60.0, kNaN, kInf, -kInf}) {
            SCOPED_TRACE(bad);
            EXPECT_THROW((void)simcore::propagate_fixed(g, s0, s0.t + 1000.0, bad, m),
                         std::invalid_argument);
        }
    }
}

TEST(PropagateFixed, IsDeterministic) {
    // Same inputs, same machine -> bit-identical outputs (CLAUDE.md replay requirement).
    const PointMassGravity g;
    const State s0 = eccentric_periapsis_state();
    const Epoch t_end = s0.t + 0.37 * eccentric_period_s();
    for (const FixedStepMethod m : {FixedStepMethod::kVelocityVerlet, FixedStepMethod::kRk4}) {
        const State a = simcore::propagate_fixed(g, s0, t_end, 30.0, m);
        const State b = simcore::propagate_fixed(g, s0, t_end, 30.0, m);
        EXPECT_EQ(a.t, b.t);
        EXPECT_EQ(a.r_eci, b.r_eci);
        EXPECT_EQ(a.v_eci, b.v_eci);
    }
}

// M1.2: adaptive Dormand-Prince 5(4) (DOPRI5) with error control.
//
// Each step computes 7 stages; the 5th-order solution y5 is propagated ("local extrapolation")
// and the difference to the embedded 4th-order solution, est = y5 - y4 = O(h^5), estimates the
// local error. The last stage is f(t + h, y5), reused as the first stage of the next step (FSAL),
// so an attempted step costs 6 new force evaluations.
//
// Error control (Hairer's RMS norm over the 6 components of y = [r; v]):
//   err = sqrt(mean_i (est_i / sc_i)^2),  sc_i = atol_i + rtol * max(|y0_i|, |y1_i|)
//   accept if err <= 1;  h_new = h * min(5, max(0.2, 0.9 err^(-1/5)))  (no growth after a reject)
//
// Error model used to derive the accuracy bounds below:
//  1. Per accepted step the controller guarantees sum_i (est_i/sc_i)^2 <= 6. By Cauchy-Schwarz
//     the induced specific-energy error dE = v.dv + g r_hat.dr (g = mu/r^2) obeys
//       |dE| <= sqrt(6) * sqrt((v sc_v)^2 + (g sc_r)^2),   sc_v <= atol_v + rtol |v|, etc.
//  2. The propagated y5 is one order more accurate than est: its local error is ~(h/tau) * est,
//     tau the local orbital time scale (1/n for a circular orbit, r^2/|h| in general), with an
//     O(1) constant taken as 1. Summed over the steps, sum(h_k/tau_k) = swept angle = 2 pi per
//     orbit, independent of the number of steps.
//  3. An energy error dE/E is a semi-major-axis error da/a = -dE/E, i.e. a period error
//     dT/T = 3/2 da/a, which turns into an along-track (timing) error growing linearly in time.
//  The bounds also assume every local error is aligned to change the energy the same way, so they
//  are worst-case; the measured errors (scratch reference) are 15-30x below them.
//  Because (2) is only a model, PropagatedSolutionIsFifthOrder checks it separately: a variant
//  that propagated y4 instead of y5 passed the accuracy bounds in the scratch reference (energy
//  errors partly cancel) but fails the order test (observed order 4.0 instead of 5).

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "orbit_fixtures.hpp"
#include "simcore/constants.hpp"
#include "simcore/dynamics/point_mass_gravity.hpp"
#include "simcore/dynamics/state.hpp"
#include "simcore/integrators/dense_output.hpp"
#include "simcore/integrators/dormand_prince.hpp"
#include "simcore/integrators/fixed_step.hpp"
#include "simcore/orbits/epoch.hpp"

namespace {

using simcore::DenseSegment;
using simcore::DormandPrince54;
using simcore::DpOptions;
using simcore::Epoch;
using simcore::PointMassGravity;
using simcore::StepAction;
using namespace orbit_fixtures;

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// state_at(t_end) of a segment vs the integrator's own end state: the interpolant at theta = 1
// reduces to y0 + (y1 - y0) (or an equivalent polynomial sum), <= ~1.5 ulp per component; in norm
// over 3 components <= sqrt(3) * 1.5 eps |y| = 2.6 eps |y|. 4 eps leaves margin. (Epoch rounding of
// the segment end, ~2e-16 s, moves the state by v * 2e-16 = 2e-12 m, far below 4 eps |r| = 6e-9 m.)
constexpr double kEndpointRel = 4.0 * kEps;

std::vector<DenseSegment> collect_segments(DormandPrince54& dp, const simcore::ForceModel& f,
                                           const simcore::State& s0, const Epoch& t_end,
                                           simcore::State* out = nullptr) {
    std::vector<DenseSegment> segs;
    const simcore::State end = dp.propagate(f, s0, t_end, [&](const DenseSegment& seg) {
        segs.push_back(seg);
        return StepAction::kContinue;
    });
    if (out != nullptr) {
        *out = end;
    }
    return segs;
}

// Bound for one e = 0.7 orbit started at periapsis (derivation: header comment, steps 1-3).
// The per-step energy error is largest at periapsis (v, g and rtol*|y| all maximal there), so
// evaluating it there and multiplying by the 2 pi of swept angle is an upper bound.
//   dE_step = sqrt(6) hypot(v_p sc_v, g_p sc_r);  dE = 2 pi dE_step;  dt = 3/2 (dE/|E|) T
//   |dr| <= v_p dt;  |dv| <= g_p dt   (a timing error at periapsis displaces r by v dt, v by a dt)
// Defaults (rtol 1e-10, atol 1e-3 m, 1e-6 m/s): dE_step = 0.0604 J/kg, dE/E = 4.31e-8,
// dt = 2.20e-3 s -> |dr| <= 21.9 m, |dv| <= 0.019 m/s. Measured: 0.68 m, 5.8e-4 m/s.
struct EccentricBound {
    double rel_energy;
    double dr_m;
    double dv_mps;
};
EccentricBound eccentric_one_orbit_bound(const DpOptions& o) {
    const double v_p = eccentric_periapsis_speed_mps();
    const double r_p = kPeriapsisRadius_m;
    const double g_p = earth::mu / (r_p * r_p);
    const double sc_v = o.atol_vel_mps + o.rtol * v_p;
    const double sc_r = o.atol_pos_m + o.rtol * r_p;
    const double de_step = std::sqrt(6.0) * std::hypot(v_p * sc_v, g_p * sc_r);
    const double energy = earth::mu / (2.0 * kEccSemiMajorAxis_m);  // |E|
    const double rel_e = 2.0 * kPi * de_step / energy;
    const double dt = 1.5 * rel_e * eccentric_period_s();
    return {rel_e, v_p * dt, g_p * dt};
}

DpOptions tolerance_scaled_options(double rtol) {
    // All three tolerances scaled together, so "tighter by 100x" means exactly that.
    DpOptions o;
    o.rtol = rtol;
    o.atol_pos_m = rtol * kR420_m;
    o.atol_vel_mps = rtol * circular_speed_mps();
    return o;
}

double eccentric_return_error_m(const DpOptions& o) {
    const PointMassGravity g;
    DormandPrince54 dp(o);
    const simcore::State s0 = eccentric_periapsis_state();
    const simcore::State s1 = dp.propagate(g, s0, s0.t + eccentric_period_s());
    return (s1.r_eci - s0.r_eci).norm();
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Order of the propagated solution
// ---------------------------------------------------------------------------------------------
// With huge tolerances every step is accepted (err << 1) and growth is capped by h_max, so the
// integrator runs at a fixed step h = T/N: a pure test of the propagated formula. Error after one
// circular orbit should fall as h^5: ratio 32 per halving.
// N = 400 / 800 (n h = 0.016 / 0.008). Band |p_obs - 5| <= 0.4, wider than RK4's 0.2 because
// DOPRI5's 5th-order error coefficients were minimised by design, so the next-order term is
// relatively larger and the asymptote is approached more slowly (scratch reference: p_obs 4.67 at
// 200/400, 4.90 at 400/800). Roundoff at N = 800 (error 3.1e-6 m) is a few 1e-8 to 1e-7 m: varying
// the orbit orientation changed the error by +/-3%, i.e. +/-0.04 in p_obs. The band still excludes
// order 4 (propagating y4 instead of y5 gives p_obs 4.0) by 0.6.
TEST(DormandPrince54, PropagatedSolutionIsFifthOrder) {
    const PointMassGravity g;
    const double T = circular_period_s();
    const auto error_with_steps = [&](int n_steps) {
        DpOptions o;
        o.rtol = 1e10;
        o.atol_pos_m = 1e10;
        o.atol_vel_mps = 1e10;
        o.h_initial_s = T / n_steps;
        o.h_max_s = T / n_steps * (1.0 + 1e-9);  // last step is clipped, never a stub
        DormandPrince54 dp(o);
        const simcore::State s0 = circular_initial_state();
        const simcore::State s = dp.propagate(g, s0, s0.t + T);
        EXPECT_EQ(dp.stats().accepted, n_steps);
        EXPECT_EQ(dp.stats().rejected, 0);
        return (s.r_eci - s0.r_eci).norm();
    };
    const double e400 = error_with_steps(400);
    const double e800 = error_with_steps(800);
    EXPECT_NEAR(std::log2(e400 / e800), 5.0, 0.4) << "errors " << e400 << " m, " << e800 << " m";
}

// ---------------------------------------------------------------------------------------------
// Accuracy against analytic references
// ---------------------------------------------------------------------------------------------

// Milestone M1 "done when": analytic and numerical agree over 100 orbits (circular case here;
// elliptic Kepler comparison comes in M1.4).
// rtol 1e-12, atol 1e-6 m / 1e-9 m/s. For the circular orbit |E| = v^2/2 and g = v^2/r, so
// step (1) gives |dE/E| <= 2 sqrt(6) sqrt((sc_v/v)^2 + (sc_r/r)^2) per step at estimate level:
//   sc_v = 1e-9 + 1e-12 * 7657.27 = 8.66e-9 m/s, sc_r = 1e-6 + 1e-12 * 6.798e6 = 7.80e-6 m
//   -> 2 * 2.449 * sqrt(1.131e-12^2 + 1.147e-12^2) = 7.89e-12.
// Step (2): times the swept angle 100 * 2 pi = 628.3 -> |dE/E| <= 4.96e-9 after 100 orbits.
// Step (3): the energy error grows linearly, so the along-track drift is
//   ds = 3/2 n r * integral(dE/E dt) = 3/2 (n t) r (dE/E)_end / 2
//      = 0.75 * 628.3 * 6.798e6 * 4.96e-9 = 15.9 m;   |dv| <= n ds = 0.018 m/s.
// Angular momentum: |dh|/h <= dv/v + dr/r -> half the energy bound, 2.48e-9.
// Measured (scratch reference): 0.72 m, 8.1e-4 m/s, dE/E = 2.3e-10 (43,624 steps).
TEST(DormandPrince54, Circular420km100OrbitsMatchesAnalyticRotation) {
    const PointMassGravity g;
    DpOptions o;
    o.rtol = 1e-12;
    o.atol_pos_m = 1e-6;
    o.atol_vel_mps = 1e-9;
    DormandPrince54 dp(o);

    const double v = circular_speed_mps();
    const double n = circular_mean_motion_radps();
    const double swept = 100.0 * 2.0 * kPi;
    const double sc_v = o.atol_vel_mps + o.rtol * v;
    const double sc_r = o.atol_pos_m + o.rtol * kR420_m;
    const double rel_e_step = 2.0 * std::sqrt(6.0) * std::hypot(sc_v / v, sc_r / kR420_m);
    const double rel_e_bound = swept * rel_e_step;                 // 4.96e-9
    const double dr_bound = 0.75 * swept * kR420_m * rel_e_bound;  // 15.9 m
    const double dv_bound = n * dr_bound;                          // 0.018 m/s
    const double rel_h_bound = 0.5 * rel_e_bound;                  // 2.48e-9

    const simcore::State s0 = circular_initial_state();
    const double span = 100.0 * circular_period_s();
    const simcore::State s1 = dp.propagate(g, s0, s0.t + span);
    const simcore::State exact = circular_analytic_state(s1.t - s0.t);

    EXPECT_LT((s1.r_eci - exact.r_eci).norm(), dr_bound);
    EXPECT_LT((s1.v_eci - exact.v_eci).norm(), dv_bound);
    EXPECT_LT(std::abs(specific_energy(s1) / specific_energy(s0) - 1.0), rel_e_bound);
    EXPECT_LT((angular_momentum(s1) - angular_momentum(s0)).norm(),
              rel_h_bound * angular_momentum(s0).norm());
}

// e = 0.7 from periapsis, default options: after exactly one period it is back at periapsis.
TEST(DormandPrince54, EccentricOrbitReturnsToPeriapsisAfterOnePeriod) {
    const PointMassGravity g;
    DormandPrince54 dp;  // defaults: rtol 1e-10, atol 1e-3 m, 1e-6 m/s
    const EccentricBound b = eccentric_one_orbit_bound(dp.options());  // 21.9 m, 0.019 m/s

    const simcore::State s0 = eccentric_periapsis_state();
    const simcore::State s1 = dp.propagate(g, s0, s0.t + eccentric_period_s());
    EXPECT_LT((s1.r_eci - s0.r_eci).norm(), b.dr_m);
    EXPECT_LT((s1.v_eci - s0.v_eci).norm(), b.dv_mps);
    EXPECT_LT(std::abs(specific_energy(s1) / specific_energy(s0) - 1.0), b.rel_energy);
}

// Tolerance proportionality. With est ~ C h^5 held at tol, h ~ tol^(1/5); y5's local error is
// ~h^6 ~ tol^(6/5) and there are ~tol^(-1/5) steps, so the global error ~ tol^1: 100x tighter
// tolerances -> ~100x smaller error. The exponent is only asymptotic (safety factor 0.9, integer
// step counts, step-size history), so we accept an exponent in [0.75, 1.25]: ratio in
// [100^0.75, 100^1.25] = [31.6, 316]. That still rejects "no improvement" and an order-of-magnitude
// mismatch either way. rtol 1e-9 vs 1e-11 keeps both errors (6.7 m, 0.043 m measured) far above
// roundoff (~1e-6 m). Measured ratio (scratch reference): 156 (exponent 1.10).
TEST(DormandPrince54, ErrorScalesRoughlyLinearlyWithTolerance) {
    const double loose = eccentric_return_error_m(tolerance_scaled_options(1e-9));
    const double tight = eccentric_return_error_m(tolerance_scaled_options(1e-11));
    const double ratio = loose / tight;
    EXPECT_GT(ratio, std::pow(100.0, 0.75)) << loose << " m vs " << tight << " m";
    EXPECT_LT(ratio, std::pow(100.0, 1.25)) << loose << " m vs " << tight << " m";
}

// Adaptive step: the controller holds a fixed relative error per step, and the local error of an
// orbit scales like (h / tau)^5 with tau ~ r^(3/2) / sqrt(mu) (local dynamical time), so
// h ~ r^(3/2).
// Classify segments by their start radius: "periapsis" r < 1.5 r_p, "apoapsis" r > 0.9 r_a. The
// worst-case comparison (largest periapsis step vs smallest apoapsis step) is predicted at
//   (0.9 r_a / 1.5 r_p)^(3/2) = (0.9 * 38.52e6 / (1.5 * 6.798e6))^1.5 = 6.27.
// We require > half of that (3.13): atol terms (atol_pos = 1e-3 m exceeds rtol r_p = 6.8e-4 m) and
// the per-component scaling perturb the pure r^(3/2) law. Measured: 6.34 (61.6 s vs 390.5 s).
TEST(DormandPrince54, StepIsSmallerNearPeriapsisThanNearApoapsis) {
    const PointMassGravity g;
    DormandPrince54 dp;
    const simcore::State s0 = eccentric_periapsis_state();
    const std::vector<DenseSegment> segs = collect_segments(dp, g, s0, s0.t + eccentric_period_s());

    double max_h_peri = 0.0;
    double min_h_apo = kInf;
    for (const DenseSegment& seg : segs) {
        const double r = seg.state_at(seg.t_begin()).r_eci.norm();
        if (r < 1.5 * kPeriapsisRadius_m) {
            max_h_peri = std::max(max_h_peri, seg.h_s());
        }
        if (r > 0.9 * kApoapsisRadius_m) {
            min_h_apo = std::min(min_h_apo, seg.h_s());
        }
    }
    ASSERT_GT(max_h_peri, 0.0);
    ASSERT_LT(min_h_apo, kInf);
    const double predicted =
        std::pow(0.9 * kApoapsisRadius_m / (1.5 * kPeriapsisRadius_m), 1.5);  // 6.27
    EXPECT_GT(min_h_apo / max_h_peri, 0.5 * predicted)
        << "periapsis max h " << max_h_peri << " s, apoapsis min h " << min_h_apo << " s";
}

// ---------------------------------------------------------------------------------------------
// Landing on t_end, direction, early stop
// ---------------------------------------------------------------------------------------------

TEST(DormandPrince54, LandsExactlyOnTEndAndNeverOversteps) {
    // Non-round span (1.37 orbits) so the last step must be clipped.
    RecordingGravity g;
    DormandPrince54 dp;
    const simcore::State s0 = circular_initial_state();
    const Epoch t_end = s0.t + 1.37 * circular_period_s();
    simcore::State out;
    const std::vector<DenseSegment> segs = collect_segments(dp, g, s0, t_end, &out);

    EXPECT_EQ(out.t, t_end);  // exactly, not to roundoff
    ASSERT_FALSE(segs.empty());
    EXPECT_EQ(static_cast<long>(segs.size()), dp.stats().accepted);
    EXPECT_EQ(segs.front().t_begin(), s0.t);
    EXPECT_EQ(segs.back().t_end(), t_end);
    for (std::size_t k = 0; k < segs.size(); ++k) {
        SCOPED_TRACE(k);
        EXPECT_GT(segs[k].h_s(), 0.0);
        EXPECT_LT(segs[k].t_begin(), segs[k].t_end());
        EXPECT_GE(segs[k].t_begin(), s0.t);
        EXPECT_LE(segs[k].t_end(), t_end);
        if (k > 0) {
            EXPECT_EQ(segs[k].t_begin(), segs[k - 1].t_end());  // contiguous, no gaps/overlaps
        }
    }
    // The returned state is the last step's end state.
    const simcore::State last = segs.back().state_at(t_end);
    EXPECT_LT((last.r_eci - out.r_eci).norm(), kEndpointRel * out.r_eci.norm());
    EXPECT_LT((last.v_eci - out.v_eci).norm(), kEndpointRel * out.v_eci.norm());
    // Not even a trial stage is evaluated outside [t0, t_end].
    for (const simcore::State& c : g.calls) {
        EXPECT_GE(c.t, s0.t);
        EXPECT_LE(c.t, t_end);
    }
}

TEST(DormandPrince54, BackwardIntegrationReturnsToPeriapsis) {
    // Kepler motion is time-reversible, so one period backward from periapsis also ends at
    // periapsis, with the same error model and bound as the forward case (21.9 m, 0.019 m/s).
    const PointMassGravity g;
    DormandPrince54 dp;
    const EccentricBound b = eccentric_one_orbit_bound(dp.options());
    const simcore::State s0 = eccentric_periapsis_state();
    const Epoch t_end = s0.t - eccentric_period_s();
    simcore::State out;
    const std::vector<DenseSegment> segs = collect_segments(dp, g, s0, t_end, &out);

    EXPECT_EQ(out.t, t_end);
    EXPECT_LT((out.r_eci - s0.r_eci).norm(), b.dr_m);
    EXPECT_LT((out.v_eci - s0.v_eci).norm(), b.dv_mps);
    ASSERT_FALSE(segs.empty());
    EXPECT_EQ(segs.front().t_begin(), s0.t);
    EXPECT_EQ(segs.back().t_end(), t_end);
    for (std::size_t k = 0; k < segs.size(); ++k) {
        SCOPED_TRACE(k);
        EXPECT_LT(segs[k].h_s(), 0.0);
        EXPECT_LE(segs[k].t_begin(), s0.t);
        EXPECT_GE(segs[k].t_end(), t_end);
        EXPECT_GT(segs[k].t_begin(), segs[k].t_end());
        if (k > 0) {
            EXPECT_EQ(segs[k].t_begin(), segs[k - 1].t_end());
        }
    }
}

TEST(DormandPrince54, ForwardThenBackwardReturnsToStart) {
    // Two legs, each bounded as above -> errors add: <= 2 * 21.9 m, 2 * 0.019 m/s.
    // Measured (scratch reference): 0.36 m, 3.1e-4 m/s.
    const PointMassGravity g;
    DormandPrince54 dp;
    const EccentricBound b = eccentric_one_orbit_bound(dp.options());
    const simcore::State s0 = eccentric_periapsis_state();
    const simcore::State fwd = dp.propagate(g, s0, s0.t + eccentric_period_s());
    const simcore::State back = dp.propagate(g, fwd, s0.t);
    EXPECT_EQ(back.t, s0.t);
    EXPECT_LT((back.r_eci - s0.r_eci).norm(), 2.0 * b.dr_m);
    EXPECT_LT((back.v_eci - s0.v_eci).norm(), 2.0 * b.dv_mps);
}

TEST(DormandPrince54, StopRequestEndsAtThatSegmentsEnd) {
    const PointMassGravity g;
    const simcore::State s0 = eccentric_periapsis_state();
    const Epoch t_end = s0.t + eccentric_period_s();
    for (const long stop_at : {1L, 5L}) {
        SCOPED_TRACE(stop_at);
        DormandPrince54 dp;
        long calls = 0;
        std::vector<DenseSegment> seen;
        const simcore::State out = dp.propagate(g, s0, t_end, [&](const DenseSegment& seg) {
            ++calls;
            seen.push_back(seg);
            return calls == stop_at ? StepAction::kStop : StepAction::kContinue;
        });
        EXPECT_EQ(calls, stop_at);
        EXPECT_EQ(dp.stats().accepted, stop_at);
        ASSERT_EQ(static_cast<long>(seen.size()), stop_at);
        const DenseSegment& last = seen.back();
        EXPECT_EQ(out.t, last.t_end());
        EXPECT_LT(out.t, t_end);
        const simcore::State at_end = last.state_at(last.t_end());
        EXPECT_LT((at_end.r_eci - out.r_eci).norm(), kEndpointRel * out.r_eci.norm());
        EXPECT_LT((at_end.v_eci - out.v_eci).norm(), kEndpointRel * out.v_eci.norm());
    }
}

TEST(DormandPrince54, ZeroDurationReturnsInitialState) {
    const PointMassGravity g;
    DormandPrince54 dp;
    const simcore::State s0 = circular_initial_state();
    long calls = 0;
    const simcore::State out = dp.propagate(g, s0, s0.t, [&](const DenseSegment&) {
        ++calls;
        return StepAction::kContinue;
    });
    EXPECT_EQ(out.t, s0.t);
    EXPECT_EQ(out.r_eci, s0.r_eci);
    EXPECT_EQ(out.v_eci, s0.v_eci);
    EXPECT_EQ(calls, 0);
    EXPECT_EQ(dp.stats().accepted, 0);
}

// ---------------------------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------------------------

TEST(DormandPrince54, StatsCountForceEvaluationsWithFsal) {
    // Explicit h_initial: 1 evaluation at t0, then 6 per attempted step (7 stages, the 7th is
    // reused as the next step's 1st: FSAL), accepted or rejected.
    RecordingGravity g;
    DpOptions o;
    o.h_initial_s = 60.0;
    DormandPrince54 dp(o);
    const simcore::State s0 = circular_initial_state();
    long callbacks = 0;
    (void)dp.propagate(g, s0, s0.t + circular_period_s(), [&](const DenseSegment&) {
        ++callbacks;
        return StepAction::kContinue;
    });
    const simcore::DpStats st = dp.stats();
    EXPECT_GT(st.accepted, 0);
    EXPECT_EQ(callbacks, st.accepted);
    EXPECT_EQ(st.force_evals, 1 + 6 * (st.accepted + st.rejected));
    EXPECT_EQ(static_cast<long>(g.calls.size()), st.force_evals);  // the count is honest

    // stats() is reset at the start of every propagate call, not accumulated.
    (void)dp.propagate(g, s0, s0.t + circular_period_s());
    EXPECT_EQ(dp.stats().accepted, st.accepted);
    EXPECT_EQ(dp.stats().rejected, st.rejected);
    EXPECT_EQ(dp.stats().force_evals, st.force_evals);
}

TEST(DormandPrince54, AutomaticInitialStepCostsOneExtraEvaluation) {
    // Hairer's starting-step heuristic (Hairer, Norsett & Wanner I, II.4) needs f at t0 and one
    // trial explicit-Euler point f(t0 + h0, y0 + h0 f0) to estimate the second derivative, so the
    // honest count is 2 + 6 per attempted step. (Interface note: the DpStats comment says
    // "1 initial"; that holds only for an explicit h_initial_s.)
    RecordingGravity g;
    DormandPrince54 dp;  // h_initial_s = 0 -> automatic
    const simcore::State s0 = circular_initial_state();
    (void)dp.propagate(g, s0, s0.t + circular_period_s());
    const simcore::DpStats st = dp.stats();
    EXPECT_EQ(st.force_evals, 2 + 6 * (st.accepted + st.rejected));
    EXPECT_EQ(static_cast<long>(g.calls.size()), st.force_evals);
}

TEST(DormandPrince54, OversizedInitialStepIsRejectedThenRecovers) {
    // h_initial = T/2 (2789 s) on a LEO orbit: the error estimate is enormous, so the first
    // attempts must be rejected (each shrinks h by at most 5x).
    const PointMassGravity g;
    DpOptions o;
    o.h_initial_s = 0.5 * circular_period_s();
    DormandPrince54 dp(o);
    const simcore::State s0 = circular_initial_state();
    const simcore::State out = dp.propagate(g, s0, s0.t + circular_period_s());
    const simcore::DpStats st = dp.stats();
    EXPECT_GT(st.rejected, 0);
    EXPECT_GT(st.accepted, 0);
    EXPECT_EQ(st.force_evals, 1 + 6 * (st.accepted + st.rejected));
    // Rejected steps leave no trace: the result is still accurate. One orbit is 1/100 of the
    // 100-orbit circular case; at the default tolerances the same derivation (step-level dE/E
    // 2 sqrt(6) hypot(sc_v/v, sc_r/r) with sc_v = 1.77e-6 m/s, sc_r = 1.68e-3 m -> 1.64e-9, times
    // 2 pi, times 0.75 (2 pi) r) gives 3.3e2 m.
    const double v = circular_speed_mps();
    const double sc_v = o.atol_vel_mps + o.rtol * v;
    const double sc_r = o.atol_pos_m + o.rtol * kR420_m;
    const double rel_e = 2.0 * kPi * 2.0 * std::sqrt(6.0) * std::hypot(sc_v / v, sc_r / kR420_m);
    const double dr_bound = 0.75 * 2.0 * kPi * kR420_m * rel_e;
    EXPECT_LT((out.r_eci - s0.r_eci).norm(), dr_bound);
}

// ---------------------------------------------------------------------------------------------
// Option validation and failure modes
// ---------------------------------------------------------------------------------------------

TEST(DormandPrince54, ConstructorRejectsInvalidOptions) {
    std::vector<std::function<void(DpOptions&)>> bad = {
        [](DpOptions& o) { o.rtol = 0.0; },
        [](DpOptions& o) { o.rtol = -1e-10; },
        [](DpOptions& o) { o.rtol = kNaN; },
        [](DpOptions& o) { o.rtol = kInf; },
        [](DpOptions& o) { o.atol_pos_m = -1e-3; },
        [](DpOptions& o) { o.atol_pos_m = kNaN; },
        [](DpOptions& o) { o.atol_pos_m = kInf; },
        [](DpOptions& o) { o.atol_vel_mps = -1e-6; },
        [](DpOptions& o) { o.atol_vel_mps = kNaN; },
        [](DpOptions& o) { o.atol_vel_mps = kInf; },
        [](DpOptions& o) { o.h_min_s = 0.0; },
        [](DpOptions& o) { o.h_min_s = -1.0; },
        [](DpOptions& o) { o.h_min_s = kNaN; },
        [](DpOptions& o) { o.h_max_s = kNaN; },
        [](DpOptions& o) { o.h_max_s = -1.0; },
        [](DpOptions& o) {  // h_min > h_max
            o.h_min_s = 10.0;
            o.h_max_s = 1.0;
        },
        [](DpOptions& o) { o.h_initial_s = -1.0; },
        [](DpOptions& o) { o.h_initial_s = kNaN; },
        [](DpOptions& o) { o.max_steps = 0; },
        [](DpOptions& o) { o.max_steps = -1; },
    };
    for (std::size_t i = 0; i < bad.size(); ++i) {
        SCOPED_TRACE(i);
        DpOptions o;
        bad[i](o);
        EXPECT_THROW((void)DormandPrince54{o}, std::invalid_argument);
    }
}

TEST(DormandPrince54, ConstructorRejectsInfiniteHMinAndHInitial) {
    // Not literally covered by "0 < h_min <= h_max, h_initial >= 0" (inf passes both), but an
    // infinite minimum or initial step is meaningless: h_min = inf would make every step a
    // "collapse", h_initial = inf is never a usable first step. Both must be finite.
    DpOptions a;
    a.h_min_s = kInf;
    EXPECT_THROW((void)DormandPrince54{a}, std::invalid_argument);
    DpOptions b;
    b.h_initial_s = kInf;
    EXPECT_THROW((void)DormandPrince54{b}, std::invalid_argument);
}

TEST(DormandPrince54, ConstructorAcceptsValidEdgeOptions) {
    DpOptions zero_atol;  // pure relative control is allowed while rtol > 0
    zero_atol.atol_pos_m = 0.0;
    zero_atol.atol_vel_mps = 0.0;
    DpOptions equal_h;  // h_min == h_max is allowed
    equal_h.h_min_s = 10.0;
    equal_h.h_max_s = 10.0;
    DpOptions explicit_h0;
    explicit_h0.h_initial_s = 30.0;
    for (const DpOptions& o : {DpOptions{}, zero_atol, equal_h, explicit_h0}) {
        EXPECT_NO_THROW((void)DormandPrince54{o});
    }
    const DormandPrince54 dp(explicit_h0);
    EXPECT_EQ(dp.options().h_initial_s, 30.0);
    EXPECT_EQ(dp.options().rtol, DpOptions{}.rtol);
}

TEST(DormandPrince54, StepSizeCollapseThrows) {
    // Start at apoapsis of the e = 0.7 orbit with h_min = 100 s. Near apoapsis the controller is
    // happy with ~400 s steps, so the run makes progress; approaching periapsis it needs
    // h ~ (r_p / r_a)^(3/2) * 400 s ~ 30 s < h_min -> must throw rather than silently accept an
    // over-tolerance step.
    const PointMassGravity g;
    DpOptions o;
    o.h_initial_s = 400.0;
    o.h_min_s = 100.0;
    DormandPrince54 dp(o);
    const simcore::State s0 = eccentric_apoapsis_state();
    long accepted_before_throw = 0;
    EXPECT_THROW((void)dp.propagate(g, s0, s0.t + eccentric_period_s(),
                                    [&](const DenseSegment&) {
                                        ++accepted_before_throw;
                                        return StepAction::kContinue;
                                    }),
                 std::invalid_argument);
    EXPECT_GT(accepted_before_throw, 0) << "collapse should happen mid-run, not on step 1";
}

TEST(DormandPrince54, ClippedFinalStepShorterThanHMinIsNotACollapse) {
    // A final step clipped to land on t_end can be arbitrarily short; that is not the controller
    // asking for a too-small step. Otherwise any t_end that happens to fall 1e-8 s after a step
    // boundary would throw.
    const PointMassGravity g;
    const simcore::State s0 = circular_initial_state();

    // (a) 10 steps of exactly 10 s (h_max caps growth), then a 1e-8 s remainder < h_min = 1e-6 s.
    DpOptions o;
    o.h_initial_s = 10.0;
    o.h_max_s = 10.0;
    DormandPrince54 dp(o);
    const Epoch t_end = s0.t + (100.0 + 1e-8);
    simcore::State out;
    EXPECT_NO_THROW(out = dp.propagate(g, s0, t_end));
    EXPECT_EQ(out.t, t_end);

    // (b) a whole span shorter than h_min.
    DormandPrince54 dp_auto;
    const Epoch t_short = s0.t + 1e-7;
    EXPECT_NO_THROW(out = dp_auto.propagate(g, s0, t_short));
    EXPECT_EQ(out.t, t_short);
}

TEST(DormandPrince54, NeverEvaluatesForceOutsideTheSpanEvenForShortSpans) {
    // The automatic starting-step heuristic evaluates f at a trial point t0 + h0; for a span
    // shorter than h0 (~9 s here) that trial must be clipped to the span too. Force models with
    // bounded validity (ephemeris tables, thrust arcs, events in M1.3) must never be queried
    // outside [t0, t_end].
    RecordingGravity g;
    DormandPrince54 dp;
    const simcore::State s0 = circular_initial_state();
    for (const double span : {1e-3, -1e-3}) {
        SCOPED_TRACE(span);
        g.calls.clear();
        const Epoch t_end = s0.t + span;
        (void)dp.propagate(g, s0, t_end);
        ASSERT_FALSE(g.calls.empty());
        for (const simcore::State& c : g.calls) {
            EXPECT_GE(c.t, std::min(s0.t, t_end));
            EXPECT_LE(c.t, std::max(s0.t, t_end));
        }
    }
}

TEST(DormandPrince54, ExceedingMaxStepsThrows) {
    // max_steps counts attempted steps (accepted + rejected). Find the exact count of a run, then
    // check the boundary: exactly that many is allowed, one fewer throws.
    const PointMassGravity g;
    const simcore::State s0 = circular_initial_state();
    const Epoch t_end = s0.t + circular_period_s();
    DpOptions o;
    o.h_initial_s = 60.0;
    DormandPrince54 probe(o);
    (void)probe.propagate(g, s0, t_end);
    const long attempts = probe.stats().accepted + probe.stats().rejected;
    ASSERT_GT(attempts, 2);

    o.max_steps = attempts;
    DormandPrince54 enough(o);
    EXPECT_NO_THROW((void)enough.propagate(g, s0, t_end));

    o.max_steps = attempts - 1;
    DormandPrince54 too_few(o);
    EXPECT_THROW((void)too_few.propagate(g, s0, t_end), std::invalid_argument);
}

// ---------------------------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------------------------

TEST(DormandPrince54, IsDeterministic) {
    // Same inputs on the same machine -> bit-identical results and identical step sequences,
    // whether the integrator object is reused or fresh.
    const PointMassGravity g;
    const simcore::State s0 = eccentric_periapsis_state();
    const Epoch t_end = s0.t + 1.3 * eccentric_period_s();

    const auto run = [&](DormandPrince54& dp) {
        std::vector<double> hs;
        const simcore::State out = dp.propagate(g, s0, t_end, [&](const DenseSegment& seg) {
            hs.push_back(seg.h_s());
            return StepAction::kContinue;
        });
        return std::make_pair(out, hs);
    };
    DormandPrince54 a;
    DormandPrince54 b;
    const auto r1 = run(a);
    const auto r2 = run(a);  // reused object
    const auto r3 = run(b);  // fresh object
    for (const auto* r : {&r2, &r3}) {
        EXPECT_EQ(r->first.t, r1.first.t);
        EXPECT_EQ(r->first.r_eci, r1.first.r_eci);
        EXPECT_EQ(r->first.v_eci, r1.first.v_eci);
        EXPECT_EQ(r->second, r1.second);
    }
}

// ---------------------------------------------------------------------------------------------
// Efficiency vs fixed-step RK4 (README claim)
// ---------------------------------------------------------------------------------------------
// On e = 0.7 a fixed step must be small enough for periapsis everywhere, while the adaptive step
// grows like r^(3/2) (see StepIsSmallerNearPeriapsisThanNearApoapsis). At equal per-step accuracy
// the adaptive run needs only the time-average of (r_p / r)^(3/2) as many steps: 0.186 (computed
// over one orbit), i.e. 5.4x fewer. DOPRI5 costs 6 evaluations per step vs RK4's 4, which leaves a
// 5.4 * 4/6 = 3.6x advantage in evaluations from step placement alone; its higher order adds to
// that. So RK4 given 4x DOPRI5's evaluation budget should still be less accurate.
// Measured (scratch reference): DOPRI5 defaults 1172 evaluations, 0.68 m; RK4 with 4688 evaluations
// 26 m; RK4 needs ~11,500 evaluations (~10x) to reach 0.68 m.
TEST(DormandPrince54, NeedsFewerForceEvaluationsThanRk4ForSameAccuracy) {
    const PointMassGravity g;
    const simcore::State s0 = eccentric_periapsis_state();
    const double T = eccentric_period_s();

    DormandPrince54 dp;
    const simcore::State dp_end = dp.propagate(g, s0, s0.t + T);
    const double dp_err = (dp_end.r_eci - s0.r_eci).norm();
    const long dp_evals = dp.stats().force_evals;

    // 4x the budget: RK4 uses 4 evaluations per step, so n = dp_evals steps.
    const double n_steps = static_cast<double>(dp_evals);
    const double h_max = T / (n_steps - 0.5);  // ceil(T / h_max) = n_steps exactly
    const simcore::State rk_end =
        simcore::propagate_fixed(g, s0, s0.t + T, h_max, simcore::FixedStepMethod::kRk4);
    const double rk_err = (rk_end.r_eci - s0.r_eci).norm();

    EXPECT_GT(rk_err, dp_err) << "DOPRI5 " << dp_evals << " evals: " << dp_err << " m; RK4 "
                              << 4 * dp_evals << " evals: " << rk_err << " m";
}

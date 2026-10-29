// M1.2: DenseSegment — continuous extension of one accepted Dormand-Prince step [t0, t0 + h].
//
// Hairer's DOPRI5 interpolant: with theta = (t - t0) / h,
//   y(theta) = y0 + theta (c2 + (1 - theta) (c3 + theta (c4 + (1 - theta) c5)))
// where c2 = y1 - y0, c3, c4 are built from the step's derivative endpoints (Hermite part) and c5
// from all 7 stages. It is a 4th-order polynomial in theta: y(0) = y0, y(1) = y1, y'(0) = f0,
// y'(1) = f1 exactly, and its error inside the step is O(h^5), the same order as the 4th-order
// embedded solution whose error estimate the controller holds at err <= 1. Events (M1.3) root-find
// on this interpolant, so its accuracy sets the event-time accuracy.
//
// Segments are only produced by the integrator (on_step callback); the tests copy them out.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <Eigen/Dense>

#include "orbit_fixtures.hpp"
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
using simcore::State;
using simcore::StepAction;
using namespace orbit_fixtures;

static_assert(std::is_copy_constructible_v<DenseSegment>,
              "callers store segments (event search, plotting) beyond the callback");
static_assert(std::is_copy_assignable_v<DenseSegment>);

// state_at at an endpoint vs the step's own endpoint state. At theta = 0 the interpolant is y0
// plus terms multiplied by 0; at theta = 1 it is y0 + (y1 - y0) (+ terms times 0), which rounds by
// <= ~1.5 ulp per component -> <= sqrt(3) * 1.5 eps |y| = 2.6 eps |y| in norm. 4 eps leaves margin.
// (Epoch rounding of t_end, ~2e-16 s, moves the state by |v| * 2e-16 ~ 2e-12 m, far below
// 4 eps |r| = 6e-9 m.)
constexpr double kEndpointRel = 4.0 * kEps;

struct SegmentRun {
    std::vector<DenseSegment> segs;
    State end;
};

SegmentRun run_collect(const State& s0, const Epoch& t_end, const DpOptions& o = {}) {
    const PointMassGravity g;
    DormandPrince54 dp(o);
    SegmentRun run;
    run.end = dp.propagate(g, s0, t_end, [&](const DenseSegment& seg) {
        run.segs.push_back(seg);
        return StepAction::kContinue;
    });
    return run;
}

// The controller's own error measure (RMS over the 6 components of [r; v] with
// sc_i = atol_i + rtol * max(|a_i|, |b_i|)), applied to the difference of two states.
double scaled_rms(const State& a, const State& b, const DpOptions& o) {
    double sum = 0.0;
    for (int i = 0; i < 3; ++i) {
        const double sc_r =
            o.atol_pos_m + o.rtol * std::max(std::abs(a.r_eci[i]), std::abs(b.r_eci[i]));
        const double sc_v =
            o.atol_vel_mps + o.rtol * std::max(std::abs(a.v_eci[i]), std::abs(b.v_eci[i]));
        const double er = (a.r_eci[i] - b.r_eci[i]) / sc_r;
        const double ev = (a.v_eci[i] - b.v_eci[i]) / sc_v;
        sum += er * er + ev * ev;
    }
    return std::sqrt(sum / 6.0);
}

void expect_states_close(const State& a, const State& b, double rel) {
    EXPECT_LT((a.r_eci - b.r_eci).norm(), rel * b.r_eci.norm());
    EXPECT_LT((a.v_eci - b.v_eci).norm(), rel * b.v_eci.norm());
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------------------------

TEST(DenseSegment, AccessorsAreConsistentAndStateAtCarriesRequestedTime) {
    const State s0 = eccentric_periapsis_state();
    const SegmentRun run = run_collect(s0, s0.t + 0.5 * eccentric_period_s());
    ASSERT_GT(run.segs.size(), 3u);
    for (const DenseSegment& seg : run.segs) {
        // t_end - t_begin is h up to Epoch/double rounding: ulp(|h|) for h itself (h < 1e4 s ->
        // <= 1.8e-12 s) plus ~1e-16 s for the fraction arithmetic; 4 eps max(|h|, 1) covers it.
        EXPECT_NEAR(seg.t_end() - seg.t_begin(), seg.h_s(),
                    4.0 * kEps * std::max(std::abs(seg.h_s()), 1.0));
        const Epoch mid = seg.t_begin() + 0.5 * seg.h_s();
        EXPECT_EQ(seg.state_at(mid).t, mid);
    }
}

// ---------------------------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------------------------

TEST(DenseSegment, EndpointsReproduceStepStates) {
    const PointMassGravity g;
    const State s0 = eccentric_periapsis_state();
    const Epoch t_end = s0.t + eccentric_period_s();
    const SegmentRun run = run_collect(s0, t_end);
    ASSERT_GT(run.segs.size(), 20u);

    // First segment starts at the initial state; last segment ends at the returned state.
    expect_states_close(run.segs.front().state_at(run.segs.front().t_begin()), s0, kEndpointRel);
    expect_states_close(run.segs.back().state_at(run.segs.back().t_end()), run.end, kEndpointRel);

    // An interior step's end state y1 is obtained exactly by stopping the (deterministic) run
    // after that step: propagate returns the step's end state.
    constexpr long kStep = 12;
    DormandPrince54 dp;
    long calls = 0;
    const State y1 = dp.propagate(g, s0, t_end, [&](const DenseSegment&) {
        return ++calls == kStep ? StepAction::kStop : StepAction::kContinue;
    });
    const DenseSegment& seg = run.segs[static_cast<std::size_t>(kStep - 1)];
    ASSERT_EQ(y1.t, seg.t_end());
    expect_states_close(seg.state_at(seg.t_end()), y1, kEndpointRel);
    // ...and the next segment starts from it.
    const DenseSegment& next = run.segs[static_cast<std::size_t>(kStep)];
    expect_states_close(next.state_at(next.t_begin()), y1, kEndpointRel);
}

TEST(DenseSegment, IsContinuousAcrossAdjacentSegments) {
    // Segment k ends and segment k+1 begins at the same accepted state, so both interpolants must
    // give it at the shared time, to endpoint roundoff (twice kEndpointRel: one rounding per side).
    const State s0 = eccentric_periapsis_state();
    const SegmentRun run = run_collect(s0, s0.t + eccentric_period_s());
    ASSERT_GT(run.segs.size(), 20u);
    for (std::size_t k = 0; k + 1 < run.segs.size(); ++k) {
        SCOPED_TRACE(k);
        const DenseSegment& a = run.segs[k];
        const DenseSegment& b = run.segs[k + 1];
        ASSERT_EQ(a.t_end(), b.t_begin());
        expect_states_close(a.state_at(a.t_end()), b.state_at(b.t_begin()), 2.0 * kEndpointRel);
    }
}

// ---------------------------------------------------------------------------------------------
// Interior accuracy
// ---------------------------------------------------------------------------------------------
// Reference: an independent fixed-step RK4 run from the segment's own start state to the same t,
// with h <= 1 s. That isolates the interpolation error from the global error both share.
//   RK4 reference error: local error ~ r (h/tau)^5 with tau = r_p / v_p = 681 s at periapsis:
//   6.8e6 * (1/681)^5 = 4.6e-8 m per step, <= 500 steps (longest segments ~500 s) -> <= 2.3e-5 m,
//   ~1% of the smallest position scale sc_r >= atol_pos = 1e-3 m (velocity similarly ~3% of sc_v);
//   roundoff 500 * eps * r = 7.5e-7 m. So the reference is exact for this purpose.
// Bound: interpolation error <= 3 in the controller's scaled RMS norm. The interpolant's error is
// O(h^5) like the embedded estimate the controller held <= 1 for that step, but with a different,
// problem-dependent O(1) constant, so it may exceed 1 somewhat: in the scratch reference the worst
// case over e = 0 to 0.9 and rtol 1e-8 to 1e-12 was 2.9 (e = 0.9) and 1.35 for this orbit. A
// 3rd-order (cubic Hermite only) interpolant gives ~200 here, so the bound still discriminates.
TEST(DenseSegment, InteriorStateMatchesIndependentFineStepReference) {
    const PointMassGravity g;
    const State s0 = eccentric_periapsis_state();
    const DpOptions o;  // defaults
    const SegmentRun run = run_collect(s0, s0.t + eccentric_period_s(), o);
    ASSERT_GT(run.segs.size(), 20u);

    double worst = 0.0;
    // Every 10th segment covers periapsis, flanks and apoapsis.
    for (std::size_t k = 0; k < run.segs.size(); k += 10) {
        const DenseSegment& seg = run.segs[k];
        const State y0 = seg.state_at(seg.t_begin());
        for (const double theta : {0.25, 0.5, 0.75}) {
            SCOPED_TRACE(::testing::Message() << "segment " << k << " theta " << theta);
            const Epoch t = seg.t_begin() + theta * seg.h_s();
            const State ref =
                simcore::propagate_fixed(g, y0, t, 1.0, simcore::FixedStepMethod::kRk4);
            const double err = scaled_rms(seg.state_at(t), ref, o);
            worst = std::max(worst, err);
            EXPECT_LE(err, 3.0) << "h = " << seg.h_s() << " s";
        }
    }
    // Not vacuous: the comparison does see interpolation error (a test that compared a state with
    // itself would give 0). Roughly 0.3-1.3 expected.
    EXPECT_GT(worst, 1e-3);
}

// ---------------------------------------------------------------------------------------------
// Domain
// ---------------------------------------------------------------------------------------------
// state_at accepts [t_begin, t_end] plus a slack of 1e-9 |h| at each end (so roundoff in a
// computed endpoint time is not an error) and throws std::invalid_argument beyond it. Probes:
// 1e-10 |h| outside (inside the slack, 10x margin) must work; 1e-7 |h| (100x the slack) and
// 1e-2 |h| must throw. |h| ~ 10-500 s, so 1e-10 |h| >= 1e-9 s is resolved by Epoch (ulp 1e-16 s).
TEST(DenseSegment, StateAtThrowsOutsideSegment) {
    const State s0 = circular_initial_state();
    const SegmentRun fwd = run_collect(s0, s0.t + 0.3 * circular_period_s());
    const SegmentRun bwd = run_collect(s0, s0.t - 0.3 * circular_period_s());
    ASSERT_FALSE(fwd.segs.empty());
    ASSERT_FALSE(bwd.segs.empty());

    const DenseSegment* fwd_mid = &fwd.segs[fwd.segs.size() / 2];
    const DenseSegment* bwd_mid = &bwd.segs[bwd.segs.size() / 2];
    for (const DenseSegment* seg : {fwd_mid, bwd_mid}) {
        const double h = seg->h_s();  // negative for the backward segment
        SCOPED_TRACE(h);
        // "Before" the segment is t_begin - fraction * h; "after" is t_end + fraction * h. For
        // h < 0 these are later/earlier in absolute time respectively, as they should be.
        EXPECT_NO_THROW((void)seg->state_at(seg->t_begin()));
        EXPECT_NO_THROW((void)seg->state_at(seg->t_end()));
        EXPECT_NO_THROW((void)seg->state_at(seg->t_begin() - 1e-10 * h));
        EXPECT_NO_THROW((void)seg->state_at(seg->t_end() + 1e-10 * h));
        EXPECT_THROW((void)seg->state_at(seg->t_begin() - 1e-7 * h), std::invalid_argument);
        EXPECT_THROW((void)seg->state_at(seg->t_end() + 1e-7 * h), std::invalid_argument);
        EXPECT_THROW((void)seg->state_at(seg->t_begin() - 1e-2 * h), std::invalid_argument);
        EXPECT_THROW((void)seg->state_at(seg->t_end() + 1e-2 * h), std::invalid_argument);
    }
}

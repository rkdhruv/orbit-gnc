// M5.1: propagate_sampled — one Dormand-Prince run, read out at caller-chosen times through the
// dense output (what the Python bindings hand a notebook: a trajectory at plot times, not at the
// integrator's own steps).
//
// The samples come from the accepted steps' interpolants, so sampling never changes the steps:
// the integration is the same one propagate() makes to the last sample time.

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "orbit_fixtures.hpp"
#include "simcore/dynamics/point_mass_gravity.hpp"
#include "simcore/integrators/dormand_prince.hpp"

namespace {

using simcore::DormandPrince54;
using simcore::DpOptions;
using simcore::Epoch;
using simcore::PointMassGravity;
using simcore::State;
using namespace orbit_fixtures;

DpOptions tight() {
    DpOptions o;
    o.rtol = 1e-12;
    o.atol_pos_m = 1e-6;
    o.atol_vel_mps = 1e-9;
    return o;
}

// Circular 420 km orbit, 10 orbits, 1000 samples at a spacing (T / 100 * 0.999...) that does not
// line up with the steps. Bound: test_dormand_prince's 100-orbit derivation gives an along-track
// drift ds = 0.75 (n t)^2 r (dE/E per radian), quadratic in the swept angle: 15.9 m at 100 orbits
// -> 15.9 / 100 = 0.159 m at 10, and |dv| <= n ds = 1.8e-4 m/s. The interpolant adds at most
// ~3 scaled-RMS units (test_dense_output), 3 sqrt(6) * 7.8e-6 m = 5.7e-5 m: below 0.16 m total.
TEST(PropagateSampled, SamplesFollowTheAnalyticOrbit) {
    const PointMassGravity g;
    DormandPrince54 dp(tight());
    const State s0 = circular_initial_state();
    const double dt = circular_period_s() / 100.0 * 0.9993;
    std::vector<Epoch> times;
    for (int k = 0; k < 1000; ++k) {
        times.push_back(s0.t + k * dt);
    }
    const std::vector<State> out = simcore::propagate_sampled(dp, g, s0, times);
    ASSERT_EQ(out.size(), times.size());
    const double n = circular_mean_motion_radps();
    double worst_r = 0.0;
    double worst_v = 0.0;
    for (std::size_t k = 0; k < out.size(); ++k) {
        EXPECT_EQ(out[k].t, times[k]);
        const State exact = circular_analytic_state(times[k] - s0.t);
        worst_r = std::max(worst_r, (out[k].r_eci - exact.r_eci).norm());
        worst_v = std::max(worst_v, (out[k].v_eci - exact.v_eci).norm());
    }
    std::printf("[sampled] 1000 samples over 10 orbits: worst %.3g m, %.3g m/s\n", worst_r,
                worst_v);
    EXPECT_LT(worst_r, 0.16);
    EXPECT_LT(worst_v, n * 0.16);
}

// The first sample at s0.t is s0 itself (no interpolation), and the last sample is exactly the
// state propagate() returns at that time: same steps, the final one clipped to land on it.
TEST(PropagateSampled, EndpointsAreExact) {
    const PointMassGravity g;
    DormandPrince54 dp(tight());
    const State s0 = eccentric_periapsis_state();
    const Epoch t_end = s0.t + 0.37 * eccentric_period_s();
    const std::vector<State> out =
        simcore::propagate_sampled(dp, g, s0, std::vector<Epoch>{s0.t, s0.t + 1234.5, t_end});
    ASSERT_EQ(out.size(), 3u);
    EXPECT_EQ(out[0].t, s0.t);
    EXPECT_EQ(out[0].r_eci, s0.r_eci);
    EXPECT_EQ(out[0].v_eci, s0.v_eci);
    const long accepted = dp.stats().accepted;
    const State direct = dp.propagate(g, s0, t_end);
    EXPECT_EQ(dp.stats().accepted, accepted);  // sampling did not change the steps
    EXPECT_EQ(out[2].t, direct.t);
    EXPECT_EQ(out[2].r_eci, direct.r_eci);
    EXPECT_EQ(out[2].v_eci, direct.v_eci);
}

// Backward in time: samples before s0.t, in decreasing order. After one period backward from
// periapsis the orbit is at periapsis again (test_dormand_prince's backward case: same bound).
TEST(PropagateSampled, RunsBackwardForDecreasingTimes) {
    const PointMassGravity g;
    DormandPrince54 dp(tight());
    const State s0 = eccentric_periapsis_state();
    const double T = eccentric_period_s();
    const std::vector<State> out = simcore::propagate_sampled(
        dp, g, s0, std::vector<Epoch>{s0.t - 0.25 * T, s0.t - 0.5 * T, s0.t - T});
    ASSERT_EQ(out.size(), 3u);
    const State direct = dp.propagate(g, s0, s0.t - T);
    EXPECT_EQ(out[2].r_eci, direct.r_eci);
    EXPECT_EQ(out[2].v_eci, direct.v_eci);
    // Half a period back is apoapsis: radius r_a. The e = 0.7 one-period error at default options
    // is 0.68 m (M1.2) and scales ~linearly with the tolerance, so ~7 mm at rtol 1e-12; 1 m is a
    // loose bound on that.
    EXPECT_NEAR(out[1].r_eci.norm(), kApoapsisRadius_m, 1.0);
}

// Repeated times are allowed (same state); no times means no integration and no samples.
TEST(PropagateSampled, RepeatsAndEmpty) {
    const PointMassGravity g;
    DormandPrince54 dp(tight());
    const State s0 = circular_initial_state();
    const std::vector<State> out =
        simcore::propagate_sampled(dp, g, s0, std::vector<Epoch>{s0.t + 600.0, s0.t + 600.0});
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0].r_eci, out[1].r_eci);
    EXPECT_TRUE(simcore::propagate_sampled(dp, g, s0, std::vector<Epoch>{}).empty());
}

// Times must run one way from s0.t: non-monotone, or on both sides of s0.t, throws (before any
// integration).
TEST(PropagateSampled, RejectsTimesOutOfOrder) {
    const PointMassGravity g;
    DormandPrince54 dp(tight());
    const State s0 = circular_initial_state();
    EXPECT_THROW(
        (void)simcore::propagate_sampled(dp, g, s0, std::vector<Epoch>{s0.t + 600.0, s0.t + 300.0}),
        std::invalid_argument);
    EXPECT_THROW(
        (void)simcore::propagate_sampled(dp, g, s0, std::vector<Epoch>{s0.t + 600.0, s0.t - 300.0}),
        std::invalid_argument);
    EXPECT_THROW(
        (void)simcore::propagate_sampled(dp, g, s0, std::vector<Epoch>{s0.t - 600.0, s0.t + 300.0}),
        std::invalid_argument);
}

}  // namespace

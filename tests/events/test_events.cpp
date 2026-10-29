// M1.3: event detection on Dormand-Prince dense output.
//
// An event is a sign change of a scalar g(State) along the trajectory. After each accepted step
// the driver compares g at the segment's two ends; on a sign change (in the requested direction,
// judged in physical time) it finds the root with Brent on g(segment.state_at(t)). The dense
// output is 4th order and tied to the integrator tolerance, so no extra force evaluations.
//
// References come from Kepler's equation in the forward direction (closed form, no iteration):
//   true anomaly nu -> eccentric anomaly E = 2 atan2(sqrt(1-e) sin(nu/2), sqrt(1+e) cos(nu/2))
//   -> mean anomaly M = E - e sin E -> time since periapsis t = M / n,  n = sqrt(mu / a^3).
// Orbit: the M1.2 fixture e = 0.7, periapsis 420 km altitude, inclined 51.6 deg:
//   a = 22,660,456.67 m, p = a (1 - e^2) = 11,556,832.9 m, T = 33,947.9823 s.
//
// The milestone requirement is detected periapsis time vs Kepler < 1 ms, and that is the
// tolerance used for every event time below. Expected size of the error: with rtol 1e-12 the
// along-track position error is ~1e-2 m per orbit (M1.2 measured 0.68 m at rtol 1e-10, error ~
// tol), i.e. ~1e-6 s at ~1e4 m/s; Brent's time tolerance is 1e-6 s. So 1 ms has > 100x margin
// and still catches a wrong root (a step-sized error is >= 1 s) or a wrong direction (~T/2).

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../integrators/orbit_fixtures.hpp"
#include "simcore/dynamics/point_mass_gravity.hpp"
#include "simcore/events/events.hpp"
#include "simcore/integrators/dormand_prince.hpp"
#include "simcore/orbits/kepler.hpp"

namespace {

using namespace orbit_fixtures;
using simcore::DormandPrince54;
using simcore::DpOptions;
using simcore::EventDirection;
using simcore::EventRun;
using simcore::EventSpec;
using simcore::PointMassGravity;

constexpr double kTimeTol_s = 1e-3;  // milestone requirement (see header)

double semi_latus_rectum_m() {
    return kEccSemiMajorAxis_m * (1.0 - kEcc * kEcc);
}

// Perifocal-frame state of the e = 0.7 orbit at true anomaly nu, at start_epoch():
//   r = p / (1 + e cos nu) (cos nu P + sin nu Q),  v = sqrt(mu / p) (-sin nu P + (e + cos nu) Q)
State eccentric_state_at(double nu) {
    const double p = semi_latus_rectum_m();
    const double r = p / (1.0 + kEcc * std::cos(nu));
    const double vk = std::sqrt(earth::mu / p);
    return State{start_epoch(), r * (std::cos(nu) * p_hat() + std::sin(nu) * q_hat()),
                 vk * (-std::sin(nu) * p_hat() + (kEcc + std::cos(nu)) * q_hat())};
}

// Time since the last periapsis at true anomaly nu, in [0, T).
double time_since_periapsis_s(double nu) {
    double E = 2.0 * std::atan2(std::sqrt(1.0 - kEcc) * std::sin(0.5 * nu),
                                std::sqrt(1.0 + kEcc) * std::cos(0.5 * nu));
    if (E < 0.0) {
        E += 2.0 * kPi;
    }
    const double n = 2.0 * kPi / eccentric_period_s();
    return (E - kEcc * std::sin(E)) / n;
}

// True anomaly at which the orbit crosses radius rc on the way out (the inbound crossing is at
// -nu_c): from r = p / (1 + e cos nu), cos nu_c = (p / rc - 1) / e.
double crossing_true_anomaly(double rc_m) {
    return std::acos((semi_latus_rectum_m() / rc_m - 1.0) / kEcc);
}

// Tight integrator settings (M1.2's 100-orbit configuration).
DpOptions tight_options() {
    DpOptions o;
    o.rtol = 1e-12;
    o.atol_pos_m = 1e-6;
    o.atol_vel_mps = 1e-9;
    return o;
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Built-in event functions
// ---------------------------------------------------------------------------------------------

// Periapsis: g = r . v = r dr/dt, which goes from - (falling in) to + (climbing out) at
// periapsis. Apoapsis is the opposite crossing, so direction must be kRising.
TEST(EventFunctions, PeriapsisIsRisingRadialRate) {
    const EventSpec e = simcore::periapsis_event();
    EXPECT_EQ(e.direction, EventDirection::kRising);
    EXPECT_FALSE(e.terminal);
    EXPECT_FALSE(e.name.empty());
    const State s{start_epoch(), Eigen::Vector3d(7e6, 1e6, -2e6),
                  Eigen::Vector3d(-100.0, 7e3, 5.0)};
    EXPECT_EQ(e.g(s), s.r_eci.dot(s.v_eci));
}

// Impact: g = |r| - R, falling through zero, and terminal (the trajectory ends at the surface).
TEST(EventFunctions, ImpactIsFallingAndTerminal) {
    const EventSpec e = simcore::impact_event();
    EXPECT_EQ(e.direction, EventDirection::kFalling);
    EXPECT_TRUE(e.terminal);
    const State s{start_epoch(), Eigen::Vector3d(3e6, 4e6, 12e6), Eigen::Vector3d::Zero()};
    EXPECT_EQ(e.g(s), 13e6 - earth::radius);  // |(3, 4, 12)| = 13 exactly
    const EventSpec moon = simcore::impact_event(1.7374e6);
    EXPECT_EQ(moon.g(s), 13e6 - 1.7374e6);
}

// Altitude crossing: g = |r| - (R + h); direction as requested, not terminal by default.
TEST(EventFunctions, AltitudeCrossingUsesRequestedDirection) {
    const State s{start_epoch(), Eigen::Vector3d(3e6, 4e6, 12e6), Eigen::Vector3d::Zero()};
    for (const EventDirection d :
         {EventDirection::kAny, EventDirection::kRising, EventDirection::kFalling}) {
        const EventSpec e = simcore::altitude_crossing_event(1.0e6, d);
        EXPECT_EQ(e.direction, d);
        EXPECT_FALSE(e.terminal);
        EXPECT_EQ(e.g(s), 13e6 - (earth::radius + 1.0e6));
    }
}

// ---------------------------------------------------------------------------------------------
// Periapsis detection vs Kepler (the milestone claim)
// ---------------------------------------------------------------------------------------------

// Start at apoapsis: periapses at T/2 + k T. Ten orbits -> hits at 0.5 T ... 9.5 T.
// Also checks the root itself: Brent zeroes g on the dense output to xtol = 1e-6 s, so
// |g(hit)| <= |dg/dt| * 1e-6 s. At periapsis dg/dt = d(r.v)/dt = v^2 - mu/r = mu e / r_p
// = 4.104e7 m^2/s^2, giving |r.v| <= 41 m^2/s (relative to |r||v| = 6.8e10: 6e-10).
TEST(Events, PeriapsisTimesMatchKeplerToUnderOneMillisecondOverTenOrbits) {
    PointMassGravity g;
    DormandPrince54 dp(tight_options());
    const State s0 = eccentric_apoapsis_state();
    const double T = eccentric_period_s();
    const EventRun run =
        simcore::propagate_with_events(dp, g, s0, s0.t + 10.0 * T, {simcore::periapsis_event()});

    EXPECT_FALSE(run.terminated);
    EXPECT_EQ(run.final_state.t, s0.t + 10.0 * T);
    ASSERT_EQ(run.hits.size(), 10u);
    const double dgdt = earth::mu * kEcc / kPeriapsisRadius_m;
    for (std::size_t k = 0; k < run.hits.size(); ++k) {
        SCOPED_TRACE(k);
        const double expected_s = (0.5 + static_cast<double>(k)) * T;
        EXPECT_EQ(run.hits[k].event_index, 0u);
        EXPECT_NEAR(run.hits[k].state.t - s0.t, expected_s, kTimeTol_s);
        const State& h = run.hits[k].state;
        EXPECT_LE(std::abs(h.r_eci.dot(h.v_eci)), dgdt * 1e-6);
    }
}

// Start at true anomaly 2 rad (past periapsis): time since periapsis 2794.988 s, so the next
// periapsis is at T - 2794.988 = 31152.994 s and the one after at 65100.976 s.
TEST(Events, PeriapsisFromArbitraryTrueAnomalyMatchesKepler) {
    PointMassGravity g;
    DormandPrince54 dp(tight_options());
    const State s0 = eccentric_state_at(2.0);
    const double T = eccentric_period_s();
    const double first_s = T - time_since_periapsis_s(2.0);
    EXPECT_NEAR(first_s, 31152.994, 1e-3);  // hand derivation, rounded to 1 ms

    const EventRun run =
        simcore::propagate_with_events(dp, g, s0, s0.t + 2.0 * T, {simcore::periapsis_event()});
    ASSERT_EQ(run.hits.size(), 2u);
    EXPECT_NEAR(run.hits[0].state.t - s0.t, first_s, kTimeTol_s);
    EXPECT_NEAR(run.hits[1].state.t - s0.t, first_s + T, kTimeTol_s);
}

// Starting exactly on the event (g = 0 at t0) is not an event: the first periapsis reported
// when starting at periapsis is the next one, at T.
TEST(Events, EventExactlyAtStartIsNotReported) {
    PointMassGravity g;
    DormandPrince54 dp(tight_options());
    const State s0 = eccentric_periapsis_state();
    const double T = eccentric_period_s();
    const EventRun run =
        simcore::propagate_with_events(dp, g, s0, s0.t + 1.5 * T, {simcore::periapsis_event()});
    ASSERT_EQ(run.hits.size(), 1u);
    EXPECT_NEAR(run.hits[0].state.t - s0.t, T, kTimeTol_s);
}

// The start rule only discards a root that is the initial condition to roundoff:
// |g(s0)| <= 1e-9 s * |dg/dt|. Here g(s0) = r . v = -6.8e-6 m^2/s (v nudged by -1e-12 m/s
// radially at periapsis), i.e. the root is 6.8e-6 / 4.1e7 = 1.7e-13 s after s0: not an event.
// Unlike EventExactlyAtStartIsNotReported (g(s0) = 0, which crosses() already rejects), this one
// has a real sign change, so it fails if the start rule is removed.
TEST(Events, RoundoffLevelRootAtStartIsNotReported) {
    PointMassGravity g;
    DormandPrince54 dp(tight_options());
    State s0 = eccentric_periapsis_state();
    s0.v_eci -= 1e-12 * s0.r_eci.normalized();
    ASSERT_LT(s0.r_eci.dot(s0.v_eci), 0.0);
    const double T = eccentric_period_s();
    const EventRun run =
        simcore::propagate_with_events(dp, g, s0, s0.t + 1.5 * T, {simcore::periapsis_event()});
    ASSERT_EQ(run.hits.size(), 1u);
    EXPECT_NEAR(run.hits[0].state.t - s0.t, T, kTimeTol_s);
}

// A real event shortly after the start is reported: starting 0.5 us before periapsis
// (g(s0) = -0.5e-6 s * 4.1e7 m^2/s^2 = -20.5 m^2/s, far above roundoff), the periapsis must be
// found at +0.5 us. Time bound: Brent's 1e-6 s tolerance + 0.5 us offset -> 2e-6 s.
TEST(Events, EventHalfAMicrosecondAfterStartIsReported) {
    PointMassGravity g;
    DormandPrince54 dp(tight_options());
    const State peri = eccentric_periapsis_state();
    const State s0 = simcore::kepler_propagate(peri, peri.t - 0.5e-6);
    ASSERT_LT(s0.r_eci.dot(s0.v_eci), -10.0);
    const EventRun run =
        simcore::propagate_with_events(dp, g, s0, s0.t + 100.0, {simcore::periapsis_event()});
    ASSERT_EQ(run.hits.size(), 1u);
    EXPECT_NEAR(run.hits[0].state.t - peri.t, 0.0, 2e-6);
}

// A terminal event is never discarded, however close to the start. 1 mm above the surface falling
// at 2 km/s: impact at +0.5 us. Exactly on the surface, (R, 0, 0) so |r| = R and g(s0) = 0,
// falling: the run must end at s0. Either way it stops at the surface instead of flying through the
// Earth.
TEST(Events, TerminalEventRightAfterStartIsNeverDropped) {
    PointMassGravity g;
    const Eigen::Vector3d x = Eigen::Vector3d::UnitX();
    for (const double height_m : {1e-3, 0.0}) {
        SCOPED_TRACE(height_m);
        DormandPrince54 dp(tight_options());
        const State s0{start_epoch(), (earth::radius + height_m) * x, -2000.0 * x};
        const EventRun run =
            simcore::propagate_with_events(dp, g, s0, s0.t + 100.0, {simcore::impact_event()});
        EXPECT_TRUE(run.terminated);
        ASSERT_EQ(run.hits.size(), 1u);
        EXPECT_LT(run.final_state.t - s0.t, 2e-6);
        // |r| = R to |dr/dt| * 1e-6 s = 2 mm (bound 1 cm, as in the free-fall test).
        EXPECT_NEAR(run.final_state.r_eci.norm(), earth::radius, 1e-2);
    }
}

// The direction filter still applies at the start: on the surface and climbing at 2 km/s is not
// an impact (g goes 0 -> +), so the run continues to t_end with no hit.
TEST(Events, StartingOnSurfaceAndClimbingIsNotAnImpact) {
    PointMassGravity g;
    DormandPrince54 dp(tight_options());
    const Eigen::Vector3d x = Eigen::Vector3d::UnitX();
    const State s0{start_epoch(), earth::radius * x, 2000.0 * x};
    const EventRun run =
        simcore::propagate_with_events(dp, g, s0, s0.t + 100.0, {simcore::impact_event()});
    EXPECT_FALSE(run.terminated);
    EXPECT_TRUE(run.hits.empty());
    EXPECT_EQ(run.final_state.t, s0.t + 100.0);
}

// A root exactly on a step boundary is reported once, at the boundary. We record the segments of
// a run and put the event g = t - t_b exactly on the end of a mid-run segment. In a second,
// identical run (deterministic steps) g is exactly 0 at that boundary: the segment ending there
// sees a rising crossing whose root is its end point, and the next segment starts at g = 0, which
// crosses() does not count again. One hit, exactly at t_b.
TEST(Events, RootExactlyOnStepBoundaryIsReportedOnceAtTheBoundary) {
    PointMassGravity g;
    const State s0 = eccentric_apoapsis_state();
    const simcore::Epoch t_end = s0.t + eccentric_period_s();
    // Non-terminal events don't change the steps, so plain propagate records the same segments.
    std::vector<simcore::DenseSegment> segs;
    DormandPrince54 rec(tight_options());
    (void)rec.propagate(g, s0, t_end, [&segs](const simcore::DenseSegment& seg) {
        segs.push_back(seg);
        return simcore::StepAction::kContinue;
    });
    ASSERT_GT(segs.size(), 10u);
    const simcore::Epoch t_b = segs[segs.size() / 2].t_end();

    DormandPrince54 dp(tight_options());
    const EventRun run = simcore::propagate_with_events(
        dp, g, s0, t_end,
        {EventSpec{"clock", [t_b](const State& s) { return s.t - t_b; }, EventDirection::kRising,
                   false}});
    ASSERT_EQ(run.hits.size(), 1u);
    EXPECT_EQ(run.hits[0].state.t, t_b);
}

// The per-segment root locator, including the guard for a sign change seen on the cached
// boundary values but not on fresh evaluations (a root on the boundary to roundoff: it happens at
// ~2 of 600 boundaries, too rarely to force end to end). Offsets dt are from the segment start.
TEST(EventRootLocator, BracketedRootUsesBrent) {
    const double dt =
        simcore::detail::locate_root_in_segment([](double t) { return t - 3.25; }, 10.0, 1e-9);
    EXPECT_NEAR(dt, 3.25, 1e-9);
    // Backward segment (h < 0): the root at -3.25 is found too.
    EXPECT_NEAR(
        simcore::detail::locate_root_in_segment([](double t) { return t + 3.25; }, -10.0, 1e-9),
        -3.25, 1e-9);
}

TEST(EventRootLocator, ExactZeroAtAnEndIsThatEnd) {
    EXPECT_EQ(
        simcore::detail::locate_root_in_segment([](double t) { return t - 10.0; }, 10.0, 1e-9),
        10.0);
    EXPECT_EQ(simcore::detail::locate_root_in_segment([](double t) { return t; }, 10.0, 1e-9), 0.0);
}

// No sign change on fresh values: the root is on a boundary to roundoff, so take the end whose
// |g| is smaller (here g = 1e-20 at the start vs 5 at the end -> start; and the mirror case).
TEST(EventRootLocator, SameSignAtBothEndsTakesTheEndNearerZero) {
    EXPECT_EQ(simcore::detail::locate_root_in_segment([](double t) { return 1e-20 + 0.5 * t; },
                                                      10.0, 1e-9),
              0.0);
    EXPECT_EQ(simcore::detail::locate_root_in_segment(
                  [](double t) { return -5.0 + 0.5 * t - 1e-20; }, 10.0, 1e-9),
              10.0);
}

// Backward in time from apoapsis: the previous periapsis is at -T/2. Direction is judged in
// physical time, so it is still the kRising crossing of r . v (not the apoapsis at -T).
TEST(Events, BackwardPropagationFindsPreviousPeriapsis) {
    PointMassGravity g;
    DormandPrince54 dp(tight_options());
    const State s0 = eccentric_apoapsis_state();
    const double T = eccentric_period_s();
    const EventRun run =
        simcore::propagate_with_events(dp, g, s0, s0.t - 0.9 * T, {simcore::periapsis_event()});
    ASSERT_EQ(run.hits.size(), 1u);
    EXPECT_NEAR(run.hits[0].state.t - s0.t, -0.5 * T, kTimeTol_s);
    EXPECT_EQ(run.final_state.t, s0.t - 0.9 * T);
}

// ---------------------------------------------------------------------------------------------
// Altitude crossings and directions
// ---------------------------------------------------------------------------------------------

// 1000 km altitude: rc = 7,378,137 m, nu_c = acos((p/rc - 1)/e) = 0.628197 rad, so the outbound
// crossing is t_c = 452.211 s after periapsis and the inbound one T - t_c = 33495.771 s.
// From periapsis over 2 T: rising at t_c, T + t_c; falling at T - t_c, 2T - t_c.
TEST(Events, AltitudeCrossingTimesAndDirectionsMatchKepler) {
    PointMassGravity g;
    const State s0 = eccentric_periapsis_state();
    const double T = eccentric_period_s();
    const double rc = earth::radius + 1.0e6;
    const double tc = time_since_periapsis_s(crossing_true_anomaly(rc));
    EXPECT_NEAR(tc, 452.211, 1e-3);  // hand derivation, rounded to 1 ms

    struct Case {
        EventDirection dir;
        std::vector<double> times_s;
    };
    const std::vector<Case> cases{
        {EventDirection::kRising, {tc, T + tc}},
        {EventDirection::kFalling, {T - tc, 2.0 * T - tc}},
        {EventDirection::kAny, {tc, T - tc, T + tc, 2.0 * T - tc}},  // time order
    };
    for (const Case& c : cases) {
        SCOPED_TRACE(static_cast<int>(c.dir));
        DormandPrince54 dp(tight_options());
        const EventRun run = simcore::propagate_with_events(
            dp, g, s0, s0.t + 2.0 * T, {simcore::altitude_crossing_event(1.0e6, c.dir)});
        ASSERT_EQ(run.hits.size(), c.times_s.size());
        for (std::size_t k = 0; k < c.times_s.size(); ++k) {
            SCOPED_TRACE(k);
            EXPECT_NEAR(run.hits[k].state.t - s0.t, c.times_s[k], kTimeTol_s);
            // On the root, |r| = rc to |dr/dt| * 1e-6 s; |dr/dt| <= v_p = 9984 m/s -> 1 cm.
            EXPECT_NEAR(run.hits[k].state.r_eci.norm(), rc, 1e-2);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Impact (terminal) vs radial free fall
// ---------------------------------------------------------------------------------------------

// Dropped from rest at r0 = R + 1000 km. Radial Kepler motion (e = 1, a = r0/2) gives the time
// to fall from r0 to r:  t = sqrt(r0^3 / (2 mu)) (acos(sqrt(x)) + sqrt(x (1 - x))),  x = r / r0.
// For r = R: t = 510.56993 s, impact speed sqrt(2 mu (1/R - 1/r0)) = 4115.9 m/s.
// Final |r| = R to |dr/dt| * 1e-6 s = 4.1 mm; bound 1 cm.
TEST(Events, ImpactTimeMatchesRadialFreeFallAndStopsIntegration) {
    PointMassGravity g;
    DormandPrince54 dp(tight_options());
    const double r0 = earth::radius + 1.0e6;
    const State s0{start_epoch(), r0 * p_hat(), Eigen::Vector3d::Zero()};
    const double x = earth::radius / r0;
    const double t_fall = std::sqrt(r0 * r0 * r0 / (2.0 * earth::mu)) *
                          (std::acos(std::sqrt(x)) + std::sqrt(x * (1.0 - x)));
    EXPECT_NEAR(t_fall, 510.56993, 1e-5);  // hand derivation

    const EventRun run =
        simcore::propagate_with_events(dp, g, s0, s0.t + 1000.0, {simcore::impact_event()});
    EXPECT_TRUE(run.terminated);
    ASSERT_EQ(run.hits.size(), 1u);
    EXPECT_NEAR(run.hits[0].state.t - s0.t, t_fall, kTimeTol_s);
    EXPECT_EQ(run.final_state.t, run.hits[0].state.t);
    EXPECT_EQ(run.final_state.r_eci, run.hits[0].state.r_eci);
    EXPECT_NEAR(run.final_state.r_eci.norm(), earth::radius, 1e-2);
}

// ---------------------------------------------------------------------------------------------
// Several events, terminal ordering, bookkeeping
// ---------------------------------------------------------------------------------------------

// From apoapsis over one orbit, events {periapsis, 1000 km kAny}: inbound crossing at
// T/2 - tc = 16521.780 s, periapsis 16973.991 s, outbound 17426.202 s. Hits must come back in
// time order with the right event_index.
TEST(Events, MultipleEventsAreReportedInTimeOrder) {
    PointMassGravity g;
    DormandPrince54 dp(tight_options());
    const State s0 = eccentric_apoapsis_state();
    const double T = eccentric_period_s();
    const double tc = time_since_periapsis_s(crossing_true_anomaly(earth::radius + 1.0e6));
    const EventRun run = simcore::propagate_with_events(
        dp, g, s0, s0.t + T, {simcore::periapsis_event(), simcore::altitude_crossing_event(1.0e6)});
    ASSERT_EQ(run.hits.size(), 3u);
    EXPECT_EQ(run.hits[0].event_index, 1u);
    EXPECT_EQ(run.hits[1].event_index, 0u);
    EXPECT_EQ(run.hits[2].event_index, 1u);
    EXPECT_NEAR(run.hits[0].state.t - s0.t, 0.5 * T - tc, kTimeTol_s);
    EXPECT_NEAR(run.hits[1].state.t - s0.t, 0.5 * T, kTimeTol_s);
    EXPECT_NEAR(run.hits[2].state.t - s0.t, 0.5 * T + tc, kTimeTol_s);
}

// A terminal event ends the run at its root; events after it (the periapsis 452 s later) are
// not reported, and final_state is the state at the terminal root.
TEST(Events, TerminalEventStopsRunAndLaterEventsAreNotReported) {
    PointMassGravity g;
    DormandPrince54 dp(tight_options());
    const State s0 = eccentric_apoapsis_state();
    const double T = eccentric_period_s();
    const double tc = time_since_periapsis_s(crossing_true_anomaly(earth::radius + 1.0e6));
    EventSpec reentry = simcore::altitude_crossing_event(1.0e6, EventDirection::kFalling);
    reentry.terminal = true;
    const EventRun run =
        simcore::propagate_with_events(dp, g, s0, s0.t + T, {simcore::periapsis_event(), reentry});
    EXPECT_TRUE(run.terminated);
    ASSERT_EQ(run.hits.size(), 1u);
    EXPECT_EQ(run.hits[0].event_index, 1u);
    EXPECT_NEAR(run.hits[0].state.t - s0.t, 0.5 * T - tc, kTimeTol_s);
    EXPECT_EQ(run.final_state.t, run.hits[0].state.t);
}

// No events: behaves exactly like DormandPrince54::propagate (same final state, bit for bit).
TEST(Events, NoEventsMatchesPlainPropagation) {
    PointMassGravity g;
    const State s0 = eccentric_apoapsis_state();
    const simcore::Epoch t_end = s0.t + eccentric_period_s();
    DormandPrince54 a(tight_options());
    DormandPrince54 b(tight_options());
    const EventRun run = simcore::propagate_with_events(a, g, s0, t_end, {});
    const State plain = b.propagate(g, s0, t_end);
    EXPECT_FALSE(run.terminated);
    EXPECT_TRUE(run.hits.empty());
    EXPECT_EQ(run.final_state.t, plain.t);
    EXPECT_EQ(run.final_state.r_eci, plain.r_eci);
    EXPECT_EQ(run.final_state.v_eci, plain.v_eci);
}

// Two identical runs give bit-identical hits (determinism requirement).
TEST(Events, IsDeterministic) {
    PointMassGravity g;
    const State s0 = eccentric_state_at(2.0);
    const simcore::Epoch t_end = s0.t + 2.0 * eccentric_period_s();
    const std::vector<EventSpec> ev{simcore::periapsis_event(),
                                    simcore::altitude_crossing_event(1.0e6)};
    DormandPrince54 a(tight_options());
    DormandPrince54 b(tight_options());
    const EventRun ra = simcore::propagate_with_events(a, g, s0, t_end, ev);
    const EventRun rb = simcore::propagate_with_events(b, g, s0, t_end, ev);
    ASSERT_EQ(ra.hits.size(), rb.hits.size());
    for (std::size_t k = 0; k < ra.hits.size(); ++k) {
        EXPECT_EQ(ra.hits[k].event_index, rb.hits[k].event_index);
        EXPECT_EQ(ra.hits[k].state.t, rb.hits[k].state.t);
        EXPECT_EQ(ra.hits[k].state.r_eci, rb.hits[k].state.r_eci);
    }
}

TEST(Events, RejectsInvalidArguments) {
    PointMassGravity g;
    DormandPrince54 dp;
    const State s0 = eccentric_apoapsis_state();
    const simcore::Epoch t_end = s0.t + 100.0;
    const std::vector<EventSpec> ok{simcore::periapsis_event()};
    for (const double tol : {0.0, -1e-6, std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::infinity()}) {
        SCOPED_TRACE(tol);
        EXPECT_THROW((void)simcore::propagate_with_events(dp, g, s0, t_end, ok, tol),
                     std::invalid_argument);
    }
    EventSpec empty;  // no g
    EXPECT_THROW((void)simcore::propagate_with_events(dp, g, s0, t_end, {empty}),
                 std::invalid_argument);
}

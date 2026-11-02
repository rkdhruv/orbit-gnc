// M3.5: deterministic replay.
//
// The state after tick n depends only on the initial state and the command in effect at each
// tick (M3.1: not on frame rate or chunking; M3.4: on-rails decisions use only the command and the
// state). So a session is reproduced exactly by its initial state plus an input log: each command
// with the tick at whose start it took effect. Simulation records an entry only when the command
// in effect changes (the front end submits every frame, mostly the same command), and replay()
// re-runs the log on a fresh Simulation. On the same machine and build the result is bit-identical
// (-ffp-contract=off; across architectures Eigen's FMA kernels may differ in the last bit).

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "../integrators/orbit_fixtures.hpp"
#include "simcore/constants.hpp"
#include "simcore/dynamics/point_mass_gravity.hpp"
#include "simcore/sim/manual_control.hpp"
#include "simcore/sim/replay.hpp"
#include "simcore/sim/simulation.hpp"
#include "simcore/sim/tick_scheduler.hpp"

namespace {

using simcore::LoggedCommand;
using simcore::ManualInput;
using simcore::SimConfig;
using simcore::Simulation;
using simcore::Snapshot;
using simcore::VehicleCommand;
using simcore::VehicleState;

// 400 kg dry + 400 kg tank, main engine 1000 N / Isp 300 s, three wheels on the body axes;
// two-body gravity with rails enabled, so a session mixes integrated and on-rails ticks.
SimConfig rails_config() {
    return SimConfig{
        simcore::VehicleConfig{
            simcore::MassProperties(400.0, Eigen::Vector3d(300.0, 400.0, 450.0).asDiagonal(),
                                    simcore::PropellantTank{400.0, Eigen::Matrix3d::Zero()}),
            {simcore::Thruster(1000.0, 300.0, Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitX())},
            {simcore::ReactionWheel(Eigen::Vector3d::UnitX(), 0.05, 0.1, 600.0),
             simcore::ReactionWheel(Eigen::Vector3d::UnitY(), 0.05, 0.1, 600.0),
             simcore::ReactionWheel(Eigen::Vector3d::UnitZ(), 0.05, 0.1, 600.0)}},
        std::make_shared<simcore::PointMassGravity>(),
        std::make_shared<simcore::ConstantBodyTorque>(Eigen::Vector3d::Zero()), 0.02,
        simcore::earth::mu};
}

VehicleState initial() {
    const simcore::State o = orbit_fixtures::circular_initial_state();
    return VehicleState{
        o.t,   o.r_eci,        o.v_eci, Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero(),
        300.0, {0.0, 0.0, 0.0}};
}

void expect_identical(const Snapshot& a, const Snapshot& b) {
    EXPECT_EQ(a.tick, b.tick);
    EXPECT_EQ(a.state.t, b.state.t);
    EXPECT_EQ(a.state.r_eci, b.state.r_eci);
    EXPECT_EQ(a.state.v_eci, b.state.v_eci);
    EXPECT_EQ(a.state.q_body_to_eci.coeffs(), b.state.q_body_to_eci.coeffs());
    EXPECT_EQ(a.state.w_body, b.state.w_body);
    EXPECT_EQ(a.state.propellant_kg, b.state.propellant_kg);
    EXPECT_EQ(a.state.wheel_h_nms, b.state.wheel_h_nms);
    EXPECT_EQ(a.command, b.command);
    EXPECT_EQ(a.on_rails, b.on_rails);
}

// A scripted "pilot" session driven like the front end: irregular frame times, a command
// submitted every frame from manual_command (throttle bursts, a pitch manoeuvre, rate damping),
// and a stretch of 10,000x warp on rails once the vehicle has settled.
struct Session {
    Simulation sim{rails_config(), initial()};
    std::vector<Snapshot> marks;  // snapshots at a few points, to replay to
};

Session fly_session() {
    Session s;
    simcore::TickScheduler sched(0.02, 0.25, 1'000'000'000);
    constexpr std::array<double, 4> kFrames{1.0 / 60.0, 1.0 / 30.0, 1.0 / 144.0, 0.021};
    const simcore::VehicleConfig& v = s.sim.config().vehicle;
    for (int f = 0; f < 3000; ++f) {
        const double t = s.sim.snapshot().state.t - initial().t;
        ManualInput in;
        in.rate_damping = true;
        in.throttle = (t > 5.0 && t < 25.0) ? 0.8 : 0.0;
        in.rotate =
            (t > 30.0 && t < 33.0) ? Eigen::Vector3d(0.0, 1.0, 0.0) : Eigen::Vector3d::Zero();
        s.sim.submit(simcore::manual_command(v, in, s.sim.snapshot().state));
        const double warp = (f > 2500) ? 1.0e4 : 1.0;  // settled by then: on rails
        s.sim.step(sched.ticks_for_frame(kFrames.at(static_cast<std::size_t>(f) % 4), warp).ticks);
        if (f == 1000 || f == 2000) {
            s.marks.push_back(s.sim.snapshot());
        }
    }
    s.marks.push_back(s.sim.snapshot());
    return s;
}

}  // namespace

// The log holds only changes of the command in effect, at the tick they took effect. The initial
// all-zero command is part of the starting conditions, so re-submitting it logs nothing; repeats
// log nothing; of several commands submitted before one tick only the last (the one that took
// effect) is logged; a command still pending when the log is read isn't in it.
TEST(Replay, LogRecordsChangesAtTheTickTheyTookEffect) {
    const VehicleCommand zero{{0.0}, {0.0, 0.0, 0.0}};
    const VehicleCommand a{{0.5}, {0.0, 0.0, 0.0}};
    const VehicleCommand b{{0.0}, {0.0, 0.01, 0.0}};
    Simulation sim(rails_config(), initial());
    sim.submit(zero);
    sim.step(3);
    sim.submit(a);
    sim.step(5);  // a takes effect at tick 3
    sim.submit(a);
    sim.step(2);  // repeat: nothing logged
    sim.submit(zero);
    sim.submit(b);
    sim.step(1);    // b (the last) takes effect at tick 10
    sim.submit(a);  // still pending
    const std::vector<LoggedCommand> log = sim.input_log();
    ASSERT_EQ(log.size(), 2u);
    EXPECT_EQ(log[0].tick, 3);
    EXPECT_EQ(log[0].command, a);
    EXPECT_EQ(log[1].tick, 10);
    EXPECT_EQ(log[1].command, b);
}

// The M3 done-when check: a flown session (3000 irregular frames, burns, a manoeuvre, rate
// damping, then 10,000x warp on rails) replays bit for bit from its initial state and log, at the
// end and at intermediate ticks.
TEST(Replay, FlownSessionReplaysBitForBit) {
    const Session s = fly_session();
    const std::vector<LoggedCommand> log = s.sim.input_log();
    EXPECT_GT(log.size(), 10u);  // the session really changed command many times
    EXPECT_TRUE(s.marks.back().on_rails);
    EXPECT_LT(s.marks.back().state.propellant_kg, 300.0);  // and really burned
    for (const Snapshot& mark : s.marks) {
        SCOPED_TRACE(mark.tick);
        expect_identical(simcore::replay(rails_config(), initial(), log, mark.tick), mark);
    }
}

// The comparison is sensitive: starting the first burn one tick (20 ms) late shortens it by one
// tick, dv by (T u / m) h = (1000 * 0.8 / 700) * 0.02 = 0.023 m/s (700 kg = 400 dry + 300
// propellant), which the warp turns into kilometres of position: bound > 1 m. (A one-ulp throttle
// change is not a reliable probe: it changes each tick's velocity by ~3e-18 m/s, far below the ulp
// of a velocity component (~1.4e-14 m/s at 100 m/s), so it flips a bit only rarely, about 2e-4 of
// ticks; over a session it may or may not change the result.)
TEST(Replay, ACommandOneTickLateChangesTheResult) {
    const Session s = fly_session();
    std::vector<LoggedCommand> log = s.sim.input_log();
    bool delayed = false;
    for (std::size_t i = 0; i + 1 < log.size(); ++i) {
        if (log[i].command.throttle[0] > 0.0 && log[i + 1].tick > log[i].tick + 1) {
            ++log[i].tick;
            delayed = true;
            break;
        }
    }
    ASSERT_TRUE(delayed);
    const Snapshot end = simcore::replay(rails_config(), initial(), log, s.marks.back().tick);
    EXPECT_GT((end.state.r_eci - s.marks.back().state.r_eci).norm(), 1.0);
}

// Entries must be in strictly increasing tick order at ticks >= 0, and commands are validated as
// by Simulation::submit. Entries at or after to_tick haven't taken effect yet, so they are ignored
// (that is how a log is replayed to an intermediate tick).
TEST(Replay, RejectsMalformedLogs) {
    const VehicleCommand a{{0.5}, {0.0, 0.0, 0.0}};
    const auto run = [](const std::vector<LoggedCommand>& log) {
        return simcore::replay(rails_config(), initial(), log, 100);
    };
    EXPECT_THROW((void)run({{5, a}, {5, a}}), std::invalid_argument);  // not increasing
    EXPECT_THROW((void)run({{7, a}, {3, a}}), std::invalid_argument);  // out of order
    EXPECT_THROW((void)run({{-1, a}}), std::invalid_argument);         // before the start
    EXPECT_THROW((void)run({{5, VehicleCommand{{2.0}, {0.0, 0.0, 0.0}}}}),
                 std::invalid_argument);  // invalid command
    EXPECT_THROW((void)simcore::replay(rails_config(), initial(), {}, -1), std::invalid_argument);
    EXPECT_EQ(run({}).tick, 100);
    const Snapshot later = run({{100, a}});  // takes effect at tick 100: not yet
    EXPECT_EQ(later.tick, 100);
    EXPECT_EQ(later.command.throttle[0], 0.0);
}

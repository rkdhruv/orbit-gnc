// M3.1: the simulation facade.
//
// Simulation owns the vehicle state and advances it in fixed ticks of h seconds with
// six_dof_rk4_step. Tick k runs from t_k = t0 + k h (computed from t0, never accumulated). The
// front end only does three things: submit(command), step(n) and snapshot() (a copy). A submitted
// command is validated at once and takes effect at the start of the next tick; commands submitted
// between two ticks apply in order, so the last one wins. The state after tick n therefore depends
// only on the initial state and the commands in effect at each tick, never on the frame rate or
// how the n ticks were chunked into step() calls: that is what makes deterministic replay
// (M3.5) possible.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "../integrators/orbit_fixtures.hpp"
#include "simcore/constants.hpp"
#include "simcore/dynamics/force_sum.hpp"
#include "simcore/dynamics/point_mass_gravity.hpp"
#include "simcore/integrators/six_dof.hpp"
#include "simcore/sim/simulation.hpp"
#include "simcore/sim/tick_scheduler.hpp"

namespace {

using simcore::ConstantBodyTorque;
using simcore::Epoch;
using simcore::MassProperties;
using simcore::PropellantTank;
using simcore::ReactionWheel;
using simcore::SimConfig;
using simcore::Simulation;
using simcore::Thruster;
using simcore::VehicleCommand;
using simcore::VehicleConfig;
using simcore::VehicleState;

const Epoch kT0 = Epoch::from_tt_seconds_since_j2000(800'000'000, 0.25);

// 400 kg dry + 400 kg tank (zero tank inertia), one 1000 N / Isp 300 s main engine along +x
// through the CoM, one 10 N thruster off-CoM (spins about z), one z reaction wheel.
VehicleConfig test_vehicle() {
    return VehicleConfig{
        MassProperties(400.0, Eigen::Vector3d(100.0, 150.0, 200.0).asDiagonal(),
                       PropellantTank{400.0, Eigen::Matrix3d::Zero()}),
        {Thruster(1000.0, 300.0, Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitX()),
         Thruster(10.0, 300.0, Eigen::Vector3d(0.0, 0.5, 0.0), Eigen::Vector3d::UnitX())},
        {ReactionWheel(Eigen::Vector3d::UnitZ(), 0.1, 0.2, 600.0)}};
}

SimConfig free_space_config(double tick_s = 0.02) {
    return SimConfig{test_vehicle(), std::make_shared<simcore::ForceSum>(),
                     std::make_shared<ConstantBodyTorque>(Eigen::Vector3d::Zero()), tick_s,
                     std::nullopt};
}

VehicleState at_rest() {
    return VehicleState{kT0,
                        Eigen::Vector3d::Zero(),
                        Eigen::Vector3d::Zero(),
                        Eigen::Quaterniond::Identity(),
                        Eigen::Vector3d::Zero(),
                        400.0,
                        {0.0}};
}

void expect_identical(const VehicleState& a, const VehicleState& b) {
    EXPECT_EQ(a.t, b.t);
    EXPECT_EQ(a.r_eci, b.r_eci);
    EXPECT_EQ(a.v_eci, b.v_eci);
    EXPECT_EQ(a.q_body_to_eci.coeffs(), b.q_body_to_eci.coeffs());
    EXPECT_EQ(a.w_body, b.w_body);
    EXPECT_EQ(a.propellant_kg, b.propellant_kg);
    EXPECT_EQ(a.wheel_h_nms, b.wheel_h_nms);
}

}  // namespace

// The architecture rule, checked at compile time: the front end gets a copy, never a reference
// into the sim's internals.
static_assert(!std::is_reference_v<decltype(std::declval<const Simulation&>().snapshot())>);

// ---------------------------------------------------------------------------------------------
// Fixed tick schedule
// ---------------------------------------------------------------------------------------------

// The initial command is all zeros (engines off, wheels idle). After n ticks the clock reads
// exactly t0 + n h, however the ticks were chunked: 100,000 ticks of 0.02 s in uneven chunks.
TEST(Simulation, ClockFollowsTheTickScheduleWithoutDrift) {
    Simulation sim(free_space_config(), at_rest());
    const simcore::Snapshot s0 = sim.snapshot();
    EXPECT_EQ(s0.tick, 0);
    EXPECT_EQ(s0.state.t, kT0);
    EXPECT_EQ(s0.command.throttle, (std::vector<double>{0.0, 0.0}));
    EXPECT_EQ(s0.command.wheel_torque_nm, (std::vector<double>{0.0}));
    for (int k = 0; k < 7; ++k) {
        sim.step(1);
    }
    sim.step(0);  // no-op
    sim.step(99'993);
    const simcore::Snapshot s = sim.snapshot();
    EXPECT_EQ(s.tick, 100'000);
    EXPECT_EQ(s.state.t, kT0 + 100'000.0 * 0.02);
    EXPECT_LE(std::abs((s.state.t - kT0) - 2000.0), 1e-12);
    EXPECT_EQ(s.state.v_eci, Eigen::Vector3d::Zero());  // nothing fired
}

// The facade adds nothing to the physics: Simulation over 100 ticks with a command change at
// tick 40 is bit-identical to a hand-written loop of six_dof_rk4_step with the same schedule.
// The sim is also driven through TickScheduler at 60 fps and at 144 fps (different chunking of
// the same ticks); all three runs must agree bit for bit. 420 km circular orbit with point-mass
// gravity; main engine at 30%, then the off-CoM thruster and the wheel.
TEST(Simulation, MatchesTheBareStepLoopAndIgnoresFrameRate) {
    const auto gravity = std::make_shared<simcore::PointMassGravity>();
    const auto no_torque = std::make_shared<ConstantBodyTorque>(Eigen::Vector3d::Zero());
    const SimConfig cfg{test_vehicle(), gravity, no_torque, 0.02, std::nullopt};
    const simcore::State orbit = orbit_fixtures::circular_initial_state();
    VehicleState init = at_rest();
    init.t = orbit.t;
    init.r_eci = orbit.r_eci;
    init.v_eci = orbit.v_eci;
    const VehicleCommand burn{{0.3, 0.0}, {0.0}};
    const VehicleCommand turn{{0.0, 1.0}, {0.05}};

    // Reference: the bare integrator loop.
    VehicleState ref = init;
    for (int k = 0; k < 100; ++k) {
        ref.t = init.t + static_cast<double>(k) * 0.02;
        ref = simcore::six_dof_rk4_step(cfg.vehicle, *gravity, *no_torque, k < 40 ? burn : turn,
                                        ref, 0.02);
    }
    ref.t = init.t + 100.0 * 0.02;

    // Frame-driven runs: submit the change when the sim reaches tick 40, as a front end would.
    for (const double fps : {60.0, 144.0}) {
        SCOPED_TRACE(fps);
        Simulation sim(cfg, init);
        simcore::TickScheduler sched(0.02);
        sim.submit(burn);
        while (sim.snapshot().tick < 100) {
            std::int64_t n = sched.ticks_for_frame(1.0 / fps).ticks;
            while (n > 0 && sim.snapshot().tick < 100) {
                const std::int64_t tick = sim.snapshot().tick;
                const std::int64_t stop = tick < 40 ? 40 : 100;  // where the driver acts next
                const std::int64_t chunk = std::min(n, stop - tick);
                sim.step(chunk);
                n -= chunk;
                if (sim.snapshot().tick == 40) {
                    sim.submit(turn);
                }
            }
        }
        expect_identical(sim.snapshot().state, ref);
        EXPECT_EQ(sim.snapshot().command.throttle, turn.throttle);
    }
}

// ---------------------------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------------------------

// Submitted at tick 25, a full-throttle command has no effect until the next tick runs; it then
// burns for ticks 25..49 = 0.5 s. Main engine only: m_dot = 1000 / (300 g0) = 0.339905 kg/s,
//   dv = v_e ln(m0 / (m0 - 0.5 m_dot)) = 2941.995 * ln(800 / 799.830047) = 0.625066 m/s.
// RK4 is exact to roundoff on this smooth burn (see the M2.4 full-burn test): bound 1e-10.
TEST(Simulation, CommandTakesEffectAtTheNextTick) {
    Simulation sim(free_space_config(), at_rest());
    sim.step(25);
    sim.submit(VehicleCommand{{1.0, 0.0}, {0.0}});
    EXPECT_EQ(sim.snapshot().state.v_eci, Eigen::Vector3d::Zero());  // not yet applied
    sim.step(25);
    const double ve = 300.0 * simcore::g0;
    const double dv = ve * std::log(800.0 / (800.0 - 0.5 * 1000.0 / ve));
    EXPECT_NEAR(dv, 0.625066, 1e-6);  // hand value
    EXPECT_NEAR(sim.snapshot().state.v_eci.x(), dv, 1e-10);
    EXPECT_EQ(sim.snapshot().command.throttle, (std::vector<double>{1.0, 0.0}));
}

// Several commands before one tick apply in order: the last wins. Full throttle then off: the
// vehicle doesn't move.
TEST(Simulation, LastCommandBeforeATickWins) {
    Simulation sim(free_space_config(), at_rest());
    sim.submit(VehicleCommand{{1.0, 0.0}, {0.0}});
    sim.submit(VehicleCommand{{0.0, 0.0}, {0.0}});
    sim.step(10);
    EXPECT_EQ(sim.snapshot().state.v_eci, Eigen::Vector3d::Zero());
    EXPECT_EQ(sim.snapshot().state.propellant_kg, 400.0);
}

// ---------------------------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------------------------

TEST(Simulation, RejectsInvalidConfigStateAndCommands) {
    const double kNaN = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((void)Simulation(free_space_config(0.0), at_rest()), std::invalid_argument);
    EXPECT_THROW((void)Simulation(free_space_config(kNaN), at_rest()), std::invalid_argument);
    SimConfig no_forces = free_space_config();
    no_forces.forces = nullptr;
    EXPECT_THROW((void)Simulation(no_forces, at_rest()), std::invalid_argument);
    SimConfig no_torques = free_space_config();
    no_torques.torques = nullptr;
    EXPECT_THROW((void)Simulation(no_torques, at_rest()), std::invalid_argument);
    VehicleState bad = at_rest();
    bad.wheel_h_nms = {};
    EXPECT_THROW((void)Simulation(free_space_config(), bad), std::invalid_argument);
    bad = at_rest();
    bad.propellant_kg = 401.0;
    EXPECT_THROW((void)Simulation(free_space_config(), bad), std::invalid_argument);

    // A bad command throws at submit and is not queued: the next tick runs the old command.
    Simulation sim(free_space_config(), at_rest());
    EXPECT_THROW(sim.submit(VehicleCommand{{1.0}, {0.0}}), std::invalid_argument);       // count
    EXPECT_THROW(sim.submit(VehicleCommand{{1.5, 0.0}, {0.0}}), std::invalid_argument);  // range
    EXPECT_THROW(sim.submit(VehicleCommand{{0.0, 0.0}, {}}), std::invalid_argument);     // wheels
    EXPECT_THROW(sim.step(-1), std::invalid_argument);
    sim.step(5);
    EXPECT_EQ(sim.snapshot().tick, 5);
    EXPECT_EQ(sim.snapshot().state.v_eci, Eigen::Vector3d::Zero());
}

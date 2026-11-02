// M3.4: "on rails" coasting for time warp.
//
// When the environment is two-body gravity (SimConfig::rails_mu set) and the vehicle is coasting
// (all throttles 0, all wheel torque commands 0, |w| <= Simulation::kRailsMaxRate_radps), the sim
// stops integrating and evaluates the Kepler solution from the state at which it went on rails:
//   (r, v)(t_k) = kepler_propagate(entry, t_k)
// That is exact (no truncation error), costs the same for 1 tick or a million, and depends only on
// the tick index, so results don't depend on how ticks are chunked into step() calls. On entry
// any residual rotation below the threshold is stopped (w := 0, a documented simplification, as in
// KSP); with w = 0 and no torques, q, w, propellant and wheel momenta are then exactly constant
// (I_s w_dot = -w x H - sum tau_m a = 0). The first tick whose command isn't a coast goes back to
// RK4 from the current rails state, so the hand-off is continuous by construction: no state is
// re-derived at the switch.

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <optional>
#include <vector>

#include "../integrators/orbit_fixtures.hpp"
#include "simcore/constants.hpp"
#include "simcore/dynamics/point_mass_gravity.hpp"
#include "simcore/integrators/six_dof.hpp"
#include "simcore/orbits/kepler.hpp"
#include "simcore/sim/simulation.hpp"

namespace {

using simcore::ConstantBodyTorque;
using simcore::SimConfig;
using simcore::Simulation;
using simcore::VehicleCommand;
using simcore::VehicleState;

// 400 kg dry + 400 kg tank, main engine 1000 N / Isp 300 s along +x, three wheels on the body
// axes (J = 0.05, 0.1 N m, 600 rad/s).
simcore::VehicleConfig test_vehicle() {
    return simcore::VehicleConfig{
        simcore::MassProperties(400.0, Eigen::Vector3d(300.0, 400.0, 450.0).asDiagonal(),
                                simcore::PropellantTank{400.0, Eigen::Matrix3d::Zero()}),
        {simcore::Thruster(1000.0, 300.0, Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitX())},
        {simcore::ReactionWheel(Eigen::Vector3d::UnitX(), 0.05, 0.1, 600.0),
         simcore::ReactionWheel(Eigen::Vector3d::UnitY(), 0.05, 0.1, 600.0),
         simcore::ReactionWheel(Eigen::Vector3d::UnitZ(), 0.05, 0.1, 600.0)}};
}

SimConfig two_body(bool rails) {
    SimConfig cfg{test_vehicle(), std::make_shared<simcore::PointMassGravity>(),
                  std::make_shared<ConstantBodyTorque>(Eigen::Vector3d::Zero()), 0.02,
                  std::nullopt};
    if (rails) {
        cfg.rails_mu = simcore::earth::mu;
    }
    return cfg;
}

// 420 km circular orbit, attitude 0.3 rad about (1, 2, 3), at rest, wheels holding some momentum.
VehicleState initial() {
    const simcore::State o = orbit_fixtures::circular_initial_state();
    return VehicleState{
        o.t,
        o.r_eci,
        o.v_eci,
        Eigen::Quaterniond(Eigen::AngleAxisd(0.3, Eigen::Vector3d(1.0, 2.0, 3.0).normalized())),
        Eigen::Vector3d::Zero(),
        300.0,
        {1.0, -2.0, 0.5}};
}

const VehicleCommand kCoast{{0.0}, {0.0, 0.0, 0.0}};
const VehicleCommand kBurn{{0.5}, {0.0, 0.0, 0.0}};

}  // namespace

// A coast on rails is the Kepler solution from the entry state, bit for bit, after any number of
// ticks; attitude, rates, propellant and wheel momenta don't change at all. 1e6 ticks (20,000 s,
// ~3.6 orbits) cost one Kepler solve.
TEST(Rails, CoastIsTheKeplerSolutionFromTheEntryState) {
    Simulation sim(two_body(true), initial());
    const VehicleState s0 = initial();
    sim.step(1'000'000);
    const simcore::Snapshot snap = sim.snapshot();
    EXPECT_TRUE(snap.on_rails);
    EXPECT_EQ(snap.state.t, s0.t + 1'000'000.0 * 0.02);
    const simcore::State k = simcore::kepler_propagate(simcore::State{s0.t, s0.r_eci, s0.v_eci},
                                                       snap.state.t, simcore::earth::mu);
    EXPECT_EQ(snap.state.r_eci, k.r_eci);
    EXPECT_EQ(snap.state.v_eci, k.v_eci);
    EXPECT_EQ(snap.state.q_body_to_eci.coeffs(), s0.q_body_to_eci.coeffs());
    EXPECT_EQ(snap.state.w_body, Eigen::Vector3d::Zero());
    EXPECT_EQ(snap.state.propellant_kg, s0.propellant_kg);
    EXPECT_EQ(snap.state.wheel_h_nms, s0.wheel_h_nms);
}

// Chunking doesn't matter on rails (every tick is evaluated from the same entry state):
// 7 x step(1) + step(993) is bit-identical to step(1000).
TEST(Rails, ChunkingDoesNotMatter) {
    Simulation a(two_body(true), initial());
    Simulation b(two_body(true), initial());
    for (int k = 0; k < 7; ++k) {
        a.step(1);
    }
    a.step(993);
    b.step(1000);
    EXPECT_EQ(a.snapshot().state.r_eci, b.snapshot().state.r_eci);
    EXPECT_EQ(a.snapshot().state.v_eci, b.snapshot().state.v_eci);
}

// Rails and integration describe the same motion: a 600 s coast on rails vs RK4 at h = 0.02 s
// (30,000 ticks, rails disabled). RK4's truncation error at this step is far below roundoff
// (M1: ~h^4 scaling, 26 m per orbit at h ~ 13 s), so the difference is accumulated roundoff,
// ~steps * eps * |r| = 30,000 * 2.2e-16 * 6.8e6 = 4.5e-5 m. Bound 1e-3 m and 1e-6 m/s.
TEST(Rails, AgreesWithIntegrationOfTheSameCoast) {
    Simulation rails(two_body(true), initial());
    Simulation rk4(two_body(false), initial());
    rails.step(30'000);
    rk4.step(30'000);
    EXPECT_TRUE(rails.snapshot().on_rails);
    EXPECT_FALSE(rk4.snapshot().on_rails);
    EXPECT_LE((rails.snapshot().state.r_eci - rk4.snapshot().state.r_eci).norm(), 1e-3);
    EXPECT_LE((rails.snapshot().state.v_eci - rk4.snapshot().state.v_eci).norm(), 1e-6);
}

// Hand-off: 1000 ticks on rails, then a burn. The burn tick starts from the rails state at tick
// 1000 (the Kepler solution) and integrates it: bit-identical to six_dof_rk4_step applied to that
// state. Cutting the engine then goes back on rails from the post-burn state, so after another
// 500 ticks the state is the Kepler solution from the state at tick 1010.
TEST(Rails, HandOffIsContinuousInBothDirections) {
    const VehicleState s0 = initial();
    const SimConfig cfg = two_body(true);
    Simulation sim(cfg, s0);
    sim.step(1000);
    const VehicleState on_rails = sim.snapshot().state;

    sim.submit(kBurn);
    sim.step(10);
    EXPECT_FALSE(sim.snapshot().on_rails);
    VehicleState ref = on_rails;
    for (int k = 1000; k < 1010; ++k) {
        ref.t = s0.t + static_cast<double>(k) * 0.02;
        ref = simcore::six_dof_rk4_step(cfg.vehicle, *cfg.forces, *cfg.torques, kBurn, ref, 0.02);
    }
    ref.t = s0.t + 1010.0 * 0.02;
    EXPECT_EQ(sim.snapshot().state.r_eci, ref.r_eci);
    EXPECT_EQ(sim.snapshot().state.v_eci, ref.v_eci);
    EXPECT_EQ(sim.snapshot().state.propellant_kg, ref.propellant_kg);
    EXPECT_LT(ref.propellant_kg, s0.propellant_kg);  // the burn really used propellant

    sim.submit(kCoast);
    sim.step(500);
    EXPECT_TRUE(sim.snapshot().on_rails);
    const simcore::State k = simcore::kepler_propagate(simcore::State{ref.t, ref.r_eci, ref.v_eci},
                                                       sim.snapshot().state.t, simcore::earth::mu);
    EXPECT_EQ(sim.snapshot().state.r_eci, k.r_eci);
    EXPECT_EQ(sim.snapshot().state.v_eci, k.v_eci);
}

// Not a coast, so no rails: thrust, a wheel torque command, rotating faster than the threshold,
// or an environment that isn't two-body (rails_mu unset).
TEST(Rails, OnlyCoastingTwoBodyVehiclesGoOnRails) {
    {
        Simulation sim(two_body(true), initial());
        sim.submit(kBurn);
        sim.step(5);
        EXPECT_FALSE(sim.snapshot().on_rails);
    }
    {
        Simulation sim(two_body(true), initial());
        sim.submit(VehicleCommand{{0.0}, {0.0, 0.01, 0.0}});
        sim.step(5);
        EXPECT_FALSE(sim.snapshot().on_rails);
    }
    {
        VehicleState spinning = initial();
        spinning.w_body = Eigen::Vector3d(0.0, 0.0, 2.0 * Simulation::kRailsMaxRate_radps);
        Simulation sim(two_body(true), spinning);
        sim.step(5);
        EXPECT_FALSE(sim.snapshot().on_rails);
        EXPECT_NE(sim.snapshot().state.w_body, Eigen::Vector3d::Zero());
    }
    {
        Simulation sim(two_body(false), initial());
        sim.step(5);
        EXPECT_FALSE(sim.snapshot().on_rails);
    }
}

// Entering rails stops a residual rotation below the threshold: w = 5e-5 rad/s becomes exactly 0
// and q is frozen at its entry value.
TEST(Rails, EntryStopsResidualRotation) {
    VehicleState slow = initial();
    slow.w_body = Eigen::Vector3d(0.0, 0.5 * Simulation::kRailsMaxRate_radps, 0.0);
    Simulation sim(two_body(true), slow);
    sim.step(100);
    EXPECT_TRUE(sim.snapshot().on_rails);
    EXPECT_EQ(sim.snapshot().state.w_body, Eigen::Vector3d::Zero());
    EXPECT_EQ(sim.snapshot().state.q_body_to_eci.coeffs(), slow.q_body_to_eci.coeffs());
}

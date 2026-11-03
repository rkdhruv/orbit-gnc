// M4.4: maneuver nodes and their execution.
//
// RSW frame at a state: R = r_hat (radial), W = h_hat (orbit normal), S = W x R (along-track, in
// the direction of motion for a circular orbit). A node is a time and a dv in RSW at that time;
// its impulsive effect is v+ = v + [R S W] dv_rsw at the node state (Kepler-propagated from now:
// exact in two-body, approximate with J2).
// Finite burn with constant thrust T from mass m0, exhaust speed v_e (rocket equation):
//   duration   t_b   = (m0 v_e / T) (1 - exp(-dv / v_e))
//   lead       t_l   = (m0 v_e / T) (1 - exp(-dv / (2 v_e)))   (time to deliver half the dv)
// Ignition at t_node - t_l centres the burn on the node by dv rather than by time (the
// acceleration grows as propellant burns, so the second half is shorter).
// Execution: hold the burn attitude (body x along dv_eci, fixed in inertial space) until ignition,
// then burn; the achieved dv is measured from the propellant used, v_e ln(m_ign / m), and near the
// end the throttle is set to deliver exactly the remainder over one command hold (cut off at a
// 1 mm/s residual).

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <stdexcept>

#include <Eigen/Geometry>

#include "../orbits/elements_fixtures.hpp"
#include "simcore/constants.hpp"
#include "simcore/dynamics/point_mass_gravity.hpp"
#include "simcore/gnc/attitude_control.hpp"
#include "simcore/gnc/maneuver.hpp"
#include "simcore/orbits/elements.hpp"
#include "simcore/sim/simulation.hpp"

namespace {

using elements_fixtures::kDeg;
using elements_fixtures::state_from;
using simcore::ManeuverExecutor;
using simcore::ManeuverNode;
using simcore::State;
using simcore::VehicleConfig;
using simcore::VehicleState;
namespace earth = simcore::earth;

constexpr double kPi = std::numbers::pi;
const double kR420 = earth::radius + 420.0e3;

// 500 kg dry, I = diag(300, 400, 450), 300 kg propellant (zero-inertia tank), main engine 400 N /
// Isp 320 s along body +x, 0.1 N m wheels on the body axes: m0 = 800 kg, v_e = 3138.128 m/s.
VehicleConfig vehicle() {
    return VehicleConfig{
        simcore::MassProperties(500.0, Eigen::Vector3d(300.0, 400.0, 450.0).asDiagonal(),
                                simcore::PropellantTank{300.0, Eigen::Matrix3d::Zero()}),
        {simcore::Thruster(400.0, 320.0, Eigen::Vector3d(-1.0, 0.0, 0.0),
                           Eigen::Vector3d::UnitX())},
        {simcore::ReactionWheel(Eigen::Vector3d::UnitX(), 0.05, 0.1, 600.0),
         simcore::ReactionWheel(Eigen::Vector3d::UnitY(), 0.05, 0.1, 600.0),
         simcore::ReactionWheel(Eigen::Vector3d::UnitZ(), 0.05, 0.1, 600.0)}};
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// RSW frame and impulsive prediction
// ---------------------------------------------------------------------------------------------

// Equatorial circular orbit at nu = 0: r along x, v along y, h along z, so R, S, W are the ECI
// axes and [R S W] is the identity. On an inclined ellipse the frame is a proper rotation with
// R = r_hat and W = h_hat. Bounds 1e-15.
TEST(Maneuver, RswFrame) {
    const State eq = state_from(kR420, 0.0, 0.0, 0.0, 0.0, 0.0);
    EXPECT_LE((simcore::rsw_to_eci(eq) - Eigen::Matrix3d::Identity()).norm(), 1e-15);
    const State el = state_from(1.2e7, 0.3, 63.4 * kDeg, 40.0 * kDeg, 270.0 * kDeg, 1.0);
    const Eigen::Matrix3d M = simcore::rsw_to_eci(el);
    EXPECT_LE((M.col(0) - el.r_eci.normalized()).norm(), 1e-15);
    EXPECT_LE((M.col(2) - el.r_eci.cross(el.v_eci).normalized()).norm(), 1e-15);
    EXPECT_LE((M.transpose() * M - Eigen::Matrix3d::Identity()).norm(), 1e-15);
    EXPECT_NEAR(M.determinant(), 1.0, 1e-15);
}

// 50 m/s prograde at a node 1000 s ahead on the 420 km circular orbit. The node state is on the
// same circle; after the burn, v+ = v_c + 50 = 7707.27 m/s, so by vis-viva
// a+ = 1 / (2 / r - v+^2 / mu) = 6888.390 km and apoapsis r_a = 2 a+ - r = R + 600.505 km, with
// periapsis at the burn point (420 km) and e = (r_a - r) / (r_a + r) = 0.0131021.
// Bounds: periapsis 1 mm, apoapsis 1 m, e 1e-7 (Kepler + elements, roundoff ~1e-9 relative).
TEST(Maneuver, ImpulsivePredictionOfAPrograde50) {
    const State now = state_from(kR420, 0.0, 0.0, 0.0, 0.0, 0.0);
    const ManeuverNode node{now.t + 1000.0, Eigen::Vector3d(0.0, 50.0, 0.0)};
    const State at = simcore::node_state(now, node);
    EXPECT_NEAR(at.r_eci.norm(), kR420, 1e-6);
    const simcore::KeplerianElements el =
        simcore::elements_from_state(simcore::post_burn_state(now, node));
    EXPECT_NEAR(el.a_m * (1.0 - el.e) - earth::radius, 420.0e3, 1e-3);
    EXPECT_NEAR(el.a_m * (1.0 + el.e) - earth::radius, 600.505e3, 1.0);
    EXPECT_NEAR(el.e, 0.0131021, 1e-7);
}

// Burn duration and lead for 50 m/s from 800 kg at 400 N, v_e = 3138.128 m/s:
//   t_b = 6276.256 * (1 - e^(-0.015933)) = 99.208 s,  t_l = 6276.256 * (1 - e^(-0.0079666))
//   = 49.801 s (> t_b / 2 = 49.604: the acceleration grows as the tank empties).
TEST(Maneuver, BurnDurationAndLead) {
    const double ve = 320.0 * simcore::g0;
    EXPECT_NEAR(simcore::burn_duration_s(50.0, 800.0, 400.0, ve), 99.208, 1e-3);
    EXPECT_NEAR(simcore::burn_lead_s(50.0, 800.0, 400.0, ve), 49.801, 1e-3);
    EXPECT_GT(simcore::burn_lead_s(50.0, 800.0, 400.0, ve),
              0.5 * simcore::burn_duration_s(50.0, 800.0, 400.0, ve));
}

// ---------------------------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------------------------

// The autopilot flies the node: start 90 deg off (body x radial, at rest), node 1500 s ahead,
// 50 m/s prograde; PD hold (wn = 0.05, zeta = 0.7) with the command recomputed every 0.5 s
// (0.1 s ticks, two-body gravity). Checks:
//   - it is aligned before ignition (body x within 0.5 deg of dv_eci);
//   - ignition at t_node - t_l = t_node - 49.80 s, to one command hold (0.5 s);
//   - achieved dv (from propellant) = 50 m/s within 2 mm/s (the last hold is throttled to the
//     remainder; mass change over 0.5 s is 1e-4 relative);
//   - propellant used m0 (1 - e^(-dv / v_e)) = 12.645 kg (follows from the dv): within 0.01 kg;
//   - the orbit matches the impulsive prediction (periapsis 420 km, apoapsis 600.505 km) to within
//     the finite-burn loss: the 99 s burn sweeps 6.4 deg of the orbit with a fixed inertial
//     direction, up to 3.2 deg off the local prograde, so the cosine loss is ~0.05% of 50 m/s
//     (~0.03 m/s, ~0.1 km of apoapsis) plus a small radial component: bound 2 km on both.
TEST(Maneuver, ExecutorFliesAPrograde50) {
    const VehicleConfig cfg = vehicle();
    const State o = state_from(kR420, 0.0, 0.0, 0.0, 0.0, 0.0);
    const VehicleState s0{
        o.t,   o.r_eci,        o.v_eci, Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero(),
        300.0, {0.0, 0.0, 0.0}};
    simcore::Simulation sim(
        simcore::SimConfig{cfg, std::make_shared<simcore::PointMassGravity>(),
                           std::make_shared<simcore::ConstantBodyTorque>(Eigen::Vector3d::Zero()),
                           0.1, std::nullopt},
        s0);
    const ManeuverNode node{o.t + 1500.0, Eigen::Vector3d(0.0, 50.0, 0.0)};
    ManeuverExecutor exec(cfg, s0, node, 0.5);
    const double ve = 320.0 * simcore::g0;
    EXPECT_NEAR(exec.ignition_time() - node.t, -49.801, 1e-3);

    double ign_seen = -1.0;
    double align_err_deg = 180.0;
    for (int k = 0; k < 4000 && exec.phase() != ManeuverExecutor::Phase::kDone; ++k) {
        const VehicleState& s = sim.snapshot().state;
        const simcore::AttitudeGains g =
            simcore::pd_gains(cfg.mass.inertia_kgm2(s.propellant_kg), simcore::PdGains{0.05, 0.7});
        const ManeuverExecutor::Phase before = exec.phase();
        const simcore::VehicleCommand c = exec.command(cfg, s, g);
        if (before == ManeuverExecutor::Phase::kAlign &&
            exec.phase() == ManeuverExecutor::Phase::kBurn) {
            ign_seen = s.t - node.t;
            const Eigen::Vector3d x = s.q_body_to_eci * Eigen::Vector3d::UnitX();
            align_err_deg =
                std::acos(std::min(1.0, x.dot(exec.burn_direction_eci()))) * 180.0 / kPi;
        }
        sim.submit(c);
        sim.step(5);
    }
    ASSERT_EQ(exec.phase(), ManeuverExecutor::Phase::kDone);
    const VehicleState& f = sim.snapshot().state;
    const simcore::KeplerianElements el =
        simcore::elements_from_state(State{f.t, f.r_eci, f.v_eci});
    const double dv = ve * std::log(800.0 / cfg.mass.mass_kg(f.propellant_kg));
    std::printf(
        "[node] ignition %+.2f s, aligned to %.3f deg, dv %.4f m/s, propellant %.4f kg, "
        "peri %.3f km, apo %.3f km\n",
        ign_seen, align_err_deg, dv, 300.0 - f.propellant_kg,
        (el.a_m * (1.0 - el.e) - earth::radius) / 1e3,
        (el.a_m * (1.0 + el.e) - earth::radius) / 1e3);
    EXPECT_LE(align_err_deg, 0.5);
    EXPECT_NEAR(ign_seen, -49.801, 0.5);
    EXPECT_NEAR(dv, 50.0, 2e-3);
    EXPECT_NEAR(exec.achieved_dv_mps(), dv, 1e-9);
    EXPECT_NEAR(300.0 - f.propellant_kg, 12.645, 0.01);
    EXPECT_NEAR(el.a_m * (1.0 - el.e) - earth::radius, 420.0e3, 2.0e3);
    EXPECT_NEAR(el.a_m * (1.0 + el.e) - earth::radius, 600.505e3, 2.0e3);
}

// A node too close for the slew: start retrograde (180 deg off) with 0.1 N m wheels and only
// 100 s to the node (ignition scheduled at 50 s), while the saturated slew takes minutes. Firing on
// schedule would put part of the dv out of plane (found live: a 300 s node changed the inclination
// by 0.25 deg). The executor instead waits until it is aligned (attitude within 1 deg, rate settled
// within 0.05 deg/s) and ignites late, flagged as delayed. While delayed it re-plans the burn as
// centred at (now + lead) and chases that attitude (it turns with the orbit), because the node's
// own inertial direction goes stale at the orbit rate (found live too: igniting ~4 min late with
// the stale direction lost ~2 m/s of the along-track dv, 6 km of apoapsis). Checks: ignition after
// its scheduled time, aligned within 1 deg when it fires, dv still 50 m/s (2 mm/s), the equatorial
// orbit stays equatorial (inclination < 0.01 deg: no out-of-plane dv), and apoapsis within the
// same 2 km of the impulsive 600.505 km as an on-time burn (the dv went along-track).
TEST(Maneuver, ExecutorWaitsForAlignmentBeforeIgniting) {
    const VehicleConfig cfg = vehicle();
    const State o = state_from(kR420, 0.0, 0.0, 0.0, 0.0, 0.0);
    const VehicleState s0{
        o.t,
        o.r_eci,
        o.v_eci,
        Eigen::Quaterniond(Eigen::AngleAxisd(-kPi / 2.0, Eigen::Vector3d::UnitZ())),
        Eigen::Vector3d::Zero(),
        300.0,
        {0.0, 0.0, 0.0}};
    simcore::Simulation sim(
        simcore::SimConfig{cfg, std::make_shared<simcore::PointMassGravity>(),
                           std::make_shared<simcore::ConstantBodyTorque>(Eigen::Vector3d::Zero()),
                           0.1, std::nullopt},
        s0);
    const ManeuverNode node{o.t + 100.0, Eigen::Vector3d(0.0, 50.0, 0.0)};
    ManeuverExecutor exec(cfg, s0, node, 0.5);
    const simcore::Epoch scheduled = exec.ignition_time();  // before any re-planning
    double ign_after_schedule = -1.0;
    double align_err_deg = 180.0;
    for (int k = 0; k < 4000 && exec.phase() != ManeuverExecutor::Phase::kDone; ++k) {
        const VehicleState& s = sim.snapshot().state;
        const simcore::AttitudeGains g =
            simcore::pd_gains(cfg.mass.inertia_kgm2(s.propellant_kg), simcore::PdGains{0.05, 0.7});
        const ManeuverExecutor::Phase before = exec.phase();
        const simcore::VehicleCommand c = exec.command(cfg, s, g);
        if (before == ManeuverExecutor::Phase::kAlign &&
            exec.phase() == ManeuverExecutor::Phase::kBurn) {
            ign_after_schedule = s.t - scheduled;
            const Eigen::Vector3d x = s.q_body_to_eci * Eigen::Vector3d::UnitX();
            align_err_deg =
                std::acos(std::min(1.0, x.dot(exec.burn_direction_eci()))) * 180.0 / kPi;
        }
        sim.submit(c);
        sim.step(5);
    }
    ASSERT_EQ(exec.phase(), ManeuverExecutor::Phase::kDone);
    const VehicleState& f = sim.snapshot().state;
    const double dv = 320.0 * simcore::g0 * std::log(800.0 / cfg.mass.mass_kg(f.propellant_kg));
    const simcore::KeplerianElements el =
        simcore::elements_from_state(State{f.t, f.r_eci, f.v_eci});
    std::printf(
        "[late node] ignited %.1f s after schedule, aligned to %.3f deg, dv %.4f m/s, "
        "i %.5f deg, apo %.3f km\n",
        ign_after_schedule, align_err_deg, dv, el.i_rad * 180.0 / kPi,
        (el.a_m * (1.0 + el.e) - earth::radius) / 1e3);
    EXPECT_GT(ign_after_schedule, 0.0);
    EXPECT_TRUE(exec.ignition_delayed());
    EXPECT_LE(align_err_deg, 1.0);
    EXPECT_NEAR(dv, 50.0, 2e-3);
    EXPECT_LT(el.i_rad * 180.0 / kPi, 0.01);
    EXPECT_NEAR(el.a_m * (1.0 + el.e) - earth::radius, 600.505e3, 2.0e3);
}

TEST(Maneuver, RejectsInvalidNodes) {
    const VehicleConfig cfg = vehicle();
    const State o = state_from(kR420, 0.0, 0.0, 0.0, 0.0, 0.0);
    const VehicleState s0{
        o.t,   o.r_eci,        o.v_eci, Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero(),
        300.0, {0.0, 0.0, 0.0}};
    const double kNaN = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(
        (void)ManeuverExecutor(cfg, s0, ManeuverNode{o.t + 100.0, Eigen::Vector3d::Zero()}, 0.5),
        std::invalid_argument);  // zero dv
    EXPECT_THROW((void)ManeuverExecutor(
                     cfg, s0, ManeuverNode{o.t + 100.0, Eigen::Vector3d(kNaN, 0, 0)}, 0.5),
                 std::invalid_argument);
    VehicleConfig no_engine = cfg;
    no_engine.thrusters.clear();
    EXPECT_THROW((void)ManeuverExecutor(no_engine, s0,
                                        ManeuverNode{o.t + 100.0, Eigen::Vector3d(0, 50, 0)}, 0.5),
                 std::invalid_argument);
    EXPECT_THROW((void)simcore::burn_duration_s(5000.0, 800.0, 400.0, 0.0), std::invalid_argument);
}

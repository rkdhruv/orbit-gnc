// M4.3: RCS torque couples.
//
// Each RcsAxis models the RCS thrusters that turn the vehicle about one body axis: couples of two
// thrusters firing in opposite directions at +-L from the centre of mass, so they give a pure
// torque (no net force). A command u in [-1, 1] picks the couple (sign) and its throttle:
//   torque = u tau_max a_hat,   total thrust = |torque| / L,   m_dot = |u| tau_max / (L Isp g0)
// (each of the two thrusters gives T = |torque| / (2 L); both burn propellant).

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>

#include "simcore/constants.hpp"
#include "simcore/dynamics/force_sum.hpp"
#include "simcore/integrators/six_dof.hpp"
#include "simcore/sim/simulation.hpp"
#include "simcore/vehicle/rcs.hpp"

namespace {

using simcore::RcsAxis;
using simcore::VehicleCommand;
using simcore::VehicleConfig;
using simcore::VehicleState;

const simcore::ConstantBodyTorque kNoTorque{Eigen::Vector3d::Zero()};
const simcore::ForceSum kFreeSpace;

// 400 kg dry, I = diag(100, 150, 200), tank 10 kg with no inertia (I constant), no main engine
// or wheels, one RCS axis about body z: 0.5 N m, L = 1 m, Isp 220 s (hydrazine).
VehicleConfig rcs_vehicle() {
    VehicleConfig cfg{
        simcore::MassProperties(400.0, Eigen::Vector3d(100.0, 150.0, 200.0).asDiagonal(),
                                simcore::PropellantTank{10.0, Eigen::Matrix3d::Zero()}),
        {},
        {}};
    cfg.rcs = {RcsAxis(Eigen::Vector3d::UnitZ(), 0.5, 1.0, 220.0)};
    return cfg;
}

VehicleState at_rest(double propellant_kg) {
    return VehicleState{simcore::Epoch::from_tt_seconds_since_j2000(800'000'000),
                        Eigen::Vector3d::Zero(),
                        Eigen::Vector3d::Zero(),
                        Eigen::Quaterniond::Identity(),
                        Eigen::Vector3d::Zero(),
                        propellant_kg,
                        {}};
}

}  // namespace

// Axis given unnormalised as (0, 0, 2). u = 0.4: torque (0, 0, 0.2) N m, total thrust 0.2 N,
// m_dot = 0.2 / (1 * 220 * 9.80665) = 9.2701e-5 kg/s; u = -0.4 reverses the torque, same flow.
TEST(Rcs, TorqueAndMassFlow) {
    const RcsAxis rcs(Eigen::Vector3d(0.0, 0.0, 2.0), 0.5, 1.0, 220.0);
    EXPECT_LE((rcs.torque_body_nm(0.4) - Eigen::Vector3d(0.0, 0.0, 0.2)).norm(), 1e-16);
    EXPECT_LE((rcs.torque_body_nm(-0.4) - Eigen::Vector3d(0.0, 0.0, -0.2)).norm(), 1e-16);
    EXPECT_NEAR(rcs.mass_flow_kgps(0.4), 9.2701e-5, 1e-9);
    EXPECT_NEAR(rcs.mass_flow_kgps(0.4), 0.2 / (220.0 * simcore::g0), 1e-18);
    EXPECT_EQ(rcs.mass_flow_kgps(-0.4), rcs.mass_flow_kgps(0.4));
    EXPECT_EQ(rcs.mass_flow_kgps(0.0), 0.0);
    EXPECT_EQ(rcs.max_torque_nm(), 0.5);
}

// Full couple about z for 10 s in free space (0.1 s ticks): w_z = tau t / I_zz = 0.5 * 10 / 200
// = 0.025 rad/s (constant torque about a principal axis: RK4 exact, roundoff only); no net force,
// so r and v stay exactly 0; propellant drops by m_dot t = 0.5 / (220 g0) * 10 = 2.31754e-3 kg.
// The rate is constant, but each of the 100 steps subtracts from m_p ~ 5 kg (ulp 8.9e-16), so
// roundoff accumulates to ~100 * 4.4e-16 = 4.4e-14 kg: bound 1e-13.
TEST(Rcs, CoupleSpinsTheVehicleWithoutMovingIt) {
    simcore::Simulation sim(
        simcore::SimConfig{rcs_vehicle(), std::make_shared<simcore::ForceSum>(),
                           std::make_shared<simcore::ConstantBodyTorque>(Eigen::Vector3d::Zero()),
                           0.1, std::nullopt, 0.0},
        at_rest(5.0));
    VehicleCommand c{{}, {}};
    c.rcs = {1.0};
    sim.submit(c);
    sim.step(100);
    const VehicleState& s = sim.snapshot().state;
    EXPECT_NEAR(s.w_body.z(), 0.025, 1e-15);
    EXPECT_EQ(s.r_eci, Eigen::Vector3d::Zero());
    EXPECT_EQ(s.v_eci, Eigen::Vector3d::Zero());
    EXPECT_NEAR(5.0 - s.propellant_kg, 0.5 / (220.0 * simcore::g0) * 10.0, 1e-13);
    EXPECT_NEAR(5.0 - s.propellant_kg, 2.31754e-3, 1e-8);  // hand value
    EXPECT_FALSE(simcore::is_coast(c));
}

// The burnout step scales RCS commands like throttles (M2 review): with 1e-4 kg left and
// m_dot = 2.3175e-4 kg/s at u = 1, a 1 s step can sustain f = 1e-4 / 2.3175e-4 = 0.4315 of the
// command, so it spins the body to w_z = f * 0.5 * 1 / 200 and ends exactly empty.
TEST(Rcs, BurnoutStepScalesTheCouple) {
    const VehicleConfig cfg = rcs_vehicle();
    VehicleCommand c{{}, {}};
    c.rcs = {1.0};
    const VehicleState s =
        simcore::six_dof_rk4_step(cfg, kFreeSpace, kNoTorque, c, at_rest(1e-4), 1.0);
    const double mdot = 0.5 / (220.0 * simcore::g0);
    EXPECT_NEAR(s.w_body.z(), (1e-4 / mdot) * 0.5 / 200.0, 1e-15);
    EXPECT_EQ(s.propellant_kg, 0.0);
}

TEST(Rcs, RejectsInvalidInput) {
    const double kNaN = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((void)RcsAxis(Eigen::Vector3d::Zero(), 0.5, 1.0, 220.0), std::invalid_argument);
    EXPECT_THROW((void)RcsAxis(Eigen::Vector3d::UnitZ(), 0.0, 1.0, 220.0), std::invalid_argument);
    EXPECT_THROW((void)RcsAxis(Eigen::Vector3d::UnitZ(), 0.5, 0.0, 220.0), std::invalid_argument);
    EXPECT_THROW((void)RcsAxis(Eigen::Vector3d::UnitZ(), 0.5, 1.0, kNaN), std::invalid_argument);
    const RcsAxis rcs(Eigen::Vector3d::UnitZ(), 0.5, 1.0, 220.0);
    EXPECT_THROW((void)rcs.torque_body_nm(1.5), std::invalid_argument);
    EXPECT_THROW((void)rcs.mass_flow_kgps(kNaN), std::invalid_argument);
    // Commands must carry one value in [-1, 1] per RCS axis.
    const VehicleConfig cfg = rcs_vehicle();
    VehicleCommand none{{}, {}};
    EXPECT_THROW(simcore::validate_command(cfg, none), std::invalid_argument);
    VehicleCommand big{{}, {}};
    big.rcs = {-1.5};
    EXPECT_THROW(simcore::validate_command(cfg, big), std::invalid_argument);
}

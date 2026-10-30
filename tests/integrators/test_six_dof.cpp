// M2.4: coupled 6-DOF vehicle propagation (fixed-step RK4).
//
// State y = [r, v, q, w, m_p, h_1..h_n]:
//   r_dot = v
//   v_dot = a_env(t, r, v) + R(q) sum F_i / m(m_p)                (thrust in body axes)
//   q_dot = 1/2 q (x) [0, w]
//   w_dot = from I_s w_dot = tau_ext + sum r_i x F_i - w x H - sum tau_motor_j a_j, I = I(m_p)
//   m_p_dot = -sum m_dot_i                                       (0 once the tank is empty)
//   h_j_dot = tau_motor_j (after torque/speed saturation)
// Commands (throttles, wheel torque commands) are held constant over a call. q is renormalised and
// m_p clamped at 0 after every step. Stated simplifications: the vehicle obeys Euler's equation
// with the current I(m_p) (no jet damping, no I_dot w term); drag models keep their own fixed mass.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <vector>

#include <Eigen/Geometry>

#include "../integrators/orbit_fixtures.hpp"
#include "simcore/dynamics/force_sum.hpp"
#include "simcore/dynamics/point_mass_gravity.hpp"
#include "simcore/integrators/attitude.hpp"
#include "simcore/integrators/fixed_step.hpp"
#include "simcore/integrators/six_dof.hpp"
#include "simcore/vehicle/vehicle.hpp"

namespace {

using simcore::ConstantBodyTorque;
using simcore::Epoch;
using simcore::MassProperties;
using simcore::PropellantTank;
using simcore::ReactionWheel;
using simcore::Thruster;
using simcore::VehicleCommand;
using simcore::VehicleConfig;
using simcore::VehicleState;

constexpr double kPi = std::numbers::pi;
const Epoch kT0 = Epoch::from_tt_seconds_since_j2000(800'000'000, 0.25);
const simcore::ForceSum kFreeSpace;  // no environment forces
const ConstantBodyTorque kNoTorque{Eigen::Vector3d::Zero()};

// 400 kg dry, I_dry = diag(100, 150, 200); 400 kg tank with zero inertia (point mass at the CoM),
// so I stays constant while burning and the tests isolate one effect at a time.
MassProperties mass_400_400() {
    return MassProperties(400.0, Eigen::Vector3d(100.0, 150.0, 200.0).asDiagonal(),
                          PropellantTank{400.0, Eigen::Matrix3d::Zero()});
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Full burn vs the rocket equation (the M2 done-when check)
// ---------------------------------------------------------------------------------------------
// 1000 N, Isp 300 s along body +x through the CoM, m0 = 800 kg -> mf = 400 kg, free space.
//   v_e = 2941.995 m/s, m_dot = 0.33990540 kg/s, burn time t_b = 400 / m_dot = 1176.798 s
//   dv = v_e ln 2 = 2039.2355 m/s
//   distance x(t) = v_e [t + (m/m_dot) ln(m/m0)] (integral of v_e ln(m0/m)) = 1,062,365.53 m
// The body is turned 30 deg about z, so thrust points along (cos 30, sin 30, 0) in ECI: this also
// checks that body-frame thrust is rotated by q.
// Error: RK4 truncation on these smooth curves is < 1e-9 m/s here (v's 5th derivative is
// ~1e-11 m/s^6); roundoff over 11,768 steps of h = 0.1 s is ~steps * eps * |v| = 5e-9 m/s and
// ~steps * eps * |x| = 3e-6 m. Bounds: 1e-6 m/s and 1e-3 m.
TEST(SixDof, FullBurnMatchesRocketEquation) {
    const VehicleConfig cfg{
        mass_400_400(),
        {Thruster(1000.0, 300.0, Eigen::Vector3d(-1.0, 0.0, 0.0), Eigen::Vector3d::UnitX())},
        {}};
    const Eigen::Quaterniond q(Eigen::AngleAxisd(kPi / 6.0, Eigen::Vector3d::UnitZ()));
    const VehicleState s0{kT0, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                          q,   Eigen::Vector3d::Zero(), 400.0,
                          {}};
    const double ve = 300.0 * simcore::g0;
    const double mdot = 1000.0 / ve;
    const double tb = 400.0 / mdot;
    EXPECT_NEAR(tb, 1176.798, 1e-6);  // hand value
    const VehicleState s = simcore::propagate_six_dof(cfg, kFreeSpace, kNoTorque,
                                                      VehicleCommand{{1.0}, {}}, s0, kT0 + tb, 0.1);
    const Eigen::Vector3d dir(std::cos(kPi / 6.0), std::sin(kPi / 6.0), 0.0);
    const double dv = ve * std::log(2.0);
    const double x = ve * (tb + (400.0 / mdot) * std::log(0.5));
    EXPECT_NEAR(dv, 2039.2355, 1e-4);
    EXPECT_NEAR(x, 1062365.53, 0.01);
    EXPECT_LE((s.v_eci - dv * dir).norm(), 1e-6);
    EXPECT_LE((s.r_eci - x * dir).norm(), 1e-3);
    EXPECT_GE(s.propellant_kg, 0.0);
    EXPECT_NEAR(s.propellant_kg, 0.0, 1e-9);
    EXPECT_LE(s.w_body.norm(), 1e-15);  // thrust through the CoM: no torque
}

// Burning past burnout: thrust stops when the tank is empty and never drives m_p negative. With a
// fixed step the cutoff falls inside a step, so the step containing burnout carries at most one
// step of extra thrust: |dv error| <= (T / m_f) h = (1000 / 400) * 0.1 = 0.25 m/s.
// After burnout v is constant (no forces): bit-identical between t = 1190 s and 1200 s.
TEST(SixDof, ThrustStopsAtBurnoutAndPropellantNeverGoesNegative) {
    const VehicleConfig cfg{
        mass_400_400(),
        {Thruster(1000.0, 300.0, Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitX())},
        {}};
    const VehicleState s0{kT0,
                          Eigen::Vector3d::Zero(),
                          Eigen::Vector3d::Zero(),
                          Eigen::Quaterniond::Identity(),
                          Eigen::Vector3d::Zero(),
                          400.0,
                          {}};
    const VehicleCommand burn{{1.0}, {}};
    const VehicleState a =
        simcore::propagate_six_dof(cfg, kFreeSpace, kNoTorque, burn, s0, kT0 + 1190.0, 0.1);
    const VehicleState b =
        simcore::propagate_six_dof(cfg, kFreeSpace, kNoTorque, burn, a, kT0 + 1200.0, 0.1);
    EXPECT_EQ(a.propellant_kg, 0.0);
    EXPECT_EQ(b.propellant_kg, 0.0);
    EXPECT_EQ(b.v_eci, a.v_eci);
    EXPECT_NEAR(b.v_eci.x(), 300.0 * simcore::g0 * std::log(2.0), 0.25);
}

// ---------------------------------------------------------------------------------------------
// Coupling between translation, attitude, mass and wheels
// ---------------------------------------------------------------------------------------------

// Off-CoM thruster: 10 N along +x at r = (0, 0.5, 0) gives tau = r x F = (0, 0, -5) N m. With
// I_zz = 200 constant (zero tank inertia) and w along a principal axis, w_z = -5 t / 200
// = -0.25 rad/s after 10 s, exactly linear (RK4 integrates it exactly; roundoff only: 1e-13).
TEST(SixDof, OffCentreThrusterSpinsTheVehicle) {
    const VehicleConfig cfg{
        mass_400_400(),
        {Thruster(10.0, 300.0, Eigen::Vector3d(0.0, 0.5, 0.0), Eigen::Vector3d::UnitX())},
        {}};
    const VehicleState s0{kT0,
                          Eigen::Vector3d::Zero(),
                          Eigen::Vector3d::Zero(),
                          Eigen::Quaterniond::Identity(),
                          Eigen::Vector3d::Zero(),
                          400.0,
                          {}};
    const VehicleState s = simcore::propagate_six_dof(
        cfg, kFreeSpace, kNoTorque, VehicleCommand{{1.0}, {}}, s0, kT0 + 10.0, 0.01);
    EXPECT_NEAR(s.w_body.z(), -0.25, 1e-13);
    EXPECT_NEAR(s.w_body.x(), 0.0, 1e-15);
    EXPECT_NEAR(s.w_body.y(), 0.0, 1e-15);
}

// Reaction wheel momentum exchange, vehicle initially at rest, z wheel (J = 0.1 kg m^2), command
// +0.1 N m for 100 s (below the 0.2 N m limit): h = 10 N m s exactly, and total H stays zero, so
// I_s w = -h z: w_z = -10 / (200 - 0.1) = -0.050025 rad/s (the body turns opposite to the wheel).
// h_dot is constant, so RK4 is exact for h; w follows algebraically: roundoff bounds 1e-12.
TEST(SixDof, WheelSpinUpCounterRotatesTheBody) {
    const VehicleConfig cfg{
        mass_400_400(), {}, {ReactionWheel(Eigen::Vector3d::UnitZ(), 0.1, 0.2, 600.0)}};
    const VehicleState s0{kT0,
                          Eigen::Vector3d::Zero(),
                          Eigen::Vector3d::Zero(),
                          Eigen::Quaterniond::Identity(),
                          Eigen::Vector3d::Zero(),
                          200.0,
                          {0.0}};
    const VehicleState s = simcore::propagate_six_dof(
        cfg, kFreeSpace, kNoTorque, VehicleCommand{{}, {0.1}}, s0, kT0 + 100.0, 0.1);
    ASSERT_EQ(s.wheel_h_nms.size(), 1u);
    EXPECT_NEAR(s.wheel_h_nms[0], 10.0, 1e-12);
    EXPECT_NEAR(s.w_body.z(), -10.0 / 199.9, 1e-12);
    EXPECT_NEAR(-10.0 / 199.9, -0.050025, 1e-6);  // hand value
}

// Saturation: commanding the 0.2 N m limit for 400 s would store 80 N m s, but the wheel's speed
// relative to the counter-rotating body, h/J - a.w = h/J + h/I_s, reaches 600 rad/s at
// h_sat = 600 / (1/0.1 + 1/199.9) = 59.97 N m s. Within one step of reaching it the motor stops,
// so h overshoots by at most tau_max * h_step = 0.02 N m s.
TEST(SixDof, WheelSaturatesAtItsSpeedLimit) {
    const VehicleConfig cfg{
        mass_400_400(), {}, {ReactionWheel(Eigen::Vector3d::UnitZ(), 0.1, 0.2, 600.0)}};
    const VehicleState s0{kT0,
                          Eigen::Vector3d::Zero(),
                          Eigen::Vector3d::Zero(),
                          Eigen::Quaterniond::Identity(),
                          Eigen::Vector3d::Zero(),
                          200.0,
                          {0.0}};
    const VehicleState s = simcore::propagate_six_dof(
        cfg, kFreeSpace, kNoTorque, VehicleCommand{{}, {0.5}}, s0, kT0 + 400.0, 0.1);
    const double h_sat = 600.0 / (1.0 / 0.1 + 1.0 / 199.9);
    EXPECT_NEAR(h_sat, 59.97, 1e-9);  // hand value
    EXPECT_NEAR(s.wheel_h_nms[0], h_sat, 0.02);
}

// Three skewed wheels, a tumbling body and arbitrary motor commands, no external torque: the total
// inertial angular momentum R(q) (I_s w + sum h a) is conserved. Rates ~0.3 rad/s, h = 0.05 s,
// 100 s: RK4 attitude error ~ steps * 2 (0.3 * 0.05 / 2)^5 / 120 ~ 4e-13; bound 1e-9 relative
// (allowing for the wheel-coupled dynamics being stiffer than the bare body).
TEST(SixDof, WheelsConserveTotalInertialAngularMomentum) {
    const VehicleConfig cfg{mass_400_400(),
                            {},
                            {ReactionWheel(Eigen::Vector3d(1.0, 0.1, 0.0), 0.05, 0.2, 600.0),
                             ReactionWheel(Eigen::Vector3d(0.0, 1.0, 0.3), 0.07, 0.2, 600.0),
                             ReactionWheel(Eigen::Vector3d(0.2, -0.1, 1.0), 0.09, 0.2, 600.0)}};
    const VehicleState s0{
        kT0,
        Eigen::Vector3d::Zero(),
        Eigen::Vector3d::Zero(),
        Eigen::Quaterniond(Eigen::AngleAxisd(0.4, Eigen::Vector3d(1, 2, 3).normalized())),
        Eigen::Vector3d(0.2, -0.3, 0.1),
        200.0,
        {1.0, -0.5, 2.0}};
    const VehicleState s = simcore::propagate_six_dof(
        cfg, kFreeSpace, kNoTorque, VehicleCommand{{}, {0.05, -0.08, 0.03}}, s0, kT0 + 100.0, 0.05);
    const auto H = [&cfg](const VehicleState& x) {
        const Eigen::Vector3d Hb = simcore::total_angular_momentum_body(
            cfg.mass.inertia_kgm2(x.propellant_kg), cfg.wheels, x.w_body, x.wheel_h_nms);
        return Eigen::Vector3d(x.q_body_to_eci.toRotationMatrix() * Hb);
    };
    EXPECT_LE((H(s) - H(s0)).norm(), 1e-9 * H(s0).norm());
    EXPECT_NEAR(s.q_body_to_eci.norm(), 1.0, 4.0 * std::numeric_limits<double>::epsilon());
}

// Uncoupled limit: with no thrust and no wheels, the 6-DOF propagator is just orbit RK4 plus
// attitude RK4 run side by side, so it must agree with propagate_fixed (RK4) and
// propagate_attitude at the same step to roundoff (1e-12 relative).
TEST(SixDof, CoastReducesToOrbitAndAttitudePropagators) {
    const VehicleConfig cfg{mass_400_400(), {}, {}};
    const simcore::PointMassGravity g;
    const simcore::State orbit0 = orbit_fixtures::circular_initial_state();
    const VehicleState s0{orbit0.t,
                          orbit0.r_eci,
                          orbit0.v_eci,
                          Eigen::Quaterniond(Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitY())),
                          Eigen::Vector3d(0.01, 0.2, -0.05),
                          100.0,
                          {}};
    const Epoch t_end = orbit0.t + 600.0;
    const VehicleState s =
        simcore::propagate_six_dof(cfg, g, kNoTorque, VehicleCommand{{}, {}}, s0, t_end, 1.0);
    const simcore::State o =
        simcore::propagate_fixed(g, orbit0, t_end, 1.0, simcore::FixedStepMethod::kRk4);
    const simcore::AttitudeState a = simcore::propagate_attitude(
        cfg.mass.rigid_body(100.0), kNoTorque,
        simcore::AttitudeState{orbit0.t, s0.q_body_to_eci, s0.w_body}, t_end, 1.0);
    EXPECT_LE((s.r_eci - o.r_eci).norm(), 1e-12 * o.r_eci.norm());
    EXPECT_LE((s.v_eci - o.v_eci).norm(), 1e-12 * o.v_eci.norm());
    EXPECT_LE((s.w_body - a.w_body).norm(), 1e-12 * a.w_body.norm());
    EXPECT_LE(s.q_body_to_eci.angularDistance(a.q_body_to_eci), 1e-12);
    EXPECT_EQ(s.t, t_end);
}

// ---------------------------------------------------------------------------------------------
// Bookkeeping
// ---------------------------------------------------------------------------------------------

TEST(SixDof, ValidatesInputsAndIsDeterministic) {
    const VehicleConfig cfg{
        mass_400_400(),
        {Thruster(10.0, 300.0, Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitX())},
        {ReactionWheel(Eigen::Vector3d::UnitZ(), 0.1, 0.2, 600.0)}};
    const VehicleState s0{kT0,
                          Eigen::Vector3d::Zero(),
                          Eigen::Vector3d::Zero(),
                          Eigen::Quaterniond::Identity(),
                          Eigen::Vector3d(0.0, 0.0, 0.1),
                          200.0,
                          {0.0}};
    const VehicleCommand ok{{0.5}, {0.1}};
    const auto run = [&](const VehicleCommand& c, const VehicleState& s, double h) {
        return simcore::propagate_six_dof(cfg, kFreeSpace, kNoTorque, c, s, kT0 + 10.0, h);
    };
    EXPECT_THROW((void)run(VehicleCommand{{}, {0.1}}, s0, 0.1),
                 std::invalid_argument);  // throttles
    EXPECT_THROW((void)run(VehicleCommand{{0.5}, {}}, s0, 0.1), std::invalid_argument);  // wheels
    EXPECT_THROW((void)run(VehicleCommand{{1.5}, {0.1}}, s0, 0.1),
                 std::invalid_argument);  // throttle
    VehicleState bad = s0;
    bad.wheel_h_nms = {};
    EXPECT_THROW((void)run(ok, bad, 0.1), std::invalid_argument);  // h size
    bad = s0;
    bad.propellant_kg = 500.0;
    EXPECT_THROW((void)run(ok, bad, 0.1), std::invalid_argument);  // > capacity
    EXPECT_THROW((void)run(ok, s0, 0.0), std::invalid_argument);   // h_max
    // Zero duration is a no-op; reruns are bit-identical.
    const VehicleState same =
        simcore::propagate_six_dof(cfg, kFreeSpace, kNoTorque, ok, s0, kT0, 0.1);
    EXPECT_EQ(same.w_body, s0.w_body);
    EXPECT_EQ(same.propellant_kg, s0.propellant_kg);
    const VehicleState a = run(ok, s0, 0.1);
    const VehicleState b = run(ok, s0, 0.1);
    EXPECT_EQ(a.r_eci, b.r_eci);
    EXPECT_EQ(a.w_body, b.w_body);
    EXPECT_EQ(a.q_body_to_eci.coeffs(), b.q_body_to_eci.coeffs());
    EXPECT_EQ(a.wheel_h_nms, b.wheel_h_nms);
}

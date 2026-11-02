// M3.4: manual control, mapping pilot input to a VehicleCommand.
//
// Throttle goes to the main engine (thruster 0); other thrusters stay off.
// Rotation is by the reaction wheels. A motor torque tau_m on wheel j turns the body by -tau_m a_j
// (reaction), so a wanted body torque tau_b needs wheel torques tau_m = -A+ tau_b, where
// A = [a_1 ... a_n] and A+ is its pseudo-inverse (minimum-norm allocation). If any wheel would
// exceed its torque limit, all wheel torques are scaled down together, keeping the direction.
//   - Rotate input u in [-1, 1]^3 (body x, y, z = roll, pitch, yaw): tau_b = sum_k u_k T_k e_k,
//     where T_k is the largest body torque the wheels can make about axis k alone.
//   - Rate damping (when no rotate input): tau_b = -k I w with k = 0.5 /s, so w_dot = -k w
//     (ignoring the small gyroscopic term): rates decay with a 2 s time constant, slower while
//     the wheels are torque-limited. At or below the rails threshold the command is exactly zero,
//     so a damped vehicle can go on rails (time warp).

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "simcore/constants.hpp"
#include "simcore/dynamics/force_sum.hpp"
#include "simcore/dynamics/point_mass_gravity.hpp"
#include "simcore/sim/manual_control.hpp"
#include "simcore/sim/simulation.hpp"
#include "simcore/vehicle/reaction_wheel.hpp"

namespace {

using simcore::ManualInput;
using simcore::Simulation;
using simcore::VehicleCommand;
using simcore::VehicleConfig;
using simcore::VehicleState;

// I = diag(300, 400, 450) with an empty tank (no inertia change), main engine + a second thruster,
// three wheels on the body axes (J = 0.05, 0.1 N m, 600 rad/s).
VehicleConfig test_vehicle() {
    return VehicleConfig{
        simcore::MassProperties(400.0, Eigen::Vector3d(300.0, 400.0, 450.0).asDiagonal(),
                                simcore::PropellantTank{400.0, Eigen::Matrix3d::Zero()}),
        {simcore::Thruster(1000.0, 300.0, Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitX()),
         simcore::Thruster(10.0, 300.0, Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitY())},
        {simcore::ReactionWheel(Eigen::Vector3d::UnitX(), 0.05, 0.1, 600.0),
         simcore::ReactionWheel(Eigen::Vector3d::UnitY(), 0.05, 0.1, 600.0),
         simcore::ReactionWheel(Eigen::Vector3d::UnitZ(), 0.05, 0.1, 600.0)}};
}

VehicleState at_rest() {
    return VehicleState{simcore::Epoch::from_tt_seconds_since_j2000(800'000'000),
                        Eigen::Vector3d(7.0e6, 0.0, 0.0),
                        Eigen::Vector3d(0.0, 7.5e3, 0.0),
                        Eigen::Quaterniond::Identity(),
                        Eigen::Vector3d::Zero(),
                        200.0,
                        {0.0, 0.0, 0.0}};
}

Eigen::Vector3d wheel_vector(const VehicleCommand& c) {
    return Eigen::Vector3d(c.wheel_torque_nm[0], c.wheel_torque_nm[1], c.wheel_torque_nm[2]);
}

}  // namespace

// Throttle drives the main engine only; no rotation input and no damping means wheels idle.
TEST(ManualControl, ThrottleDrivesTheMainEngineOnly) {
    const VehicleCommand c = simcore::manual_command(
        test_vehicle(), ManualInput{0.7, Eigen::Vector3d::Zero(), false}, at_rest());
    EXPECT_EQ(c.throttle, (std::vector<double>{0.7, 0.0}));
    EXPECT_EQ(c.wheel_torque_nm, (std::vector<double>{0.0, 0.0, 0.0}));
}

// Full pitch-up (u_y = +1): the wheels can make 0.1 N m about y, so tau_b = (0, 0.1, 0) and the y
// wheel is driven the opposite way, tau_m = (0, -0.1, 0). Sign check through the dynamics: with
// that motor torque the body's y rate accelerates positively, w_dot_y = 0.1 / I_s,yy
// = 0.1 / (400 - 0.05) (I_s removes the wheel's own spin inertia), and x, z stay at 0.
// Half input gives half torque.
TEST(ManualControl, RotateInputTurnsTheBodyTheRightWay) {
    const VehicleConfig cfg = test_vehicle();
    const VehicleState s = at_rest();
    const VehicleCommand c =
        simcore::manual_command(cfg, ManualInput{0.0, Eigen::Vector3d(0.0, 1.0, 0.0), false}, s);
    EXPECT_LE((wheel_vector(c) - Eigen::Vector3d(0.0, -0.1, 0.0)).norm(), 1e-15);
    const Eigen::Vector3d w_dot = simcore::angular_acceleration_with_wheels(
        cfg.mass.inertia_kgm2(s.propellant_kg), cfg.wheels, s.w_body, s.wheel_h_nms,
        Eigen::Vector3d::Zero(), c.wheel_torque_nm);
    EXPECT_NEAR(w_dot.y(), 0.1 / (400.0 - 0.05), 1e-15);
    EXPECT_EQ(w_dot.x(), 0.0);
    EXPECT_EQ(w_dot.z(), 0.0);
    const VehicleCommand half =
        simcore::manual_command(cfg, ManualInput{0.0, Eigen::Vector3d(0.0, 0.5, 0.0), false}, s);
    EXPECT_LE((wheel_vector(half) - Eigen::Vector3d(0.0, -0.05, 0.0)).norm(), 1e-15);
}

// Rate damping, unsaturated: w = (1e-3, 0, 0) gives tau_b = -0.5 * 300 * 1e-3 = -0.15 N m, over
// the 0.1 limit, so it is scaled to -0.1; w = (3e-4, 0, 0) gives -0.045 N m, within the limit,
// so tau_m = +0.045 (the x wheel spins up to take the body's momentum).
// Saturated and coupled: w = (0.01, -0.02, 0.005) wants tau_b = -k I w = (-1.5, 4, -1.125), so
// tau_m = (1.5, -4, 1.125); scaled by 0.1 / 4 to keep the direction: (0.0375, -0.1, 0.028125).
TEST(ManualControl, RateDampingOpposesTheRotationWithinTheTorqueLimit) {
    const VehicleConfig cfg = test_vehicle();
    VehicleState s = at_rest();
    const ManualInput damp{0.0, Eigen::Vector3d::Zero(), true};
    s.w_body = Eigen::Vector3d(1e-3, 0.0, 0.0);
    EXPECT_LE(
        (wheel_vector(simcore::manual_command(cfg, damp, s)) - Eigen::Vector3d(0.1, 0, 0)).norm(),
        1e-15);
    s.w_body = Eigen::Vector3d(3e-4, 0.0, 0.0);
    EXPECT_LE(
        (wheel_vector(simcore::manual_command(cfg, damp, s)) - Eigen::Vector3d(0.045, 0, 0)).norm(),
        1e-15);
    s.w_body = Eigen::Vector3d(0.01, -0.02, 0.005);
    EXPECT_LE((wheel_vector(simcore::manual_command(cfg, damp, s)) -
               Eigen::Vector3d(0.0375, -0.1, 0.028125))
                  .norm(),
              1e-15);
}

// Damping hands over to the rails threshold: at |w| <= kRailsMaxRate the wheel command is exactly
// zero. A rotate input takes precedence over damping.
TEST(ManualControl, DampingIsExactlyZeroAtTheRailsThresholdAndRotateWins) {
    const VehicleConfig cfg = test_vehicle();
    VehicleState s = at_rest();
    s.w_body = Eigen::Vector3d(0.0, 0.0, 0.9 * Simulation::kRailsMaxRate_radps);
    EXPECT_EQ(simcore::manual_command(cfg, ManualInput{0.0, Eigen::Vector3d::Zero(), true}, s)
                  .wheel_torque_nm,
              (std::vector<double>{0.0, 0.0, 0.0}));
    s.w_body = Eigen::Vector3d(0.01, 0.0, 0.0);
    const VehicleCommand c =
        simcore::manual_command(cfg, ManualInput{0.0, Eigen::Vector3d(0.0, 0.0, -1.0), true}, s);
    EXPECT_LE((wheel_vector(c) - Eigen::Vector3d(0.0, 0.0, 0.1)).norm(), 1e-15);
}

// Closed loop: damping a tumble w0 = (0.02, -0.01, 0.015) rad/s, with the command updated every
// 5 ticks (0.1 s, a 50 fps front end). Body momentum I w0 = (6, -4, 6.75) N m s, so the z wheel
// (largest) at its 0.1 N m limit needs ~67.5 s, with the others scaled in proportion; then the
// unsaturated tail decays at k = 0.5 /s from |w| ~ 4e-4 to 1e-4 in ~3 s. Bound: within 120 s the
// rates are at the rails threshold and the command is exactly zero. The body's momentum moves into
// the wheels: total H is conserved in inertial space, and the body turns ~0.7 rad while damping, so
// the per-wheel split mixes but the wheels' total |sum h a| ends at |I w0| = 9.877 N m s, less the
// residual |I w| <= 450 * 1e-4 = 0.045: bound 0.05. That is well below J Omega_max = 30 per wheel.
TEST(ManualControl, RateDampingStopsATumbleAndReleasesToRails) {
    const VehicleConfig cfg = test_vehicle();
    VehicleState s0 = at_rest();
    s0.w_body = Eigen::Vector3d(0.02, -0.01, 0.015);
    Simulation sim(
        simcore::SimConfig{cfg, std::make_shared<simcore::ForceSum>(),
                           std::make_shared<simcore::ConstantBodyTorque>(Eigen::Vector3d::Zero()),
                           0.02, std::nullopt},
        s0);
    const ManualInput damp{0.0, Eigen::Vector3d::Zero(), true};
    while (sim.snapshot().state.t - s0.t < 120.0) {
        sim.submit(simcore::manual_command(cfg, damp, sim.snapshot().state));
        sim.step(5);
    }
    const VehicleState s = sim.snapshot().state;
    EXPECT_LE(s.w_body.norm(), Simulation::kRailsMaxRate_radps);
    EXPECT_EQ(simcore::manual_command(cfg, damp, s).wheel_torque_nm,
              (std::vector<double>{0.0, 0.0, 0.0}));
    const Eigen::Vector3d h_wheels(s.wheel_h_nms[0], s.wheel_h_nms[1], s.wheel_h_nms[2]);
    const double H0 = (cfg.mass.inertia_kgm2(s0.propellant_kg) * s0.w_body).norm();
    EXPECT_NEAR(H0, 9.877, 1e-3);  // hand value
    EXPECT_NEAR(h_wheels.norm(), H0, 0.05);
}

TEST(ManualControl, RejectsInvalidInput) {
    const VehicleConfig cfg = test_vehicle();
    const VehicleState s = at_rest();
    const double kNaN = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(
        (void)simcore::manual_command(cfg, ManualInput{1.5, Eigen::Vector3d::Zero(), false}, s),
        std::invalid_argument);
    EXPECT_THROW((void)simcore::manual_command(
                     cfg, ManualInput{0.0, Eigen::Vector3d(0.0, 2.0, 0.0), false}, s),
                 std::invalid_argument);
    EXPECT_THROW((void)simcore::manual_command(
                     cfg, ManualInput{0.0, Eigen::Vector3d(kNaN, 0.0, 0.0), false}, s),
                 std::invalid_argument);
}

// Non-orthogonal, redundant wheels: a 4-wheel pyramid, axes (+-s, 0, c) and (0, +-s, c) with
// s = sin 30 deg, c = cos 30 deg (A has rank 3, so A A+ = I). Full pitch: the body torque
// -A tau_m is exactly along +y, and the most-loaded wheel is at its 0.1 N m limit (the pseudo-
// inverse authority T_y; the null space could add more, which this allocation doesn't use).
TEST(ManualControl, PyramidWheelsGiveTorqueAboutTheRequestedAxis) {
    const double sn = 0.5;
    const double cs = std::sqrt(3.0) / 2.0;
    VehicleConfig cfg = test_vehicle();
    cfg.wheels = {simcore::ReactionWheel(Eigen::Vector3d(sn, 0.0, cs), 0.05, 0.1, 600.0),
                  simcore::ReactionWheel(Eigen::Vector3d(-sn, 0.0, cs), 0.05, 0.1, 600.0),
                  simcore::ReactionWheel(Eigen::Vector3d(0.0, sn, cs), 0.05, 0.1, 600.0),
                  simcore::ReactionWheel(Eigen::Vector3d(0.0, -sn, cs), 0.05, 0.1, 600.0)};
    VehicleState s = at_rest();
    s.wheel_h_nms = {0.0, 0.0, 0.0, 0.0};
    const VehicleCommand c =
        simcore::manual_command(cfg, ManualInput{0.0, Eigen::Vector3d(0.0, 1.0, 0.0), false}, s);
    Eigen::Vector3d tau_b = Eigen::Vector3d::Zero();
    double peak = 0.0;
    for (std::size_t j = 0; j < cfg.wheels.size(); ++j) {
        tau_b -= c.wheel_torque_nm[j] * cfg.wheels[j].axis_body();
        peak = std::max(peak, std::abs(c.wheel_torque_nm[j]));
    }
    EXPECT_GT(tau_b.y(), 0.0);
    EXPECT_LE(std::abs(tau_b.x()), 1e-15);
    EXPECT_LE(std::abs(tau_b.z()), 1e-15);
    EXPECT_NEAR(peak, 0.1, 1e-15);
}

// Sample-and-hold stability (M3 review): the damping command tau = -k I w held for a time T gives
// the discrete loop w+ = (1 - k T) w, stable only for k T < 2. A front end at 10 fps and 100x warp
// holds each command for 10 s (k T = 5): it never settles (a limit cycle near tau_max T / I
// = 0.1 * 10 / 450 = 2.2e-3 rad/s). fly_manual recomputes the command at least every
// kManualHoldMax_s = 0.5 s (k T = 0.25, a monotone decay) whatever the frame: from w_z = 2e-3 the
// wheel at its 0.1 N m limit needs 2e-3 * 450 / 0.1 = 9 s, then the tail decays at 22% per hold,
// so the vehicle is on rails well within 20 frames (200 s). Holding for a whole frame instead
// (hold = 500 ticks) is still not on rails after the same 20 frames.
TEST(ManualControl, FlyManualKeepsDampingStableAtSlowFramesAndHighWarp) {
    const VehicleConfig cfg = test_vehicle();
    VehicleState s0 = at_rest();
    s0.w_body = Eigen::Vector3d(0.0, 0.0, 2e-3);
    const auto make = [&]() {
        return Simulation(simcore::SimConfig{cfg, std::make_shared<simcore::PointMassGravity>(),
                                             std::make_shared<simcore::ConstantBodyTorque>(
                                                 Eigen::Vector3d::Zero()),
                                             0.02, simcore::earth::mu},
                          s0);
    };
    const ManualInput damp{0.0, Eigen::Vector3d::Zero(), true};
    Simulation bounded = make();
    Simulation per_frame = make();
    for (int frame = 0; frame < 20; ++frame) {
        simcore::fly_manual(bounded, damp, 500);  // 0.1 s frame at 100x = 500 ticks
        simcore::fly_manual(per_frame, damp, 500, 500);
    }
    EXPECT_TRUE(bounded.snapshot().on_rails);
    EXPECT_FALSE(per_frame.snapshot().on_rails);
    EXPECT_GT(per_frame.snapshot().state.w_body.norm(), Simulation::kRailsMaxRate_radps);
    EXPECT_EQ(simcore::kManualHoldMax_s, 0.5);
}

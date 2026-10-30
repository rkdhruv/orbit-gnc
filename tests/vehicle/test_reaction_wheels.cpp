// M2.3: reaction wheels and the rotational equations of a spacecraft carrying them.
//
// Model (exact for axisymmetric wheels; Markley & Crassidis section 3.3.5). I is the whole
// vehicle's inertia with the wheels locked. Wheel i has spin axis a_i (body), axial inertia J_i and
// absolute axial angular momentum h_i (the wheel's own spin plus the body rate along a_i).
//   I_s = I - sum J_i a_i a_i^T              (inertia excluding the wheels' axial spin)
//   H   = I_s w + sum h_i a_i                (total angular momentum, body frame)
//   I_s w_dot = tau_ext - w x H - sum tau_i a_i     (body: the motor torque reacts on it)
//   h_i_dot   = tau_i                                (wheel: the motor torque spins it up)
// so with no external torque dH/dt|_inertial = 0: the wheels only move momentum around.
// Wheel speed relative to the body: Omega_i = h_i / J_i - a_i . w.
// Saturation: |tau_i| <= tau_max, and no motor torque that would push |Omega_i| past Omega_max.

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <vector>

#include <Eigen/Geometry>

#include "simcore/vehicle/reaction_wheel.hpp"

namespace {

using simcore::ReactionWheel;

constexpr double kRel = 1e-14;
const Eigen::Matrix3d kI = Eigen::Vector3d(10.0, 20.0, 30.0).asDiagonal();  // wheels locked

}  // namespace

// ---------------------------------------------------------------------------------------------
// Single wheel: speed, saturation
// ---------------------------------------------------------------------------------------------

// J = 0.1 kg m^2 on body z, h = 1 N m s, body spinning at w_z = 0.5 rad/s:
// Omega = h/J - a.w = 10 - 0.5 = 9.5 rad/s. Max momentum J Omega_max = 0.1 * 600 = 60 N m s.
TEST(ReactionWheel, RelativeSpeedAndCapacity) {
    const ReactionWheel w(Eigen::Vector3d(0.0, 0.0, 3.0), 0.1, 0.2, 600.0);  // axis normalised
    EXPECT_EQ(w.axis_body(), Eigen::Vector3d::UnitZ());
    EXPECT_NEAR(w.relative_speed_radps(1.0, Eigen::Vector3d(0.3, -0.1, 0.5)), 9.5, kRel * 10.0);
    EXPECT_NEAR(w.max_momentum_nms(), 60.0, kRel * 60.0);
}

// Torque limit 0.2 N m: commands are clipped to +/-0.2. At the speed limit (600 rad/s) a command
// that would spin it faster is refused (0), while one that slows it is allowed.
TEST(ReactionWheel, SaturatesTorqueAndSpeed) {
    const ReactionWheel w(Eigen::Vector3d::UnitZ(), 0.1, 0.2, 600.0);
    EXPECT_EQ(w.applied_motor_torque_nm(0.15, 0.0), 0.15);
    EXPECT_EQ(w.applied_motor_torque_nm(0.5, 0.0), 0.2);
    EXPECT_EQ(w.applied_motor_torque_nm(-0.5, 0.0), -0.2);
    EXPECT_EQ(w.applied_motor_torque_nm(0.1, 600.0), 0.0);    // would exceed +600
    EXPECT_EQ(w.applied_motor_torque_nm(-0.1, 600.0), -0.1);  // spinning down is fine
    EXPECT_EQ(w.applied_motor_torque_nm(-0.1, -600.0), 0.0);  // would exceed -600
    EXPECT_EQ(w.applied_motor_torque_nm(0.1, -600.0), 0.1);
    EXPECT_EQ(w.applied_motor_torque_nm(0.1, 650.0), 0.0);  // already past the limit
}

TEST(ReactionWheel, RejectsInvalidInput) {
    const double kNaN = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((void)ReactionWheel(Eigen::Vector3d::Zero(), 0.1, 0.2, 600.0),
                 std::invalid_argument);
    EXPECT_THROW((void)ReactionWheel(Eigen::Vector3d::UnitZ(), 0.0, 0.2, 600.0),
                 std::invalid_argument);
    EXPECT_THROW((void)ReactionWheel(Eigen::Vector3d::UnitZ(), 0.1, -0.2, 600.0),
                 std::invalid_argument);
    EXPECT_THROW((void)ReactionWheel(Eigen::Vector3d::UnitZ(), 0.1, 0.2, 0.0),
                 std::invalid_argument);
    const ReactionWheel w(Eigen::Vector3d::UnitZ(), 0.1, 0.2, 600.0);
    EXPECT_THROW((void)w.applied_motor_torque_nm(kNaN, 0.0), std::invalid_argument);
}

// ---------------------------------------------------------------------------------------------
// Vehicle rotational dynamics with wheels
// ---------------------------------------------------------------------------------------------

// One z wheel, J = 0.1: I_s = diag(10, 20, 29.9). Body at rest, h = 1, motor torque +0.5 N m:
// H = (0, 0, 1), w x H = 0, so I_s w_dot = -0.5 z -> w_dot = (0, 0, -0.5/29.9) = -0.016722 rad/s^2
// (the body spins the opposite way to the wheel), and h_dot = +0.5.
TEST(WheelDynamics, MotorTorqueReactsOnTheBody) {
    const std::vector<ReactionWheel> wheels{
        ReactionWheel(Eigen::Vector3d::UnitZ(), 0.1, 1.0, 600.0)};
    const Eigen::Matrix3d Is = simcore::inertia_without_wheel_spin(kI, wheels);
    EXPECT_LE((Is - Eigen::Matrix3d(Eigen::Vector3d(10.0, 20.0, 29.9).asDiagonal())).norm(),
              kRel * 30.0);
    const Eigen::Vector3d wd = simcore::angular_acceleration_with_wheels(
        kI, wheels, Eigen::Vector3d::Zero(), std::vector<double>{1.0}, Eigen::Vector3d::Zero(),
        std::vector<double>{0.5});
    EXPECT_NEAR(wd.z(), -0.5 / 29.9, kRel);
    EXPECT_NEAR(-0.5 / 29.9, -0.016722, 1e-6);  // hand value
    EXPECT_EQ(wd.x(), 0.0);
    EXPECT_EQ(wd.y(), 0.0);
}

// Stored wheel momentum gives gyroscopic stiffness: body rate w = (0.1, 0, 0), z wheel with h = 2,
// no torques. I_s w = (1, 0, 0), H = (1, 0, 2), w x H = (0, -0.2, 0), so
// w_dot = -I_s^-1 (0, -0.2, 0) = (0, 0.01, 0): the body precesses about y.
TEST(WheelDynamics, StoredMomentumCouplesAxesGyroscopically) {
    const std::vector<ReactionWheel> wheels{
        ReactionWheel(Eigen::Vector3d::UnitZ(), 0.1, 1.0, 600.0)};
    const Eigen::Vector3d w(0.1, 0.0, 0.0);
    const std::vector<double> h{2.0};
    EXPECT_LE(
        (simcore::total_angular_momentum_body(kI, wheels, w, h) - Eigen::Vector3d(1.0, 0.0, 2.0))
            .norm(),
        kRel * 2.0);
    const Eigen::Vector3d wd = simcore::angular_acceleration_with_wheels(
        kI, wheels, w, h, Eigen::Vector3d::Zero(), std::vector<double>{0.0});
    EXPECT_NEAR(wd.x(), 0.0, kRel);
    EXPECT_NEAR(wd.y(), 0.01, kRel);
    EXPECT_NEAR(wd.z(), 0.0, kRel);
}

// The equations are a statement of dH/dt|_inertial = tau_ext. In body axes,
//   dH/dt|_body + w x H = I_s w_dot + sum h_dot_i a_i + w x H  must equal tau_ext
// for any state: checked with three skewed wheels, a full (non-diagonal) inertia, arbitrary rates,
// wheel momenta, motor torques and an external torque. Roundoff only: 1e-13 relative.
TEST(WheelDynamics, SatisfiesAngularMomentumBalance) {
    const Eigen::Matrix3d R =
        Eigen::AngleAxisd(0.7, Eigen::Vector3d(1.0, 2.0, 3.0).normalized()).toRotationMatrix();
    const Eigen::Matrix3d I = R * kI * R.transpose();
    const std::vector<ReactionWheel> wheels{
        ReactionWheel(Eigen::Vector3d(1.0, 0.1, 0.0), 0.05, 0.2, 600.0),
        ReactionWheel(Eigen::Vector3d(0.0, 1.0, 0.3), 0.07, 0.2, 600.0),
        ReactionWheel(Eigen::Vector3d(0.2, -0.1, 1.0), 0.09, 0.2, 600.0)};
    const Eigen::Vector3d w(0.03, -0.2, 0.11), tau_ext(0.4, -0.25, 0.1);
    const std::vector<double> h{1.5, -0.7, 2.2}, tau_m{0.12, -0.05, 0.08};
    const Eigen::Vector3d wd =
        simcore::angular_acceleration_with_wheels(I, wheels, w, h, tau_ext, tau_m);
    Eigen::Vector3d hdot_sum = Eigen::Vector3d::Zero();
    for (std::size_t i = 0; i < wheels.size(); ++i) {
        hdot_sum += tau_m[i] * wheels[i].axis_body();
    }
    const Eigen::Vector3d H = simcore::total_angular_momentum_body(I, wheels, w, h);
    const Eigen::Vector3d balance =
        simcore::inertia_without_wheel_spin(I, wheels) * wd + hdot_sum + w.cross(H);
    EXPECT_LE((balance - tau_ext).norm(), 1e-13 * tau_ext.norm());
}

// Without wheels the model reduces to Euler's equations exactly (same as RigidBody).
TEST(WheelDynamics, NoWheelsReducesToEulerEquations) {
    const Eigen::Vector3d w(1.0, 2.0, 3.0);
    const Eigen::Vector3d wd = simcore::angular_acceleration_with_wheels(
        kI, {}, w, std::vector<double>{}, Eigen::Vector3d::Zero(), std::vector<double>{});
    // From M2.2: w_dot = (-6, 3, -2/3) for this body and rate.
    EXPECT_NEAR(wd.x(), -6.0, kRel * 6.0);
    EXPECT_NEAR(wd.y(), 3.0, kRel * 6.0);
    EXPECT_NEAR(wd.z(), -2.0 / 3.0, kRel * 6.0);
}

TEST(WheelDynamics, RejectsInconsistentInput) {
    const std::vector<ReactionWheel> wheels{
        ReactionWheel(Eigen::Vector3d::UnitZ(), 0.1, 1.0, 600.0)};
    // Wrong number of momenta or torques.
    EXPECT_THROW((void)simcore::angular_acceleration_with_wheels(
                     kI, wheels, Eigen::Vector3d::Zero(), std::vector<double>{},
                     Eigen::Vector3d::Zero(), std::vector<double>{0.0}),
                 std::invalid_argument);
    EXPECT_THROW((void)simcore::total_angular_momentum_body(kI, wheels, Eigen::Vector3d::Zero(),
                                                            std::vector<double>{1.0, 2.0}),
                 std::invalid_argument);
    // A wheel whose axial inertia is not smaller than the vehicle's along that axis leaves I_s
    // non-positive-definite: J = 30 on z with I_zz = 30.
    const std::vector<ReactionWheel> huge{
        ReactionWheel(Eigen::Vector3d::UnitZ(), 30.0, 1.0, 600.0)};
    EXPECT_THROW((void)simcore::inertia_without_wheel_spin(kI, huge), std::invalid_argument);
}

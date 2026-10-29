// M2.2: rigid-body inertia, Euler's equations and quaternion kinematics.
//
// Euler's rotational equations in the body frame (Markley & Crassidis eq. 3.81):
//   I w_dot = tau - w x (I w)            (the w x (I w) term is the gyroscopic coupling)
// Angular momentum H = I w (body frame; inertial H = R(q) I w is constant without torque),
// rotational kinetic energy T = 1/2 w^T I w.
// Quaternion kinematics, Hamilton convention, q = body-to-inertial (scalar first in the docs):
//   q_dot = 1/2 q (x) [0, w_body]        (body rates compose on the right)
// A valid inertia tensor is symmetric positive definite with principal moments obeying the
// triangle inequality I_i <= I_j + I_k (a physical mass distribution).

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

#include <Eigen/Geometry>

#include "simcore/dynamics/rigid_body.hpp"

namespace {

using simcore::RigidBody;

constexpr double kRel = 1e-14;

}  // namespace

// Solid box, mass m, edges (a, b, c): I = m/12 diag(b^2 + c^2, a^2 + c^2, a^2 + b^2).
// m = 12 kg, (1, 2, 3) m -> diag(13, 10, 5) kg m^2.
TEST(RigidBody, BoxInertia) {
    const RigidBody box = RigidBody::box(12.0, 1.0, 2.0, 3.0);
    const Eigen::Matrix3d expected = Eigen::Vector3d(13.0, 10.0, 5.0).asDiagonal();
    EXPECT_LE((box.inertia_body() - expected).norm(), kRel * 13.0);
}

// H = I w and T = 1/2 w . I w: I = diag(10, 20, 30), w = (1, 2, 3) -> H = (10, 40, 90),
// T = 1/2 (10 + 80 + 270) = 180 J.
TEST(RigidBody, AngularMomentumAndEnergy) {
    const RigidBody b(Eigen::Vector3d(10.0, 20.0, 30.0).asDiagonal());
    const Eigen::Vector3d w(1.0, 2.0, 3.0);
    EXPECT_EQ(b.angular_momentum_body(w), Eigen::Vector3d(10.0, 40.0, 90.0));
    EXPECT_EQ(b.rotational_kinetic_energy(w), 180.0);
}

// Gyroscopic term by hand: w x (I w) = (1, 2, 3) x (10, 40, 90) = (60, -60, 20), so with no torque
// w_dot = -I^-1 (60, -60, 20) = (-6, 3, -2/3).
TEST(RigidBody, EulerEquationsIncludeGyroscopicTerm) {
    const RigidBody b(Eigen::Vector3d(10.0, 20.0, 30.0).asDiagonal());
    const Eigen::Vector3d wd =
        b.angular_acceleration_body({1.0, 2.0, 3.0}, Eigen::Vector3d::Zero());
    EXPECT_NEAR(wd.x(), -6.0, kRel * 6.0);
    EXPECT_NEAR(wd.y(), 3.0, kRel * 6.0);
    EXPECT_NEAR(wd.z(), -2.0 / 3.0, kRel * 6.0);
    // With torque and w along a principal axis there is no coupling: w_dot = tau / I.
    const Eigen::Vector3d wt =
        b.angular_acceleration_body({0.0, 0.0, 5.0}, Eigen::Vector3d(0.0, 0.0, 6.0));
    EXPECT_NEAR(wt.z(), 0.2, kRel);
    EXPECT_EQ(wt.x(), 0.0);
    EXPECT_EQ(wt.y(), 0.0);
}

// A full (non-diagonal) tensor works the same way: build I' = R I R^T for a rotated body and check
// w_dot' = R w_dot for w' = R w (Euler's equations are frame-covariant).
TEST(RigidBody, NonPrincipalBodyFrameIsCovariant) {
    const Eigen::Matrix3d I = Eigen::Vector3d(10.0, 20.0, 30.0).asDiagonal();
    const Eigen::Matrix3d R =
        Eigen::AngleAxisd(0.7, Eigen::Vector3d(1.0, 2.0, 3.0).normalized()).toRotationMatrix();
    const RigidBody principal(I);
    const RigidBody rotated(R * I * R.transpose());
    const Eigen::Vector3d w(0.3, -1.2, 0.8), tau(0.5, 0.1, -0.4);
    const Eigen::Vector3d a = R * principal.angular_acceleration_body(w, tau);
    const Eigen::Vector3d b = rotated.angular_acceleration_body(R * w, R * tau);
    EXPECT_LE((a - b).norm(), 1e-13 * a.norm());
}

TEST(RigidBody, RejectsUnphysicalInertia) {
    Eigen::Matrix3d asym = Eigen::Vector3d(10.0, 20.0, 30.0).asDiagonal();
    asym(0, 1) = 1.0;  // not symmetric
    EXPECT_THROW((void)RigidBody(asym), std::invalid_argument);
    EXPECT_THROW((void)RigidBody(Eigen::Vector3d(10.0, 0.0, 30.0).asDiagonal()),
                 std::invalid_argument);  // singular
    EXPECT_THROW((void)RigidBody(Eigen::Vector3d(10.0, -5.0, 30.0).asDiagonal()),
                 std::invalid_argument);  // not positive definite
    EXPECT_THROW((void)RigidBody(Eigen::Vector3d(1.0, 1.0, 3.0).asDiagonal()),
                 std::invalid_argument);  // 3 > 1 + 1: no mass distribution has this
    Eigen::Matrix3d nan = Eigen::Matrix3d::Identity();
    nan(2, 2) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((void)RigidBody(nan), std::invalid_argument);
    EXPECT_THROW((void)RigidBody::box(-1.0, 1.0, 1.0, 1.0), std::invalid_argument);
    // A thin rod/plate edge case is allowed: I = (1, 1, 2) satisfies 2 <= 1 + 1 exactly.
    EXPECT_NO_THROW((void)RigidBody(Eigen::Vector3d(1.0, 1.0, 2.0).asDiagonal()));
}

// q_dot = 1/2 q (x) [0, w]: for q = identity it is [0, w/2]. For q = 90 deg about x and w = z the
// factors don't commute, so the order matters: checked against Eigen's Hamilton product q * [0, w],
// and shown to differ from the inertial-rate order [0, w] * q (a classic convention slip).
TEST(QuaternionKinematics, RateIsHalfQTimesBodyRate) {
    const Eigen::Vector3d w(0.2, -0.4, 0.6);
    const Eigen::Quaterniond qi = Eigen::Quaterniond::Identity();
    const Eigen::Quaterniond r = simcore::quaternion_rate(qi, w);
    EXPECT_NEAR(r.w(), 0.0, kRel);
    EXPECT_NEAR(r.x(), 0.1, kRel);
    EXPECT_NEAR(r.y(), -0.2, kRel);
    EXPECT_NEAR(r.z(), 0.3, kRel);

    const Eigen::Quaterniond q(Eigen::AngleAxisd(std::numbers::pi / 2.0, Eigen::Vector3d::UnitX()));
    const Eigen::Quaterniond rq = simcore::quaternion_rate(q, Eigen::Vector3d::UnitZ());
    const Eigen::Quaterniond expected = q * Eigen::Quaterniond(0.0, 0.0, 0.0, 1.0);  // q (x) [0, w]
    EXPECT_NEAR(rq.w(), 0.5 * expected.w(), kRel);
    EXPECT_NEAR(rq.x(), 0.5 * expected.x(), kRel);
    EXPECT_NEAR(rq.y(), 0.5 * expected.y(), kRel);
    EXPECT_NEAR(rq.z(), 0.5 * expected.z(), kRel);
    // The wrong (inertial-rate) order [0, w] (x) q differs here:
    const Eigen::Quaterniond wrong = Eigen::Quaterniond(0.0, 0.0, 0.0, 1.0) * q;
    EXPECT_GT(std::abs(wrong.y() - expected.y()) + std::abs(wrong.z() - expected.z()), 0.5);
}

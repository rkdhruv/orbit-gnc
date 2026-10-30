// M2.2: attitude propagation (quaternion + Euler's equations) with fixed-step RK4.
//
// State y = [q (4); w_body (3)], y' = [1/2 q (x) [0, w]; I^-1 (tau - w x I w)]. One classical RK4
// step on y, then q is renormalised (CLAUDE.md rule), since RK4 does not preserve |q| = 1.
// propagate_attitude uses n = ceil(T / h_max) equal steps, like propagate_fixed. It is attitude
// only: the translational state handed to the torque model is held fixed (coupled motion is
// propagate_six_dof).
//
// Error estimates used below. RK4 on a rotation at rate lambda: per-step relative amplitude error
// ~ (lambda h)^6 / 144 (|R(i lambda h)| = 1 - (lambda h)^6/144 + ...) and phase error
// ~ (lambda h)^5 / 120. With lambda ~ 1 rad/s and h = 0.01 s: 7e-15 and 8e-13 per step.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

#include <Eigen/Geometry>

#include "simcore/dynamics/rigid_body.hpp"
#include "simcore/dynamics/state.hpp"
#include "simcore/integrators/attitude.hpp"

namespace {

using simcore::AttitudeState;
using simcore::ConstantBodyTorque;
using simcore::Epoch;
using simcore::RigidBody;

constexpr double kPi = std::numbers::pi;
const Epoch kT0 = Epoch::from_tt_seconds_since_j2000(800'000'000, 0.25);
const ConstantBodyTorque kNoTorque{Eigen::Vector3d::Zero()};
// Frozen translational state for the torque model (unused by the torques below unless stated).
const simcore::State kOrbit{kT0, Eigen::Vector3d(7.0e6, 0.0, 0.0),
                            Eigen::Vector3d(0.0, 7.5e3, 0.0)};

AttitudeState start(const Eigen::Vector3d& w,
                    const Eigen::Quaterniond& q = Eigen::Quaterniond::Identity()) {
    return AttitudeState{kT0, q, w};
}

Eigen::Vector3d inertial_h(const RigidBody& b, const AttitudeState& s) {
    return s.q_body_to_eci.toRotationMatrix() * b.angular_momentum_body(s.w_body);
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Kinematics
// ---------------------------------------------------------------------------------------------

// Spin at w = 0.5 rad/s about body z (a principal axis, so w stays constant) from q0 = 90 deg about
// inertial x. Body rates compose on the right: q(t) = q0 (x) [cos(wt/2), 0, 0, sin(wt/2)], so the
// body x axis in ECI is Rx(90 deg) [cos wt, sin wt, 0] = [cos wt, 0, sin wt]. The inertial-rate
// (wrong-order) convention would give [cos wt, sin wt, 0] instead.
// Error: the quaternion ODE is linear with rate w/2 = 0.25 rad/s, so each RK4 step has a phase
// error of (0.0025)^5 / 120 = 8.1e-16 in the quaternion's half-angle; over 1000 steps that is
// 8.1e-13, and a vector rotates by twice the half-angle: 1.6e-12. Bound 5e-12.
TEST(AttitudePropagation, BodyRatesComposeOnTheRight) {
    const RigidBody b(Eigen::Vector3d(10.0, 20.0, 30.0).asDiagonal());
    const Eigen::Quaterniond q0(Eigen::AngleAxisd(kPi / 2.0, Eigen::Vector3d::UnitX()));
    const double w = 0.5, t = 10.0;
    const AttitudeState s =
        simcore::propagate_attitude(b, kNoTorque, kOrbit, start({0.0, 0.0, w}, q0), kT0 + t, 0.01);
    EXPECT_EQ(s.t, kT0 + t);
    const Eigen::Vector3d x_eci = s.q_body_to_eci.toRotationMatrix() * Eigen::Vector3d::UnitX();
    EXPECT_LE((x_eci - Eigen::Vector3d(std::cos(w * t), 0.0, std::sin(w * t))).norm(), 5e-12);
    EXPECT_LE((s.w_body - Eigen::Vector3d(0.0, 0.0, w)).norm(), 1e-15);
}

// |q| = 1 after every step (renormalised): checked after 10^4 steps of a general tumble.
TEST(AttitudePropagation, QuaternionStaysUnitNorm) {
    const RigidBody b(Eigen::Vector3d(100.0, 200.0, 300.0).asDiagonal());
    const AttitudeState s = simcore::propagate_attitude(b, kNoTorque, kOrbit,
                                                        start({0.3, 1.0, -0.2}), kT0 + 100.0, 0.01);
    EXPECT_NEAR(s.q_body_to_eci.norm(), 1.0, 4.0 * std::numeric_limits<double>::epsilon());
}

// Constant torque tau = 2 N m about principal axis z (I3 = 20) from rest: no gyroscopic coupling,
// so w3 = (tau / I3) t = 0.1 t exactly (RK4 integrates a linear w exactly) and the rotation angle
// is theta = 0.05 t^2 about z. At t = 10 s: w3 = 1 rad/s, theta = 5 rad. The quaternion turns at
// phi_dot = w3 / 2 = 0.05 t, so its phase error accumulates as h^4 phi_dot^5 / 120 per unit time:
// h^4 / 120 * integral_0^10 (0.05 t)^5 dt = 1e-8 / 120 * 0.0521 = 4.3e-12, times 2 for the
// rotation angle: 8.7e-12. Bound 3e-11.
TEST(AttitudePropagation, ConstantTorqueAboutPrincipalAxis) {
    const RigidBody b(Eigen::Vector3d(10.0, 15.0, 20.0).asDiagonal());
    const ConstantBodyTorque tau(Eigen::Vector3d(0.0, 0.0, 2.0));
    const AttitudeState s = simcore::propagate_attitude(
        b, tau, kOrbit, start(Eigen::Vector3d::Zero()), kT0 + 10.0, 0.01);
    EXPECT_NEAR(s.w_body.z(), 1.0, 1e-13);
    EXPECT_EQ(s.w_body.x(), 0.0);
    EXPECT_EQ(s.w_body.y(), 0.0);
    const Eigen::Quaterniond expected(Eigen::AngleAxisd(5.0, Eigen::Vector3d::UnitZ()));
    EXPECT_LE(s.q_body_to_eci.angularDistance(expected), 3e-11);
}

// The (frozen) translational state reaches the torque model, with t equal to the attitude's t at
// every stage. tau_z = 1e-6 r_x = 7 N m about principal axis z (I3 = 20): w3 = 0.35 t, exact for
// RK4 (linear in t): 0.7 rad/s at t = 2 s, roundoff only (bound 1e-13).
TEST(AttitudePropagation, TorqueModelSeesTheFrozenOrbit) {
    struct PositionTorque final : simcore::TorqueModel {
        [[nodiscard]] Eigen::Vector3d torque_body(const simcore::State& orbit,
                                                  const AttitudeState& att) const override {
            if (orbit.t != att.t) {
                return Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
            }
            return Eigen::Vector3d(0.0, 0.0, 1e-6 * orbit.r_eci.x());
        }
    };
    const RigidBody b(Eigen::Vector3d(10.0, 15.0, 20.0).asDiagonal());
    const AttitudeState s = simcore::propagate_attitude(
        b, PositionTorque{}, kOrbit, start(Eigen::Vector3d::Zero()), kT0 + 2.0, 0.01);
    EXPECT_NEAR(s.w_body.z(), 0.7, 1e-13);
}

// ---------------------------------------------------------------------------------------------
// Torque-free motion (the M2 done-when checks)
// ---------------------------------------------------------------------------------------------

// Axisymmetric body I = diag(10, 10, 20), w0 = (0.1, 0, 1): w3 is constant and the transverse rate
// precesses in the body at lambda = (I3 - I1)/I1 w3 = 1 rad/s: w1 = 0.1 cos t, w2 = 0.1 sin t
// (from I1 w1' = (I2 - I3) w2 w3 = -10 w2, I2 w2' = (I3 - I1) w3 w1 = 10 w1).
// Error at t = 100 s: phase error (0.01)^5 / 120 = 8.3e-13 rad per step, 8.3e-9 rad over 1e4 steps,
// on an amplitude of 0.1 -> 8.3e-10 rad/s. Bound 3e-9 rad/s.
TEST(AttitudePropagation, AxisymmetricBodyPrecessesAtAnalyticRate) {
    const RigidBody b(Eigen::Vector3d(10.0, 10.0, 20.0).asDiagonal());
    const double t = 100.0;
    const AttitudeState s =
        simcore::propagate_attitude(b, kNoTorque, kOrbit, start({0.1, 0.0, 1.0}), kT0 + t, 0.01);
    EXPECT_NEAR(s.w_body.x(), 0.1 * std::cos(t), 3e-9);
    EXPECT_NEAR(s.w_body.y(), 0.1 * std::sin(t), 3e-9);
    EXPECT_NEAR(s.w_body.z(), 1.0, 1e-12);
}

// General asymmetric body, torque free: kinetic energy T and the INERTIAL angular momentum vector
// R(q) I w are conserved (the latter also checks that q is body-to-inertial and evolves with the
// right convention). 600 s at h = 0.01 s (6e4 steps), rates ~1 rad/s.
// T: per-step relative amplitude error (lambda h)^6/144 ~ 7e-15 -> 4e-10 over the run; bound 1e-8.
// H vector: limited by the attitude error; q turns at ~|w|/2 = 0.5 rad/s, so 2 * (0.005)^5 / 120 =
// 5e-14 rad per step, 3e-9 over 6e4 steps. Bound 1e-7.
TEST(AttitudePropagation, TorqueFreeConservesEnergyAndInertialAngularMomentum) {
    const RigidBody b(Eigen::Vector3d(100.0, 200.0, 300.0).asDiagonal());
    const AttitudeState s0 =
        start({0.3, 1.0, -0.2},
              Eigen::Quaterniond(Eigen::AngleAxisd(0.4, Eigen::Vector3d(1, 1, 0).normalized())));
    const AttitudeState s =
        simcore::propagate_attitude(b, kNoTorque, kOrbit, s0, kT0 + 600.0, 0.01);
    const double T0 = b.rotational_kinetic_energy(s0.w_body);
    EXPECT_NEAR(b.rotational_kinetic_energy(s.w_body) / T0, 1.0, 1e-8);
    const Eigen::Vector3d H0 = inertial_h(b, s0);
    EXPECT_LE((inertial_h(b, s) - H0).norm(), 1e-7 * H0.norm());
}

// Intermediate-axis theorem (the Dzhanibekov effect). I = diag(100, 200, 300), spin Omega = 1 rad/s
// about axis 2 (the intermediate one) with a 1e-4 rad/s nudge on axes 1 and 3. Linearising Euler's
// equations about w = (0, Omega, 0):
//   d1'' = ((I2 - I3)(I1 - I2) / (I1 I3)) Omega^2 d1 = (1/3) Omega^2 d1
// so the nudge grows as exp(lambda t), lambda = Omega / sqrt(3) = 0.5774 /s, and the spin flips.
// Measured from |w1| at t = 6 s and 10 s. Solving the linear system with this nudge, w1 = A e^(lt)
// + B e^(-lt) with A = -0.366e-4, B = 1.366e-4: the decaying part is 0.37% of |w1| at 6 s and
// biases the measured rate by 0.16%; |w1| < 1.2% of Omega, so nonlinear terms are ~1e-4.
// Band 0.5% (3x the predicted bias).
// Flip: from 1e-4 to O(1) takes ~ln(1e4)/lambda = 16 s, so within 60 s w2 must reach < -0.99 Omega.
TEST(AttitudePropagation, IntermediateAxisSpinIsUnstableAndFlips) {
    const RigidBody b(Eigen::Vector3d(100.0, 200.0, 300.0).asDiagonal());
    const double lambda = 1.0 / std::sqrt(3.0);
    const AttitudeState s0 = start({1e-4, 1.0, 1e-4});
    const AttitudeState s6 = simcore::propagate_attitude(b, kNoTorque, kOrbit, s0, kT0 + 6.0, 0.01);
    const AttitudeState s10 =
        simcore::propagate_attitude(b, kNoTorque, kOrbit, s0, kT0 + 10.0, 0.01);
    const double growth = std::log(std::abs(s10.w_body.x()) / std::abs(s6.w_body.x())) / 4.0;
    EXPECT_NEAR(growth / lambda, 1.0, 0.005);

    double w2_min = 1.0;
    AttitudeState s = s0;
    for (int k = 0; k < 600; ++k) {
        s = simcore::propagate_attitude(b, kNoTorque, kOrbit, s, s.t + 0.1, 0.01);
        w2_min = std::min(w2_min, s.w_body.y());
    }
    EXPECT_LT(w2_min, -0.99);
    // The flip conserves T and inertial H as well (same estimate as the general test, 6e3 steps).
    EXPECT_NEAR(b.rotational_kinetic_energy(s.w_body) / b.rotational_kinetic_energy(s0.w_body), 1.0,
                1e-8);
    EXPECT_LE((inertial_h(b, s) - inertial_h(b, s0)).norm(), 1e-7 * inertial_h(b, s0).norm());
}

// Spin about the minor (1) or major (3) axis with the same nudge is stable: the transverse rates
// only oscillate (linearised: d'' = -k Omega^2 d with k > 0) with amplitude of order the nudge
// times sqrt of an inertia ratio (<= ~2 here). Bound: they stay below 1e-3 rad/s for 60 s, 10x the
// nudge, far below the O(1) excursion of the unstable case.
TEST(AttitudePropagation, MinorAndMajorAxisSpinsAreStable) {
    const RigidBody b(Eigen::Vector3d(100.0, 200.0, 300.0).asDiagonal());
    for (const int axis : {0, 2}) {
        SCOPED_TRACE(axis);
        Eigen::Vector3d w = Eigen::Vector3d::Constant(1e-4);
        w(axis) = 1.0;
        AttitudeState s = start(w);
        double max_transverse = 0.0;
        for (int k = 0; k < 600; ++k) {
            s = simcore::propagate_attitude(b, kNoTorque, kOrbit, s, s.t + 0.1, 0.01);
            Eigen::Vector3d tr = s.w_body;
            tr(axis) = 0.0;
            max_transverse = std::max(max_transverse, tr.norm());
        }
        EXPECT_LT(max_transverse, 1e-3);
    }
}

// ---------------------------------------------------------------------------------------------
// Bookkeeping
// ---------------------------------------------------------------------------------------------

TEST(AttitudePropagation, ZeroDurationBackwardAndValidation) {
    const RigidBody b(Eigen::Vector3d(100.0, 200.0, 300.0).asDiagonal());
    const AttitudeState s0 = start({0.3, 1.0, -0.2});
    const AttitudeState same = simcore::propagate_attitude(b, kNoTorque, kOrbit, s0, kT0, 0.01);
    EXPECT_EQ(same.t, s0.t);
    EXPECT_EQ(same.w_body, s0.w_body);
    // Forward 10 s then back 10 s returns to the start (RK4 is not exactly reversible; the error
    // is the global error ~1e-11 relative here): bound 1e-9.
    const AttitudeState fwd =
        simcore::propagate_attitude(b, kNoTorque, kOrbit, s0, kT0 + 10.0, 0.01);
    const AttitudeState back = simcore::propagate_attitude(b, kNoTorque, kOrbit, fwd, kT0, 0.01);
    EXPECT_EQ(back.t, kT0);
    EXPECT_LE((back.w_body - s0.w_body).norm(), 1e-9);
    EXPECT_LE(back.q_body_to_eci.angularDistance(s0.q_body_to_eci), 1e-9);
    EXPECT_THROW((void)simcore::propagate_attitude(b, kNoTorque, kOrbit, s0, kT0 + 1.0, 0.0),
                 std::invalid_argument);
    EXPECT_THROW((void)simcore::propagate_attitude(b, kNoTorque, kOrbit, s0, kT0 + 1.0,
                                                   std::numeric_limits<double>::infinity()),
                 std::invalid_argument);
    // Determinism: bit-identical reruns.
    const AttitudeState again =
        simcore::propagate_attitude(b, kNoTorque, kOrbit, s0, kT0 + 10.0, 0.01);
    EXPECT_EQ(again.w_body, fwd.w_body);
    EXPECT_EQ(again.q_body_to_eci.coeffs(), fwd.q_body_to_eci.coeffs());
}

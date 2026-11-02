// M4.2: LQR attitude control, and a PD vs LQR comparison.
//
// CARE: for x' = A x + B u and J = integral (x^T Q x + u^T R u) dt, the optimal law is u = -K x,
// K = R^-1 B^T P, with P the stabilising solution of A^T P + P A - P B R^-1 B^T P + Q = 0.
// Solved by the matrix sign function: W = sign(H), H = [[A, -B R^-1 B^T], [-Q, -A^T]], then P from
// [W12; W22 + I] P = -[W11 + I; W21] (Byers 1987; the stable invariant subspace of H is [I; P]).
//
// Attitude model: with the w x H feedforward (M4.1), each axis's error is a double integrator,
// d_theta' = d_w, I d_w' = tau: x = [d_theta; d_w] (6), u = tau (3), A = [[0, 1], [0, 0]],
// B = [[0], [I^-1]]. The law is tau = -K x + w x H, the same structure as PD with K = [Kp Kd].
// Diagonal I, Q = diag(q_theta, q_w), R = r per axis decouples, and the CARE solves by hand:
//   p12 = m sqrt(q_theta r),   k1 = sqrt(q_theta / r),   k2 = sqrt((2 m sqrt(q_theta r) + q_w) / r)
// (m the axis inertia), so the closed loop is m theta'' + k2 theta' + k1 theta = 0.
// Bryson's rule: q_theta = 1 / theta_max^2, q_w = 1 / w_max^2, r = 1 / tau_max^2, so
// k1 = tau_max / theta_max: an error of theta_max asks for exactly the torque limit.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <numbers>
#include <stdexcept>

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include "simcore/dynamics/force_sum.hpp"
#include "simcore/gnc/attitude_control.hpp"
#include "simcore/gnc/lqr.hpp"
#include "simcore/sim/simulation.hpp"

namespace {

using simcore::AttitudeGains;
using simcore::AttitudeTarget;
using simcore::Simulation;
using simcore::VehicleConfig;
using simcore::VehicleState;

constexpr double kPi = std::numbers::pi;
constexpr double kDeg = kPi / 180.0;

double care_residual(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& Q,
                     const Eigen::MatrixXd& R, const Eigen::MatrixXd& P) {
    return (A.transpose() * P + P * A - P * B * R.inverse() * B.transpose() * P + Q).norm();
}

// The default app vehicle: 500 kg dry, I = diag(300, 400, 450), 0.1 N m wheels on the body axes
// (empty tank, so I is constant).
VehicleConfig default_vehicle() {
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
// CARE solver
// ---------------------------------------------------------------------------------------------

// Scalar, open-loop unstable: x' = x + u, q = r = 1. The CARE 2p - p^2 + 1 = 0 has the
// stabilising root p = 1 + sqrt(2) = 2.41421356 (closed loop 1 - p = -sqrt(2) < 0).
TEST(Lqr, ScalarCareMatchesTheClosedForm) {
    const Eigen::MatrixXd A = Eigen::MatrixXd::Constant(1, 1, 1.0);
    const Eigen::MatrixXd one = Eigen::MatrixXd::Identity(1, 1);
    const Eigen::MatrixXd P = simcore::solve_care(A, one, one, one);
    EXPECT_NEAR(P(0, 0), 1.0 + std::sqrt(2.0), 1e-12);
}

// A coupled problem with no closed form: the attitude model with a full (non-diagonal) inertia
// and cross-weighted Q. Checks: residual of the CARE ~0 (1e-9 of |Q|), P symmetric positive
// definite, and A - B K Hurwitz (every eigenvalue in the left half plane).
TEST(Lqr, CoupledCareSolutionIsStabilising) {
    Eigen::Matrix3d I;
    I << 300.0, 12.0, -5.0, 12.0, 400.0, 8.0, -5.0, 8.0, 450.0;
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(6, 6);
    A.topRightCorner(3, 3) = Eigen::Matrix3d::Identity();
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(6, 3);
    B.bottomRows(3) = I.inverse();
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(6, 6);
    Q(0, 3) = Q(3, 0) = 0.3;
    Q.bottomRightCorner(3, 3) *= 50.0;
    const Eigen::MatrixXd R = 4.0 * Eigen::MatrixXd::Identity(3, 3);
    const Eigen::MatrixXd P = simcore::solve_care(A, B, Q, R);
    EXPECT_LE(care_residual(A, B, Q, R, P), 1e-9 * Q.norm());
    EXPECT_LE((P - P.transpose()).norm(), 1e-9 * P.norm());
    EXPECT_GT(Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(P).eigenvalues().minCoeff(), 0.0);
    const Eigen::MatrixXd K = simcore::lqr_gain(A, B, Q, R);
    const Eigen::VectorXcd eig = (A - B * K).eigenvalues();
    EXPECT_LT(eig.real().maxCoeff(), 0.0);
}

// ---------------------------------------------------------------------------------------------
// Attitude LQR gains
// ---------------------------------------------------------------------------------------------

// Diagonal inertia and weights decouple: the solver's Kp, Kd match the hand solution per axis
// (k1 = sqrt(q_theta / r), k2 = sqrt((2 m sqrt(q_theta r) + q_w) / r)), off-diagonals 0.
// Weights: q_theta = 4, q_w = 900, r = 25 on every axis. Bound 1e-9 relative (iterative solve).
TEST(Lqr, DecoupledAttitudeGainsMatchTheHandSolution) {
    const Eigen::Vector3d m(300.0, 400.0, 450.0);
    const double q_th = 4.0;
    const double q_w = 900.0;
    const double r = 25.0;
    Eigen::Matrix<double, 6, 6> Q = Eigen::Matrix<double, 6, 6>::Zero();
    Q.diagonal() << q_th, q_th, q_th, q_w, q_w, q_w;
    const AttitudeGains g =
        simcore::lqr_attitude_gains(m.asDiagonal(), Q, r * Eigen::Matrix3d::Identity());
    for (int k = 0; k < 3; ++k) {
        SCOPED_TRACE(k);
        const double k1 = std::sqrt(q_th / r);
        const double k2 = std::sqrt((2.0 * m(k) * std::sqrt(q_th * r) + q_w) / r);
        EXPECT_NEAR(g.Kp(k, k) / k1, 1.0, 1e-9);
        EXPECT_NEAR(g.Kd(k, k) / k2, 1.0, 1e-9);
    }
    EXPECT_LE((g.Kp - Eigen::Matrix3d(g.Kp.diagonal().asDiagonal())).norm(), 1e-9);
    EXPECT_LE((g.Kd - Eigen::Matrix3d(g.Kd.diagonal().asDiagonal())).norm(), 1e-9);
}

// Bryson's rule ties the gains to the actuator: Kp = tau_max / theta_max per axis. With
// theta_max = 30 deg, w_max = 0.5 deg/s, tau_max = 0.1 N m: Kp = 0.1 / 0.5236 = 0.19099 N m/rad.
// About z (m = 450), with q_theta = 3.648, q_w = 1 / 0.0087266^2 = 13131, r = 100:
//   k2 = sqrt((2 * 450 * sqrt(3.648 * 100) + 13131) / 100) = 17.41,
// so the closed loop has wn = sqrt(k1 / m) = 0.0206 rad/s and zeta = k2 / (2 sqrt(k1 m)) = 0.939
// (well damped, so the peak torque is the initial one).
TEST(Lqr, BrysonWeightsPutTheErrorLimitAtTheTorqueLimit) {
    const Eigen::Matrix3d I = Eigen::Vector3d(300.0, 400.0, 450.0).asDiagonal();
    const simcore::BrysonLimits lim{30.0 * kDeg, 0.5 * kDeg, 0.1};
    const AttitudeGains g = simcore::lqr_attitude_gains_bryson(I, lim);
    for (int k = 0; k < 3; ++k) {
        EXPECT_NEAR(g.Kp(k, k), 0.1 / (30.0 * kDeg), 1e-9);
    }
    const double k1 = g.Kp(2, 2);
    const double k2 = g.Kd(2, 2);
    EXPECT_NEAR(k1, 0.19099, 1e-5);  // hand value
    EXPECT_NEAR(std::sqrt(k1 / 450.0), 0.0206, 1e-4);
    EXPECT_NEAR(k2 / (2.0 * std::sqrt(k1 * 450.0)), 0.94, 0.01);
}

// ---------------------------------------------------------------------------------------------
// PD vs LQR on the default vehicle
// ---------------------------------------------------------------------------------------------

// A 30 deg slew about a skew axis with the default vehicle's 0.1 N m wheels, from rest, to an
// inertial target. PD at the app's hold gains (wn = 0.05 rad/s, zeta = 0.7) asks for
// I wn^2 theta ~ 450 * 0.0025 * 0.52 = 0.59 N m at the start: 6x the limit, so it saturates
// (the allocation scales it down) and is no longer the linear loop it was designed as. LQR with
// Bryson limits (30 deg, 0.5 deg/s, 0.1 N m) asks for at most the limit by construction, so it
// stays linear. Claims checked: both settle below 0.1 deg within 900 s; PD's requested torque
// exceeds the wheel limit, LQR's never does (to 1%: the coupled skew-axis case isn't exactly the
// single-axis bound). Slew time, overshoot and wheel effort are printed for the README table.
TEST(Lqr, PdVsLqrSlewOnTheDefaultVehicle) {
    const VehicleConfig cfg = default_vehicle();
    const Eigen::Matrix3d I = cfg.mass.inertia_kgm2(0.0);
    const Eigen::Vector3d axis = Eigen::Vector3d(1.0, 2.0, 2.0).normalized();
    const double theta0 = 30.0 * kDeg;
    const VehicleState s0{simcore::Epoch::from_tt_seconds_since_j2000(800'000'000),
                          Eigen::Vector3d::Zero(),
                          Eigen::Vector3d::Zero(),
                          Eigen::Quaterniond(Eigen::AngleAxisd(theta0, axis)),
                          Eigen::Vector3d::Zero(),
                          0.0,
                          {0.0, 0.0, 0.0}};
    const AttitudeTarget target{Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero()};
    const AttitudeGains pd = simcore::pd_gains(I, simcore::PdGains{0.05, 0.7});
    const AttitudeGains lqr =
        simcore::lqr_attitude_gains_bryson(I, simcore::BrysonLimits{30.0 * kDeg, 0.5 * kDeg, 0.1});

    struct Result {
        double settle_s = -1.0;     // last time the error was >= 0.1 deg
        double overshoot = 0.0;     // largest error past the target, as a fraction of theta0
        double effort_nms = 0.0;    // integral of sum |tau_wheel| dt
        double peak_request = 0.0;  // largest |torque| asked of any wheel before scaling
    };
    const auto fly = [&](const AttitudeGains& g) {
        Simulation sim(simcore::SimConfig{cfg, std::make_shared<simcore::ForceSum>(),
                                          std::make_shared<simcore::ConstantBodyTorque>(
                                              Eigen::Vector3d::Zero()),
                                          0.1, std::nullopt, 0.0},  // 0.1 s ticks: plenty for
                       s0);                                         // minute-scale attitude motion
        Result res;
        for (int k = 0; k < 9000; ++k) {  // 900 s, command every tick (0.1 s)
            const VehicleState& s = sim.snapshot().state;
            // Wheels on the body axes: the allocation is tau_m = -tau_b exactly, so the largest
            // wheel torque requested (before scaling to the limits) is max |tau_b component|.
            const Eigen::Vector3d tau_b = simcore::attitude_torque_body(cfg, s, target, g);
            res.peak_request = std::max(res.peak_request, tau_b.cwiseAbs().maxCoeff());
            const simcore::VehicleCommand c = simcore::attitude_command(cfg, s, target, g);
            for (const double t : c.wheel_torque_nm) {
                res.effort_nms += std::abs(t) * 0.1;
            }
            sim.submit(c);
            sim.step(1);
            const VehicleState& n = sim.snapshot().state;
            const Eigen::Vector3d e =
                simcore::attitude_error_rad(n.q_body_to_eci, target.q_body_to_eci);
            if (e.norm() >= 0.1 * kDeg) {
                res.settle_s = n.t - s0.t;
            }
            res.overshoot =
                std::max(res.overshoot, -e.dot(axis) / theta0);  // past zero along -axis
        }
        return res;
    };
    const Result rp = fly(pd);
    const Result rl = fly(lqr);
    std::printf("[PD ] settle %.1f s  overshoot %.2f%%  effort %.2f N m s  peak request %.3f N m\n",
                rp.settle_s, 100.0 * rp.overshoot, rp.effort_nms, rp.peak_request);
    std::printf("[LQR] settle %.1f s  overshoot %.2f%%  effort %.2f N m s  peak request %.3f N m\n",
                rl.settle_s, 100.0 * rl.overshoot, rl.effort_nms, rl.peak_request);
    EXPECT_GT(rp.settle_s, 0.0);
    EXPECT_LT(rp.settle_s, 900.0);
    EXPECT_GT(rl.settle_s, 0.0);
    EXPECT_LT(rl.settle_s, 900.0);
    EXPECT_GT(rp.peak_request, 0.1);
    EXPECT_LE(rl.peak_request, 0.101);
}

TEST(Lqr, RejectsBadInput) {
    const Eigen::MatrixXd one = Eigen::MatrixXd::Identity(1, 1);
    EXPECT_THROW((void)simcore::solve_care(one, one, -one, one), std::invalid_argument);  // Q < 0
    EXPECT_THROW((void)simcore::solve_care(one, one, one, 0.0 * one), std::invalid_argument);  // R
    EXPECT_THROW((void)simcore::solve_care(one, Eigen::MatrixXd::Identity(2, 1), one, one),
                 std::invalid_argument);  // shapes
    const Eigen::Matrix3d I = Eigen::Vector3d(300.0, 400.0, 450.0).asDiagonal();
    EXPECT_THROW((void)simcore::lqr_attitude_gains_bryson(I, simcore::BrysonLimits{0.0, 0.01, 0.1}),
                 std::invalid_argument);
}

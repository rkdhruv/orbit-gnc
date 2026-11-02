#pragma once

#include <cstdint>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "simcore/dynamics/state.hpp"
#include "simcore/vehicle/vehicle.hpp"

namespace simcore {

// Orbit-relative attitude holds: where body +x (the main engine's thrust axis) points.
enum class HoldMode : std::uint8_t {
    kPrograde,    // along v
    kRetrograde,  // against v
    kNormal,      // along h = r x v
    kAntiNormal,  // against h
    kRadialOut,   // along r
    kRadialIn,    // against r
};

// A wanted attitude and the inertial rate at which it is turning.
struct AttitudeTarget {
    Eigen::Quaterniond q_body_to_eci;
    Eigen::Vector3d w_eci_radps;
};

// The hold frame for `mode` at the orbit state: body x along the mode's direction d, roll fixed by
// putting body z as close as possible to radial-out (to the orbit normal when d is radial):
//   x = d,  z = unit(ref - (ref . x) x),  y = z x x,  R = [x y z]
// It turns with the orbit at w = (r x v) / |r|^2 (the rate of r_hat; exact for circular orbits,
// where the whole orbit frame turns at that rate). Throws std::invalid_argument for r = 0 or
// h ~ 0 (a radial trajectory has no orbit frame).
AttitudeTarget hold_target(HoldMode mode, const State& orbit);

// Shortest-path rotation vector from q_target to q, in body axes: with q_e = q_target^-1 (x) q =
// [w, v], d_theta = 2 atan2(|v|, |w|) sign(w) v_hat (q and -q are the same attitude, so the
// error is at most pi). Both quaternions must be unit.
Eigen::Vector3d attitude_error_rad(const Eigen::Quaterniond& q_body_to_eci,
                                   const Eigen::Quaterniond& q_target_body_to_eci);

// Gain matrices of the attitude law tau_b = -Kp d_theta - Kd d_w + w x H (body axes).
struct AttitudeGains {
    Eigen::Matrix3d Kp;  // [N m / rad]
    Eigen::Matrix3d Kd;  // [N m s / rad]
};

// PD gains as a natural frequency and damping ratio: Kp = I wn^2, Kd = 2 zeta wn I, so about a
// principal axis the loop is theta'' + 2 zeta wn theta' + wn^2 theta = 0. Throws
// std::invalid_argument for wn not finite and > 0 or zeta not finite and >= 0.
struct PdGains {
    double wn_radps;
    double zeta;
};
AttitudeGains pd_gains(const Eigen::Matrix3d& inertia_kgm2, const PdGains& g);

// LQR gains for the linearised attitude error. With the w x H feedforward, the error dynamics are
// a double integrator per axis: x = [d_theta; d_w], x' = A x + B u, A = [[0, 1], [0, 0]] (3x3
// blocks), B = [[0], [I^-1]], u = tau. K = [Kp Kd] minimises integral (x^T Q x + u^T R u) dt
// (solve_care). Q is 6x6 symmetric PSD, R 3x3 symmetric PD.
AttitudeGains lqr_attitude_gains(const Eigen::Matrix3d& inertia_kgm2,
                                 const Eigen::Matrix<double, 6, 6>& Q, const Eigen::Matrix3d& R);

// Bryson's rule: weight each quantity by the inverse square of its largest acceptable value,
// Q = diag(1/theta_max^2 (x3), 1/w_max^2 (x3)), R = 1/tau_max^2 I. Then Kp = tau_max / theta_max:
// an error of theta_max asks for exactly the torque limit. Throws for non-positive limits.
struct BrysonLimits {
    double angle_rad;
    double rate_radps;
    double torque_nm;
};
AttitudeGains lqr_attitude_gains_bryson(const Eigen::Matrix3d& inertia_kgm2,
                                        const BrysonLimits& limits);

// The attitude law's body torque, before allocation and limits:
//   tau_b = -Kp d_theta - Kd d_w + w x H
// with d_w = w - R(q)^T w_target (rate relative to the target frame, so a turning target is
// tracked without lag) and H = I w + sum h a. The w x H term cancels the gyroscopic term of
// Euler's equations, leaving the linear error dynamics the gains were designed for.
Eigen::Vector3d attitude_torque_body(const VehicleConfig& cfg, const VehicleState& s,
                                     const AttitudeTarget& target, const AttitudeGains& gains);

// The command: attitude_torque_body allocated to the wheels (wheel_torques_for_body_torque,
// scaled together to the limits); the main engine at `throttle`, other thrusters off. Throws
// std::invalid_argument for throttle outside [0, 1] or a state that doesn't match the vehicle.
VehicleCommand attitude_command(const VehicleConfig& cfg, const VehicleState& s,
                                const AttitudeTarget& target, const AttitudeGains& gains,
                                double throttle = 0.0);
// The same with PD gains from the current inertia (pd_gains).
VehicleCommand attitude_command(const VehicleConfig& cfg, const VehicleState& s,
                                const AttitudeTarget& target, const PdGains& gains,
                                double throttle = 0.0);

}  // namespace simcore

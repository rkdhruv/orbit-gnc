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

// PD gains as a natural frequency and damping ratio: Kp = I wn^2, Kd = 2 zeta wn I.
struct PdGains {
    double wn_radps;
    double zeta;
};

// PD attitude control on the quaternion error, through the reaction wheels:
//   tau_b = -I (wn^2 d_theta + 2 zeta wn d_w) + w x H
// with d_w = w - R(q)^T w_target (rate relative to the target frame, so a turning target is
// tracked without lag) and H = I w + sum h a. The w x H term cancels the gyroscopic term of
// Euler's equations, so about a principal axis the loop is theta'' + 2 zeta wn theta' + wn^2 theta
// = 0. The torque is allocated to the wheels (wheel_torques_for_body_torque, scaled together to
// the limits); the main engine runs at `throttle`, other thrusters off. Throws
// std::invalid_argument for wn not finite and > 0, zeta not finite and >= 0, throttle outside
// [0, 1] or a state that doesn't match the vehicle.
VehicleCommand attitude_command(const VehicleConfig& cfg, const VehicleState& s,
                                const AttitudeTarget& target, const PdGains& gains,
                                double throttle = 0.0);

}  // namespace simcore

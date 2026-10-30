#pragma once

#include <span>

#include <Eigen/Core>

namespace simcore {

// An axisymmetric reaction wheel: spin axis (body, normalised), axial inertia J, motor torque
// limit and relative-speed limit. Its state is its absolute axial angular momentum h (the wheel's
// own spin plus the body rate along the axis), held by whoever integrates the vehicle.
class ReactionWheel {
public:
    // Throws std::invalid_argument unless the axis is finite and nonzero and J, the torque limit
    // and the speed limit are finite and > 0.
    ReactionWheel(const Eigen::Vector3d& axis_body, double axial_inertia_kgm2, double max_torque_nm,
                  double max_speed_radps);

    [[nodiscard]] const Eigen::Vector3d& axis_body() const { return a_; }
    [[nodiscard]] double axial_inertia_kgm2() const { return J_; }
    [[nodiscard]] double max_momentum_nms() const { return J_ * max_speed_; }  // J Omega_max

    // Wheel speed relative to the body: Omega = h / J - a . w.
    [[nodiscard]] double relative_speed_radps(double h_nms, const Eigen::Vector3d& w_body) const;

    // Motor torque actually applied for a command: clipped to +/- max torque, and set to zero if it
    // would push |Omega| further past the speed limit (spinning down is always allowed).
    [[nodiscard]] double applied_motor_torque_nm(double commanded_nm,
                                                 double relative_speed_radps) const;

private:
    Eigen::Vector3d a_;
    double J_;
    double max_torque_;
    double max_speed_;
};

// Rotational dynamics of a vehicle carrying reaction wheels (exact for axisymmetric wheels,
// Markley & Crassidis section 3.3.5). I_locked is the whole vehicle's inertia with the wheels
// locked; h are the wheels' axial momenta, tau_motor the applied motor torques (same order).
//   I_s = I_locked - sum J_i a_i a_i^T
//   H   = I_s w + sum h_i a_i
//   I_s w_dot = tau_ext - w x H - sum tau_motor_i a_i,      h_i_dot = tau_motor_i
// With tau_ext = 0, H is conserved in inertial space: the wheels only redistribute it.

// I_s. Throws if it is not positive definite (a wheel too heavy for the vehicle).
Eigen::Matrix3d inertia_without_wheel_spin(const Eigen::Matrix3d& inertia_locked_kgm2,
                                           std::span<const ReactionWheel> wheels);

// Total angular momentum H in body axes [N m s]. Throws if h doesn't match the wheels.
Eigen::Vector3d total_angular_momentum_body(const Eigen::Matrix3d& inertia_locked_kgm2,
                                            std::span<const ReactionWheel> wheels,
                                            const Eigen::Vector3d& w_body,
                                            std::span<const double> h_nms);

// Body angular acceleration w_dot. Throws if h or tau_motor don't match the wheels.
Eigen::Vector3d angular_acceleration_with_wheels(const Eigen::Matrix3d& inertia_locked_kgm2,
                                                 std::span<const ReactionWheel> wheels,
                                                 const Eigen::Vector3d& w_body,
                                                 std::span<const double> h_nms,
                                                 const Eigen::Vector3d& torque_ext_body_nm,
                                                 std::span<const double> motor_torque_nm);

}  // namespace simcore

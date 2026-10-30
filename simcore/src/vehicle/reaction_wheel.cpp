#include "simcore/vehicle/reaction_wheel.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include <Eigen/Dense>

#include "simcore/check.hpp"

namespace simcore {

ReactionWheel::ReactionWheel(const Eigen::Vector3d& axis_body, double axial_inertia_kgm2,
                             double max_torque_nm, double max_speed_radps)
    : J_(axial_inertia_kgm2), max_torque_(max_torque_nm), max_speed_(max_speed_radps) {
    SIMCORE_CHECK(axis_body.allFinite() && axis_body.norm() > 0.0,
                  "wheel axis must be finite and nonzero");
    SIMCORE_CHECK(std::isfinite(J_) && J_ > 0.0, "wheel inertia must be finite and > 0");
    SIMCORE_CHECK(std::isfinite(max_torque_) && max_torque_ > 0.0,
                  "wheel torque limit must be finite and > 0");
    SIMCORE_CHECK(std::isfinite(max_speed_) && max_speed_ > 0.0,
                  "wheel speed limit must be finite and > 0");
    a_ = axis_body.normalized();
}

double ReactionWheel::relative_speed_radps(double h_nms, const Eigen::Vector3d& w_body) const {
    return h_nms / J_ - a_.dot(w_body);
}

double ReactionWheel::applied_motor_torque_nm(double commanded_nm,
                                              double relative_speed_radps) const {
    SIMCORE_CHECK(std::isfinite(commanded_nm) && std::isfinite(relative_speed_radps),
                  "wheel command and speed must be finite");
    const double tau = std::clamp(commanded_nm, -max_torque_, max_torque_);
    // At (or past) the speed limit, refuse torque that would spin the wheel further out.
    if ((relative_speed_radps >= max_speed_ && tau > 0.0) ||
        (relative_speed_radps <= -max_speed_ && tau < 0.0)) {
        return 0.0;
    }
    return tau;
}

Eigen::Matrix3d inertia_without_wheel_spin(const Eigen::Matrix3d& inertia_locked_kgm2,
                                           std::span<const ReactionWheel> wheels) {
    Eigen::Matrix3d Is = inertia_locked_kgm2;
    for (const ReactionWheel& w : wheels) {
        Is -= w.axial_inertia_kgm2() * (w.axis_body() * w.axis_body().transpose());
    }
    SIMCORE_CHECK(Is.llt().info() == Eigen::Success,
                  "I - sum J a a^T must be positive definite (wheel too heavy for the vehicle)");
    return Is;
}

Eigen::Vector3d total_angular_momentum_body(const Eigen::Matrix3d& inertia_locked_kgm2,
                                            std::span<const ReactionWheel> wheels,
                                            const Eigen::Vector3d& w_body,
                                            std::span<const double> h_nms) {
    SIMCORE_CHECK(h_nms.size() == wheels.size(), "one momentum per wheel");
    Eigen::Vector3d H = inertia_without_wheel_spin(inertia_locked_kgm2, wheels) * w_body;
    for (std::size_t i = 0; i < wheels.size(); ++i) {
        H += h_nms[i] * wheels[i].axis_body();
    }
    return H;
}

Eigen::Vector3d angular_acceleration_with_wheels(const Eigen::Matrix3d& inertia_locked_kgm2,
                                                 std::span<const ReactionWheel> wheels,
                                                 const Eigen::Vector3d& w_body,
                                                 std::span<const double> h_nms,
                                                 const Eigen::Vector3d& torque_ext_body_nm,
                                                 std::span<const double> motor_torque_nm) {
    SIMCORE_CHECK(h_nms.size() == wheels.size() && motor_torque_nm.size() == wheels.size(),
                  "one momentum and one motor torque per wheel");
    const Eigen::Matrix3d Is = inertia_without_wheel_spin(inertia_locked_kgm2, wheels);
    Eigen::Vector3d H = Is * w_body;
    Eigen::Vector3d motor = Eigen::Vector3d::Zero();
    for (std::size_t i = 0; i < wheels.size(); ++i) {
        H += h_nms[i] * wheels[i].axis_body();
        motor += motor_torque_nm[i] * wheels[i].axis_body();
    }
    // I_s w_dot = tau_ext - w x H - sum tau_i a_i  (the motor's reaction on the body).
    return Is.ldlt().solve(torque_ext_body_nm - w_body.cross(H) - motor);
}

}  // namespace simcore

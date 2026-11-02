#pragma once

#include <Eigen/Core>

namespace simcore {

// The RCS thrusters that turn the vehicle about one body axis: couples of two thrusters firing in
// opposite directions at +-L from the centre of mass, one couple per sense. A couple gives a pure
// torque (its two forces cancel), so RCS turning never moves the centre of mass. A command u in
// [-1, 1] picks the couple (sign) and its throttle (|u|):
//   torque = u tau_max a_hat,   total thrust = |torque| / L,   m_dot = |u| tau_max / (L Isp g0)
// (each thruster gives T = |torque| / (2 L), and both burn propellant from the main tank).
class RcsAxis {
public:
    // Throws std::invalid_argument unless the axis is finite and nonzero and the torque, lever
    // arm and Isp are finite and > 0.
    RcsAxis(const Eigen::Vector3d& axis_body, double max_torque_nm, double lever_arm_m,
            double isp_s);

    [[nodiscard]] const Eigen::Vector3d& axis_body() const { return a_; }
    [[nodiscard]] double max_torque_nm() const { return max_torque_; }

    // u must be in [-1, 1] (throws otherwise, NaN included).
    [[nodiscard]] Eigen::Vector3d torque_body_nm(double u) const;
    [[nodiscard]] double mass_flow_kgps(double u) const;

private:
    Eigen::Vector3d a_;
    double max_torque_;
    double lever_arm_;
    double isp_;
};

}  // namespace simcore

#pragma once

#include <Eigen/Core>

namespace simcore {

// A fixed thruster on the vehicle body. Position is relative to the centre of mass (which the
// tank model keeps fixed), direction is the thrust direction (normalised on construction).
//   F = throttle T_max d_hat,   tau = r x F,   m_dot = throttle T_max / (Isp g0)
class Thruster {
public:
    // Throws std::invalid_argument unless thrust and Isp are finite and > 0, the position is
    // finite and the direction is finite and nonzero.
    Thruster(double max_thrust_n, double isp_s, const Eigen::Vector3d& position_body_m,
             const Eigen::Vector3d& direction_body);

    // Throttle must be in [0, 1] (throws otherwise).
    [[nodiscard]] Eigen::Vector3d force_body_n(double throttle) const;
    [[nodiscard]] Eigen::Vector3d torque_body_nm(double throttle) const;  // about the CoM
    [[nodiscard]] double mass_flow_kgps(double throttle) const;           // propellant used, >= 0

    // Effective exhaust speed v_e = Isp g0 [m/s].
    [[nodiscard]] double exhaust_velocity_mps() const;

private:
    double max_thrust_n_;
    double isp_s_;
    Eigen::Vector3d r_;
    Eigen::Vector3d d_;
};

// Tsiolkovsky rocket equation: dv = v_e ln(m0 / mf), the velocity change from burning m0 - mf of
// propellant with no external forces. Throws unless v_e > 0 and m0 >= mf > 0.
double rocket_delta_v_mps(double exhaust_velocity_mps, double m0_kg, double mf_kg);

}  // namespace simcore

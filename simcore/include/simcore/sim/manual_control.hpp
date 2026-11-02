#pragma once

#include <Eigen/Core>

#include "simcore/vehicle/vehicle.hpp"

namespace simcore {

// What the pilot asks for this frame.
struct ManualInput {
    double throttle{0.0};  // main engine (thruster 0), in [0, 1]
    // Body torque request per axis in [-1, 1]: x roll, y pitch, z yaw (1 = full wheel authority).
    Eigen::Vector3d rotate{Eigen::Vector3d::Zero()};
    bool rate_damping{false};  // oppose any rotation when there is no rotate input
};

// Rate damping gain k [1/s]: tau_b = -k I w, so w_dot ~ -k w (a 2 s time constant).
inline constexpr double kRateDampingGain_per_s = 0.5;

// Pilot input to a vehicle command.
// Throttle: the main engine (thruster 0) only; other thrusters off.
// Rotation is by the reaction wheels. Motor torque tau_m on wheel j turns the body by -tau_m a_j,
// so a wanted body torque tau_b needs tau_m = -A+ tau_b, with A = [a_1 ... a_n] and A+ its
// pseudo-inverse (the minimum-norm allocation). If any wheel would exceed its torque limit, all
// are scaled down by the same factor, which keeps the torque's direction.
//   rotate input u:  tau_b = sum_k u_k T_k e_k, T_k = the largest torque the wheels can make
//                    about body axis k alone
//   rate damping:    tau_b = -k I w (used only when u = 0); exactly zero once
//                    |w| <= Simulation::kRailsMaxRate_radps, so a damped vehicle can go on rails.
// Throws std::invalid_argument for throttle outside [0, 1], a rotate component that isn't finite
// and in [-1, 1], or a state that doesn't match the vehicle.
VehicleCommand manual_command(const VehicleConfig& cfg, const ManualInput& in,
                              const VehicleState& s);

}  // namespace simcore

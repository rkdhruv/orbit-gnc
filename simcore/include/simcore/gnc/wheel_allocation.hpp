#pragma once

#include <vector>

#include <Eigen/Core>

#include "simcore/vehicle/vehicle.hpp"

namespace simcore {

// Reaction-wheel torque allocation. A motor torque tau_m on wheel j turns the body by -tau_m a_j
// (reaction), so a wanted body torque tau_b needs tau_m = -A+ tau_b, with A = [a_1 ... a_n] and
// A+ its pseudo-inverse (the minimum-norm allocation; for rank-3 A, -A tau_m = A A+ tau_b = tau_b
// exactly). If any wheel would exceed its torque limit, all are scaled down by the same factor,
// which keeps the torque's direction. Returns one torque per wheel (empty with no wheels).
std::vector<double> wheel_torques_for_body_torque(const VehicleConfig& cfg,
                                                  const Eigen::Vector3d& tau_body_nm);

// The largest factor s for which s tau_b fits within every wheel's torque limit under this
// allocation (+infinity for tau_b = 0 or no wheels needed; 0 with no wheels and tau_b != 0).
double wheel_torque_headroom(const VehicleConfig& cfg, const Eigen::Vector3d& tau_body_nm);

// T_k: the largest body torque about body axis k (0, 1, 2) under this allocation, i.e. a unit
// request about axis k scaled until the first wheel hits its limit. 0 if the allocation gives no
// wheel torque for that axis. (A redundant set, e.g. a 4-wheel pyramid, could make more using its
// null space; for a set that can't act about axis k, A A+ e_k != e_k, so the torque isn't about
// axis k at all.)
double wheel_axis_authority_nm(const VehicleConfig& cfg, int axis);

}  // namespace simcore

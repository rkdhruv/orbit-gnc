#pragma once

#include <vector>

#include <Eigen/Core>

#include "simcore/gnc/attitude_control.hpp"
#include "simcore/vehicle/vehicle.hpp"

namespace simcore {

// Reaction-wheel momentum management. A torque the attitude loop resists (e.g. gravity gradient)
// accumulates in the wheels until they saturate; the RCS dumps it:
//   tau_rcs = -k h_w,   h_w = sum h_j a_j (the momentum the wheels hold, body axes)
// while the wheels are commanded tau_att - tau_rcs, so wheels and RCS together give the body
// exactly tau_att (the attitude is undisturbed) and the wheels spin down as h_w' = -k h_w +
// tau_env. Dumping starts when the busiest wheel reaches start_fraction of its speed limit and
// stops when all are below stop_fraction (hysteresis, so it doesn't chatter).
struct DesatSettings {
    double start_fraction{0.8};
    double stop_fraction{0.1};
    double gain_per_s{0.02};  // k: a 50 s time constant
};

// sum h_j a_j [N m s].
Eigen::Vector3d wheel_momentum_body(const VehicleConfig& cfg, const VehicleState& s);

// max_j |h_j / J_j - a_j . w| / Omega_max,j (0 with no wheels).
double max_wheel_saturation(const VehicleConfig& cfg, const VehicleState& s);

// Whether to dump now, given whether a dump was already running (the hysteresis).
bool desat_active(bool was_active, const VehicleConfig& cfg, const VehicleState& s,
                  const DesatSettings& settings);

// RCS commands for tau_rcs = -k h_w: u = A+ tau_rcs per unit of each axis's torque limit
// (pseudo-inverse over the RCS axes), scaled down together so every |u| <= 1 and so that the
// wheels can counter it with at most kDumpWheelShare of their torque authority (the rest stays
// free for holding attitude; otherwise a large dump would push the wheels to their torque limit,
// disturb the attitude and absorb momentum too slowly to keep up). Empty with no RCS.
inline constexpr double kDumpWheelShare = 0.8;
std::vector<double> rcs_dump_command(const VehicleConfig& cfg, const VehicleState& s,
                                     const DesatSettings& settings);

// An attitude command (attitude_torque_body with `gains`) that also dumps wheel momentum when
// `dumping`: RCS from rcs_dump_command, wheels allocated for tau_att - tau_rcs (the torque the
// RCS will actually make). Main engine at `throttle`. Throws as attitude_command does.
VehicleCommand attitude_command_with_desat(const VehicleConfig& cfg, const VehicleState& s,
                                           const AttitudeTarget& target, const AttitudeGains& gains,
                                           bool dumping, const DesatSettings& settings,
                                           double throttle = 0.0);

}  // namespace simcore

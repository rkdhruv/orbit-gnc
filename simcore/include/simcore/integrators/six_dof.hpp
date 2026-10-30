#pragma once

#include "simcore/dynamics/force_model.hpp"
#include "simcore/dynamics/rigid_body.hpp"
#include "simcore/vehicle/vehicle.hpp"

namespace simcore {

// Coupled 6-DOF vehicle dynamics, integrated with fixed-step RK4 (matching the fixed sim tick).
// State y = [r, v, q, w, m_p, h_1..h_n]:
//   r_dot = v
//   v_dot = a_env(t, r, v) + R(q) sum F_i / m(m_p)
//   q_dot = 1/2 q (x) [0, w]
//   I_s w_dot = tau_env + sum r_i x F_i - w x H - sum tau_motor_j a_j,   with I = I(m_p)
//   m_p_dot = -sum m_dot_i,   h_j_dot = tau_motor_j
// Burnout: if a step would use more propellant than is left (m_dot h > m_p), every throttle in it
// is scaled by f = m_p / (m_dot h), so the step expels exactly m_p and ends with the tank exactly
// empty. The rocket equation depends only on the mass ratio, so dv is exact; position differs from
// a full-thrust partial step by O(a h^2). After every step q is renormalised.
// Steps must run forward (h_s >= 0) whenever a throttle is nonzero: backward flight with thrust
// would refill the tank, so it isn't the inverse of forward flight. Coasting backward is fine.
// Simplifications: Euler's equation uses the current I(m_p) with no jet damping or I_dot w term;
// force models with their own mass (drag) don't see the changing vehicle mass.
// env_accel supplies environment accelerations (gravity, drag) from State{t, r, v}; env_torque
// supplies external torques from the same State and the attitude.

// One RK4 step of size h_s. Inputs are validated as in propagate_six_dof.
VehicleState six_dof_rk4_step(const VehicleConfig& cfg, const ForceModel& env_accel,
                              const TorqueModel& env_torque, const VehicleCommand& cmd,
                              const VehicleState& s, double h_s);

// Propagate s0 to t_end in n = max(1, ceil(|T| / h_max_s)) equal steps (step k starts at
// s0.t + k h; the result's t is exactly t_end; T = 0 returns s0). Throws std::invalid_argument for
// h_max_s not finite and > 0, command or wheel-state sizes that don't match the config, throttles
// outside [0, 1], propellant outside [0, capacity], or a backward span with nonzero throttle.
VehicleState propagate_six_dof(const VehicleConfig& cfg, const ForceModel& env_accel,
                               const TorqueModel& env_torque, const VehicleCommand& cmd,
                               const VehicleState& s0, const Epoch& t_end, double h_max_s);

}  // namespace simcore

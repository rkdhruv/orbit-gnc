#pragma once

#include "simcore/dynamics/rigid_body.hpp"
#include "simcore/orbits/epoch.hpp"

namespace simcore {

// One classical RK4 step of size h_s (may be negative) on the attitude state y = [q; w]:
//   q_dot = 1/2 q (x) [0, w],   w_dot = I^-1 (tau - w x I w)
// q is renormalised at the end of the step (RK4 doesn't preserve |q| = 1). Stage quaternions are
// normalised before being handed to the torque model.
AttitudeState attitude_rk4_step(const RigidBody& body, const TorqueModel& torque,
                                const AttitudeState& s, double h_s);

// Propagate s0 to t_end in n = max(1, ceil(|T| / h_max_s)) equal RK4 steps, step k starting at
// s0.t + k h (not accumulated); the result's t is exactly t_end and T = 0 returns s0.
// h_max_s must be finite and > 0.
AttitudeState propagate_attitude(const RigidBody& body, const TorqueModel& torque,
                                 const AttitudeState& s0, const Epoch& t_end, double h_max_s);

}  // namespace simcore

#pragma once

#include "simcore/dynamics/rigid_body.hpp"
#include "simcore/dynamics/state.hpp"
#include "simcore/orbits/epoch.hpp"

namespace simcore {

// One classical RK4 step of size h_s (may be negative) on the attitude state y = [q; w]:
//   q_dot = 1/2 q (x) [0, w],   w_dot = I^-1 (tau - w x I w)
// q is renormalised at the end of the step (RK4 doesn't preserve |q| = 1). Stage quaternions are
// normalised before being handed to the torque model.
// Attitude only: the torque model sees `orbit`'s position and velocity held fixed, with t set to
// each stage's time. Coupled translation and attitude is propagate_six_dof.
AttitudeState attitude_rk4_step(const RigidBody& body, const TorqueModel& torque,
                                const State& orbit, const AttitudeState& s, double h_s);

// Propagate s0 to t_end in n = max(1, ceil(|T| / h_max_s)) equal RK4 steps, step k starting at
// s0.t + k h (not accumulated); the result's t is exactly t_end and T = 0 returns s0.
// h_max_s must be finite and > 0. `orbit` is held fixed, as in attitude_rk4_step.
AttitudeState propagate_attitude(const RigidBody& body, const TorqueModel& torque,
                                 const State& orbit, const AttitudeState& s0, const Epoch& t_end,
                                 double h_max_s);

}  // namespace simcore

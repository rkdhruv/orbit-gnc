#pragma once

#include <cstdint>

#include "simcore/dynamics/force_model.hpp"
#include "simcore/dynamics/state.hpp"
#include "simcore/orbits/epoch.hpp"

namespace simcore {

// One velocity-Verlet step of size h_s (may be negative), kick-drift-kick:
//   v_half = v + h/2 a(t, r, v);  r1 = r + h v_half;  v1 = v_half + h/2 a(t + h, r1, v_half)
// 2nd order, time-reversible and symplectic for a = a(r, t). For velocity-dependent forces the
// second evaluation uses v_half (no longer symplectic).
State verlet_step(const ForceModel& f, const State& s, double h_s);

// One classical 4th-order Runge-Kutta step on y = [r; v], y' = [v; a].
State rk4_step(const ForceModel& f, const State& s, double h_s);

enum class FixedStepMethod : std::uint8_t { kVelocityVerlet, kRk4 };

// Propagate s0 to t_end in n = max(1, ceil(|T| / h_max_s)) equal steps of h = T / n, where
// T = t_end - s0.t (may be negative). Step k starts at s0.t + k h, computed directly rather than
// accumulated, so there is no stub step and no drift in the schedule. The result's t is exactly
// t_end; T = 0 returns s0. h_max_s must be finite and > 0.
State propagate_fixed(const ForceModel& f, const State& s0, const Epoch& t_end, double h_max_s,
                      FixedStepMethod method);

}  // namespace simcore

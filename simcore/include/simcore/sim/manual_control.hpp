#pragma once

#include <cstdint>
#include <functional>

#include <Eigen/Core>

#include "simcore/sim/simulation.hpp"
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
// Rotation is by the reaction wheels, through wheel_torques_for_body_torque (gnc/
// wheel_allocation.hpp: tau_m = -A+ tau_b, scaled down together to the torque limits):
//   rotate input u:  tau_b = sum_k u_k T_k e_k, T_k = wheel_axis_authority_nm(cfg, k)
//   rate damping:    tau_b = -k I w (used only when u = 0); exactly zero once
//                    |w| <= Simulation::kRailsMaxRate_radps, so a damped vehicle can go on rails.
// Throws std::invalid_argument for throttle outside [0, 1], a rotate component that isn't finite
// and in [-1, 1], or a state that doesn't match the vehicle.
VehicleCommand manual_command(const VehicleConfig& cfg, const ManualInput& in,
                              const VehicleState& s);

// The longest a manual command may be held before it is recomputed [s]. The damping command held
// for T gives the discrete loop w+ = (1 - k T) w, stable only for k T < 2; at k = 0.5 /s, 0.5 s
// gives k T = 0.25 (a monotone decay, within 4% of the continuous e^(-k T) = 0.78 per hold).
inline constexpr double kManualHoldMax_s = 0.5;

// Run `ticks` ticks of manual flight: submit manual_command(input, current state) and step, in
// chunks of at most hold_ticks (default: kManualHoldMax_s worth of ticks). The front end calls
// this once per frame, so the vehicle's behaviour doesn't depend on the frame time or the warp
// (a 10 fps frame at 100x warp would otherwise hold one damping command for 10 s, k T = 5).
// A coasting command (nothing firing) is held for the whole call, since recomputing it can't
// change it unless the input changes. Every command goes through submit(), so it is in the input
// log and replays exactly. Throws std::invalid_argument for ticks < 0 or hold_ticks < 0.
void fly_manual(Simulation& sim, const ManualInput& input, std::int64_t ticks,
                std::int64_t hold_ticks = 0);  // 0: kManualHoldMax_s / tick_s

// The same loop for any control law (state -> command), e.g. an attitude hold: the law is
// re-evaluated at least every hold_ticks, and a coasting command is held for the rest of the call.
using ControlLaw = std::function<VehicleCommand(const VehicleState&)>;
void fly_with(Simulation& sim, const ControlLaw& law, std::int64_t ticks,
              std::int64_t hold_ticks = 0);

}  // namespace simcore

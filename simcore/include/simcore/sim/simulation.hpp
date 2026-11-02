#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "simcore/dynamics/force_model.hpp"
#include "simcore/dynamics/rigid_body.hpp"
#include "simcore/dynamics/state.hpp"
#include "simcore/orbits/epoch.hpp"
#include "simcore/vehicle/vehicle.hpp"

namespace simcore {

// What the sim runs: the vehicle, its environment and the fixed tick length.
struct SimConfig {
    VehicleConfig vehicle;
    std::shared_ptr<const ForceModel> forces;    // environment accelerations (gravity, J2, drag)
    std::shared_ptr<const TorqueModel> torques;  // environment torques
    double tick_s{0.02};                         // fixed step, 50 Hz
    // Set to mu only when `forces` is exactly two-body gravity with that mu: enables coasting on
    // rails (exact Kepler propagation, for time warp). Unset, the sim always integrates.
    std::optional<double> rails_mu;
};

// True if the command fires nothing: every throttle and wheel torque command is exactly zero.
bool is_coast(const VehicleCommand& c);

// A copy of everything the front end may read, taken between ticks.
struct Snapshot {
    std::int64_t tick{0};    // ticks completed
    VehicleState state;      // state.t = t0 + tick * h
    VehicleCommand command;  // the command the last tick ran with (zeros before the first)
    bool on_rails{false};    // the last tick was a Kepler coast rather than an RK4 step
};

// The sim facade: the only way the front end drives the simulation. It owns the vehicle state and
// advances it in fixed ticks with six_dof_rk4_step; tick k runs from t_k = t0 + k h, computed from
// t0 rather than accumulated. The front end can submit commands, step, and read snapshots, and
// nothing else. A command takes effect at the start of the next tick (commands submitted between
// two ticks apply in order, so the last wins) and is held until replaced. The state after tick n
// therefore depends only on the initial state and the command in effect at each tick, not on the
// frame rate or how ticks were grouped into step() calls. Single-threaded by design (the browser
// build runs everything on its main loop).
class Simulation {
public:
    // Throws std::invalid_argument unless tick_s is finite and > 0, forces and torques are set and
    // the initial state matches the vehicle (validate_vehicle_state). Starts with all throttles
    // and wheel torques at zero.
    Simulation(SimConfig config, VehicleState initial);

    // Validated now (validate_command; throws std::invalid_argument and queues nothing).
    void submit(const VehicleCommand& cmd);

    // Run n >= 0 ticks (throws for n < 0).
    // On rails: when rails_mu is set and the vehicle is coasting (all throttles and wheel torque
    // commands zero, |w| <= kRailsMaxRate_radps), a tick doesn't integrate. The vehicle goes on
    // rails at the state it has then (any residual w is set to 0, as KSP stops rotation in warp),
    // and every rails tick evaluates the Kepler solution from that entry state:
    // (r, v)(t_k) = kepler_propagate(entry, t_k). Exact, O(1) for any n, and a function of the
    // tick index alone, so chunking still doesn't matter. With w = 0 and no torques, q, w,
    // propellant and wheel momenta are exactly constant (I_s w_dot = -w x H - sum tau_m a = 0).
    // The first tick with a non-coasting command integrates again from the current rails state.
    // Throws if the orbit is one kepler_propagate rejects (near-parabolic).
    void step(std::int64_t n = 1);

    static constexpr double kRailsMaxRate_radps = 1e-4;  // ~0.006 deg/s

    [[nodiscard]] Snapshot snapshot() const;
    [[nodiscard]] const SimConfig& config() const { return cfg_; }

private:
    SimConfig cfg_;
    Epoch t0_;
    std::int64_t tick_{0};
    VehicleState state_;
    VehicleCommand command_;
    std::vector<VehicleCommand> pending_;
    bool on_rails_{false};
    State rails_entry_;  // translational state when the vehicle went on rails
};

}  // namespace simcore

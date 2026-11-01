#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "simcore/dynamics/force_model.hpp"
#include "simcore/dynamics/rigid_body.hpp"
#include "simcore/orbits/epoch.hpp"
#include "simcore/vehicle/vehicle.hpp"

namespace simcore {

// What the sim runs: the vehicle, its environment and the fixed tick length.
struct SimConfig {
    VehicleConfig vehicle;
    std::shared_ptr<const ForceModel> forces;    // environment accelerations (gravity, J2, drag)
    std::shared_ptr<const TorqueModel> torques;  // environment torques
    double tick_s{0.02};                         // fixed step, 50 Hz
};

// A copy of everything the front end may read, taken between ticks.
struct Snapshot {
    std::int64_t tick{0};    // ticks completed
    VehicleState state;      // state.t = t0 + tick * h
    VehicleCommand command;  // the command the last tick ran with (zeros before the first)
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
    void step(std::int64_t n = 1);

    [[nodiscard]] Snapshot snapshot() const;
    [[nodiscard]] const SimConfig& config() const { return cfg_; }

private:
    SimConfig cfg_;
    Epoch t0_;
    std::int64_t tick_{0};
    VehicleState state_;
    VehicleCommand command_;
    std::vector<VehicleCommand> pending_;
};

}  // namespace simcore

#include "simcore/sim/simulation.hpp"

#include <cmath>
#include <utility>

#include "simcore/check.hpp"
#include "simcore/integrators/six_dof.hpp"

namespace simcore {

Simulation::Simulation(SimConfig config, VehicleState initial)
    : cfg_(std::move(config)), t0_(initial.t), state_(std::move(initial)) {
    SIMCORE_CHECK(std::isfinite(cfg_.tick_s) && cfg_.tick_s > 0.0, "tick must be finite and > 0");
    SIMCORE_CHECK(cfg_.forces != nullptr, "sim needs a force model");
    SIMCORE_CHECK(cfg_.torques != nullptr, "sim needs a torque model");
    validate_vehicle_state(cfg_.vehicle, state_);
    command_.throttle.assign(cfg_.vehicle.thrusters.size(), 0.0);
    command_.wheel_torque_nm.assign(cfg_.vehicle.wheels.size(), 0.0);
}

void Simulation::submit(const VehicleCommand& cmd) {
    validate_command(cfg_.vehicle, cmd);
    pending_.push_back(cmd);
}

void Simulation::step(std::int64_t n) {
    SIMCORE_CHECK(n >= 0, "tick count must be >= 0");
    const double h = cfg_.tick_s;
    for (std::int64_t k = 0; k < n; ++k) {
        // Commands are whole replacements applied in submission order, so the last one wins.
        if (!pending_.empty()) {
            command_ = pending_.back();
            pending_.clear();
        }
        state_.t = t0_ + static_cast<double>(tick_) * h;  // the schedule, not an accumulated sum
        state_ = six_dof_rk4_step(cfg_.vehicle, *cfg_.forces, *cfg_.torques, command_, state_, h);
        ++tick_;
        state_.t = t0_ + static_cast<double>(tick_) * h;
    }
}

Snapshot Simulation::snapshot() const {
    return Snapshot{tick_, state_, command_};
}

}  // namespace simcore

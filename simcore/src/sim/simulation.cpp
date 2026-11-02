#include "simcore/sim/simulation.hpp"

#include <cmath>
#include <utility>

#include "simcore/check.hpp"
#include "simcore/integrators/six_dof.hpp"
#include "simcore/orbits/kepler.hpp"

namespace simcore {

bool is_coast(const VehicleCommand& c) {
    for (const double u : c.throttle) {
        if (u != 0.0) {
            return false;
        }
    }
    for (const double tau : c.wheel_torque_nm) {
        if (tau != 0.0) {
            return false;
        }
    }
    return true;
}

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
            if (pending_.back() != command_) {
                command_ = pending_.back();
                log_.push_back(LoggedCommand{tick_, command_});  // takes effect at this tick
            }
            pending_.clear();
        }
        state_.t = t0_ + static_cast<double>(tick_) * h;  // the schedule, not an accumulated sum

        const bool coasting = cfg_.rails_mu.has_value() && is_coast(command_);
        if (!on_rails_ && coasting && state_.w_body.norm() <= kRailsMaxRate_radps) {
            on_rails_ = true;
            state_.w_body.setZero();
            rails_entry_ = State{state_.t, state_.r_eci, state_.v_eci};
        } else if (on_rails_ && !coasting) {
            on_rails_ = false;  // integrate from the current rails state, so no jump
        }
        if (on_rails_) {
            // Commands only change between step() calls, so the rest of this call is a coast:
            // jump straight to its last tick.
            tick_ += n - k;
            state_.t = t0_ + static_cast<double>(tick_) * h;
            const State s = kepler_propagate(rails_entry_, state_.t, *cfg_.rails_mu);
            state_.r_eci = s.r_eci;
            state_.v_eci = s.v_eci;
            return;
        }
        state_ = six_dof_rk4_step(cfg_.vehicle, *cfg_.forces, *cfg_.torques, command_, state_, h);
        ++tick_;
        state_.t = t0_ + static_cast<double>(tick_) * h;
    }
}

Snapshot Simulation::snapshot() const {
    return Snapshot{tick_, state_, command_, on_rails_};
}

}  // namespace simcore

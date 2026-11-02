#include "simcore/sim/simulation.hpp"

#include <cmath>
#include <utility>

#include "simcore/check.hpp"
#include "simcore/constants.hpp"
#include "simcore/integrators/six_dof.hpp"
#include "simcore/orbits/elements.hpp"
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
    SIMCORE_CHECK(std::isfinite(cfg_.impact_radius_m) && cfg_.impact_radius_m >= 0.0,
                  "impact radius must be finite and >= 0");
    validate_vehicle_state(cfg_.vehicle, state_);
    command_.throttle.assign(cfg_.vehicle.thrusters.size(), 0.0);
    command_.wheel_torque_nm.assign(cfg_.vehicle.wheels.size(), 0.0);
}

void Simulation::submit(const VehicleCommand& cmd) {
    validate_command(cfg_.vehicle, cmd);
    pending_.push_back(cmd);
}

bool Simulation::rails_eligible(const VehicleCommand& cmd) const {
    if (!cfg_.rails_mu.has_value() || !is_coast(cmd)) {
        return false;
    }
    if (on_rails_) {
        return true;  // already on rails: w = 0 and the orbit was checked at entry
    }
    if (state_.w_body.norm() > kRailsMaxRate_radps) {
        return false;
    }
    // The Kepler solver needs a conic (not rectilinear, not near-parabolic), and a trajectory
    // that dips below the surface must be integrated so the impact is caught at its tick.
    const State orbit{state_.t, state_.r_eci, state_.v_eci};
    if (!has_conic_elements(orbit, *cfg_.rails_mu)) {
        return false;
    }
    const KeplerianElements el = elements_from_state(orbit, *cfg_.rails_mu);
    return el.a_m * (1.0 - el.e) > cfg_.impact_radius_m;  // periapsis above the surface
}

bool Simulation::would_rail(const VehicleCommand& cmd) const {
    return rails_eligible(cmd);
}

void Simulation::step(std::int64_t n) {
    SIMCORE_CHECK(n >= 0, "tick count must be >= 0");
    const double h = cfg_.tick_s;
    for (std::int64_t k = 0; k < n && !impacted_; ++k) {
        // Commands are whole replacements applied in submission order, so the last one wins.
        if (!pending_.empty()) {
            if (pending_.back() != command_) {
                command_ = pending_.back();
                log_.push_back(LoggedCommand{tick_, command_});  // takes effect at this tick
            }
            pending_.clear();
        }
        state_.t = t0_ + static_cast<double>(tick_) * h;  // the schedule, not an accumulated sum

        // rails_eligible implies rails_mu is set; the explicit test lets the checker see it.
        if (cfg_.rails_mu.has_value() && rails_eligible(command_)) {
            // Commands only change between step() calls, so the rest of this call is a coast:
            // jump straight to its last tick. The Kepler solve runs before anything is changed,
            // so if it throws the sim is left exactly as it was (strong exception guarantee).
            const State entry =
                on_rails_ ? rails_entry_ : State{state_.t, state_.r_eci, state_.v_eci};
            const std::int64_t end_tick = tick_ + (n - k);
            const Epoch t_end = t0_ + static_cast<double>(end_tick) * h;
            const State s = kepler_propagate(entry, t_end, *cfg_.rails_mu);
            if (!on_rails_) {
                on_rails_ = true;
                state_.w_body.setZero();  // stop the residual rotation (documented)
                rails_entry_ = entry;
            }
            tick_ = end_tick;
            state_.t = t_end;
            state_.r_eci = s.r_eci;
            state_.v_eci = s.v_eci;
            return;  // rails never reach the surface (periapsis checked at entry)
        }
        on_rails_ =
            false;  // integrate from the current state (the rails state if we were on rails)
        VehicleState next =
            six_dof_rk4_step(cfg_.vehicle, *cfg_.forces, *cfg_.torques, command_, state_, h);
        ++tick_;
        next.t = t0_ + static_cast<double>(tick_) * h;
        state_ = std::move(next);
        if (state_.r_eci.norm() < cfg_.impact_radius_m) {
            impacted_ = true;  // the first tick ending below the surface; the sim stops here
        }
    }
}

Snapshot Simulation::snapshot() const {
    return Snapshot{tick_, state_, command_, on_rails_, impacted_};
}

}  // namespace simcore

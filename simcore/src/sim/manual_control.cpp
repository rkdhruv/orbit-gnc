#include "simcore/sim/manual_control.hpp"

#include <algorithm>
#include <cmath>

#include "simcore/check.hpp"
#include "simcore/gnc/wheel_allocation.hpp"
#include "simcore/integrators/six_dof.hpp"
#include "simcore/sim/simulation.hpp"

namespace simcore {

VehicleCommand manual_command(const VehicleConfig& cfg, const ManualInput& in,
                              const VehicleState& s) {
    SIMCORE_CHECK(in.throttle >= 0.0 && in.throttle <= 1.0, "throttle must be in [0, 1]");
    SIMCORE_CHECK(in.rotate.allFinite() && in.rotate.cwiseAbs().maxCoeff() <= 1.0,
                  "rotate inputs must be in [-1, 1]");
    validate_vehicle_state(cfg, s);

    VehicleCommand c;
    c.throttle.assign(cfg.thrusters.size(), 0.0);
    if (!c.throttle.empty()) {
        c.throttle.front() = in.throttle;
    }
    c.wheel_torque_nm.assign(cfg.wheels.size(), 0.0);
    c.rcs.assign(cfg.rcs.size(), 0.0);

    Eigen::Vector3d tau_b = Eigen::Vector3d::Zero();
    if (in.rotate != Eigen::Vector3d::Zero()) {
        for (int k = 0; k < 3; ++k) {
            tau_b(k) = in.rotate(k) * wheel_axis_authority_nm(cfg, k);
        }
    } else if (in.rate_damping && s.w_body.norm() > Simulation::kRailsMaxRate_radps) {
        tau_b = -kRateDampingGain_per_s * (cfg.mass.inertia_kgm2(s.propellant_kg) * s.w_body);
    } else {
        return c;
    }
    c.wheel_torque_nm = wheel_torques_for_body_torque(cfg, tau_b);
    return c;
}

void fly_manual(Simulation& sim, const ManualInput& input, std::int64_t ticks,
                std::int64_t hold_ticks) {
    const VehicleConfig& cfg = sim.config().vehicle;
    fly_with(
        sim, [&cfg, &input](const VehicleState& s) { return manual_command(cfg, input, s); }, ticks,
        hold_ticks);
}

void fly_with(Simulation& sim, const ControlLaw& law, std::int64_t ticks, std::int64_t hold_ticks) {
    SIMCORE_CHECK(ticks >= 0, "tick count must be >= 0");
    SIMCORE_CHECK(hold_ticks >= 0, "hold must be >= 1 tick (or 0 for the default)");
    if (hold_ticks == 0) {
        hold_ticks = std::max<std::int64_t>(
            1, static_cast<std::int64_t>(std::floor(kManualHoldMax_s / sim.config().tick_s)));
    }
    while (ticks > 0 && !sim.snapshot().impacted) {
        const VehicleCommand cmd = law(sim.snapshot().state);
        sim.submit(cmd);
        const std::int64_t chunk = is_coast(cmd) ? ticks : std::min(ticks, hold_ticks);
        sim.step(chunk);
        ticks -= chunk;
    }
}

}  // namespace simcore

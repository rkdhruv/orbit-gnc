#include "simcore/sim/telemetry.hpp"

#include <cmath>
#include <limits>
#include <numbers>

#include <Eigen/Geometry>

#include "simcore/dynamics/state.hpp"
#include "simcore/integrators/six_dof.hpp"
#include "simcore/vehicle/thruster.hpp"

namespace simcore {

Telemetry telemetry(const VehicleConfig& cfg, const VehicleState& s, double mu) {
    validate_vehicle_state(cfg, s);
    Telemetry tm{};
    tm.altitude_m = s.r_eci.norm() - earth::radius;
    tm.speed_mps = s.v_eci.norm();
    const State orbit{s.t, s.r_eci, s.v_eci};
    tm.elements_valid = has_conic_elements(orbit, mu);
    if (tm.elements_valid) {
        tm.elements = elements_from_state(orbit, mu);
        const double a = tm.elements.a_m;
        const double e = tm.elements.e;
        tm.periapsis_altitude_m = a * (1.0 - e) - earth::radius;  // a < 0 and e > 1 for hyperbolas
        if (e < 1.0) {
            tm.apoapsis_altitude_m = a * (1.0 + e) - earth::radius;
            tm.period_s = 2.0 * std::numbers::pi * std::sqrt(a * a * a / mu);
        } else {
            tm.apoapsis_altitude_m = std::numeric_limits<double>::infinity();
            tm.period_s = std::numeric_limits<double>::infinity();
        }
    } else {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        tm.elements = KeplerianElements{nan, nan, nan, nan, nan, nan};
        tm.periapsis_altitude_m = nan;
        tm.apoapsis_altitude_m = nan;
        tm.period_s = nan;
    }

    tm.mass_kg = cfg.mass.mass_kg(s.propellant_kg);
    tm.propellant_kg = s.propellant_kg;
    tm.delta_v_remaining_mps =
        cfg.thrusters.empty() ? 0.0
                              : rocket_delta_v_mps(cfg.thrusters.front().exhaust_velocity_mps(),
                                                   tm.mass_kg, cfg.mass.dry_mass_kg());

    // Angle via atan2(|x cross v|, x . v): unlike acos, accurate near 0 and pi.
    const Eigen::Vector3d x_eci = s.q_body_to_eci * Eigen::Vector3d::UnitX();
    tm.thrust_axis_to_prograde_rad = std::atan2(x_eci.cross(s.v_eci).norm(), x_eci.dot(s.v_eci));

    tm.w_body = s.w_body;
    tm.wheel_h_nms = s.wheel_h_nms;
    tm.wheel_saturation.reserve(cfg.wheels.size());
    for (std::size_t j = 0; j < cfg.wheels.size(); ++j) {
        const ReactionWheel& wh = cfg.wheels[j];
        tm.wheel_saturation.push_back(wh.relative_speed_radps(s.wheel_h_nms[j], s.w_body) /
                                      wh.max_speed_radps());
    }
    return tm;
}

}  // namespace simcore

#pragma once

#include <vector>

#include <Eigen/Core>

#include "simcore/constants.hpp"
#include "simcore/orbits/elements.hpp"
#include "simcore/vehicle/vehicle.hpp"

namespace simcore {

// Derived quantities for display (the HUD), computed here so they are tested; the front end only
// converts units and formats them. SI units throughout.
struct Telemetry {
    double altitude_m;            // |r| - R, spherical Earth
    double speed_mps;             // |v|, inertial
    KeplerianElements elements;   // osculating (instantaneous two-body) elements
    double periapsis_altitude_m;  // a (1 - e) - R
    double apoapsis_altitude_m;   // a (1 + e) - R; +infinity for open orbits (e >= 1)
    double period_s;              // 2 pi sqrt(a^3 / mu); +infinity for open orbits
    double mass_kg;
    double propellant_kg;
    // Rocket equation for the main engine (thruster 0): v_e ln(m / m_dry). 0 with no thrusters.
    double delta_v_remaining_mps;
    // Angle between body +x (the main engine's thrust axis by convention) and the velocity:
    // 0 = prograde, pi = retrograde.
    double thrust_axis_to_prograde_rad;
    Eigen::Vector3d w_body;           // body rates [rad/s]
    std::vector<double> wheel_h_nms;  // per wheel, as in the state
    // Per wheel: speed relative to the body, h / J - a.w, over its limit (signed; |1| saturated).
    std::vector<double> wheel_saturation;
};

// Throws std::invalid_argument for a state that doesn't match the vehicle (validate_vehicle_state)
// or one elements_from_state rejects.
Telemetry telemetry(const VehicleConfig& cfg, const VehicleState& s, double mu = earth::mu);

}  // namespace simcore

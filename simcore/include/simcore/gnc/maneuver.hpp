#pragma once

#include <cstdint>

#include <Eigen/Core>

#include "simcore/constants.hpp"
#include "simcore/dynamics/state.hpp"
#include "simcore/gnc/attitude_control.hpp"
#include "simcore/vehicle/vehicle.hpp"

namespace simcore {

// RSW (radial, along-track, cross-track) frame at a state, as the columns of the RSW-to-ECI
// rotation: R = r_hat, W = h_hat (orbit normal), S = W x R (along-track; the velocity direction
// for a circular orbit). Throws std::invalid_argument for r = 0 or h ~ 0.
Eigen::Matrix3d rsw_to_eci(const State& s);

// A planned burn: an impulsive dv in the RSW frame of the orbit at time t.
struct ManeuverNode {
    Epoch t;
    Eigen::Vector3d dv_rsw_mps{Eigen::Vector3d::Zero()};  // Eigen doesn't zero-initialise
};

// The state at the node, Kepler-propagated from `now` (exact for two-body; with J2 or drag it is
// a prediction that drifts), and the state just after an impulsive burn there:
// v+ = v + [R S W] dv_rsw. Throw as kepler_propagate does.
State node_state(const State& now, const ManeuverNode& node, double mu = earth::mu);
State post_burn_state(const State& now, const ManeuverNode& node, double mu = earth::mu);

// Finite burn with constant thrust T from mass m0 at exhaust speed v_e (rocket equation):
//   duration t_b = (m0 v_e / T) (1 - exp(-dv / v_e)),
//   lead     t_l = (m0 v_e / T) (1 - exp(-dv / (2 v_e))): the time to deliver half the dv.
// Igniting at t_node - t_l centres the burn on the node by dv (the second half is shorter,
// because the acceleration grows as the tank empties). Throw for non-positive or non-finite input.
double burn_duration_s(double dv_mps, double m0_kg, double thrust_n, double ve_mps);
double burn_lead_s(double dv_mps, double m0_kg, double thrust_n, double ve_mps);

// Flies one node with the main engine (thruster 0, which must thrust along body +x) and the
// attitude law. Burn attitude: body x along the node's dv in ECI, fixed in inertial space, roll
// set by putting body z toward the node's radial direction (its orbit normal for a radial burn).
//   kAlign: hold the burn attitude until ignition: t >= t_node - t_l and aligned (attitude error
//           <= kIgnitionAlign_rad, rate <= kIgnitionRate_radps). If the slew isn't done in time
//           the burn waits and starts late (ignition_delayed()). While late it re-plans the burn
//           as centred at (now + t_l) and aims for that (a target turning with the orbit),
//           because the node's own inertial direction goes stale at the orbit rate; on ignition
//           the direction is frozen. node() and ignition_time() then report the re-planned values.
//   kBurn:  hold it and fire. Achieved dv = v_e ln(m_ignition / m), from the propellant used (the
//           RCS stays off during the burn, so all of it went through the main engine). Remaining
//           dv: the node's dv minus the achieved (kDeltaV), or for an apsis cutoff
//           (target - r_apsis) / (d r_apsis / d dv along the burn direction, by a 1 m/s finite
//           difference). The throttle is min(1, remaining / (a hold_s)), so the last command hold
//           delivers just the remainder; below a 1 mm/s remainder the engine is cut. (A safety
//           stop also ends an apsis-targeted burn at twice the node's dv.)
//   kDone:  hold the burn attitude, engine off.
// hold_s is how long each command is held (the fly_with chunk). Every command is meant to be
// submitted, so a flown node replays exactly.
// When a burn ends. kDeltaV: once the node's dv has been delivered. kApoapsis / kPeriapsis: once
// the osculating apoapsis (raising) or periapsis (lowering) reaches radius_m. Near periapsis the
// apoapsis is very sensitive to dv (dr_a/dv ~ 4 a^2 v / mu, ~60 km per m/s on a GTO), so an
// apsis-targeted cutoff absorbs the finite-burn loss that a dv cutoff would leave as a miss.
struct BurnCutoff {
    enum class Kind : std::uint8_t { kDeltaV, kApoapsis, kPeriapsis };
    Kind kind{Kind::kDeltaV};
    double radius_m{0.0};
};

inline constexpr double kIgnitionAlign_rad = 0.017453292519943295;   // 1 deg
inline constexpr double kIgnitionRate_radps = 8.726646259971648e-4;  // 0.05 deg/s

class ManeuverExecutor {
public:
    enum class Phase : std::uint8_t { kAlign, kBurn, kDone };

    // Throws std::invalid_argument for a non-finite or zero dv, hold_s not finite and > 0, no
    // thrusters, a main engine not along body +x, or a state that doesn't match the vehicle.
    ManeuverExecutor(const VehicleConfig& cfg, const VehicleState& now, const ManeuverNode& node,
                     double hold_s, BurnCutoff cutoff = {}, double mu = earth::mu);

    // The command for the next hold, advancing the phase as needed.
    VehicleCommand command(const VehicleConfig& cfg, const VehicleState& s,
                           const AttitudeGains& gains);

    [[nodiscard]] Phase phase() const { return phase_; }
    [[nodiscard]] const ManeuverNode& node() const { return node_; }
    [[nodiscard]] Epoch ignition_time() const { return t_ignition_; }
    [[nodiscard]] const Eigen::Vector3d& burn_direction_eci() const { return dir_eci_; }
    [[nodiscard]] double dv_target_mps() const { return dv_target_; }
    [[nodiscard]] double achieved_dv_mps() const { return achieved_; }
    [[nodiscard]] double burn_duration_s() const { return t_burn_; }
    // True if the scheduled ignition time came before the vehicle was aligned.
    [[nodiscard]] bool ignition_delayed() const { return delayed_; }

private:
    void aim(const State& at);  // set dir_eci_ and target_'s attitude for a burn applied at `at`
    [[nodiscard]] double remaining_dv(const VehicleState& s) const;

    ManeuverNode node_;
    double hold_s_;
    BurnCutoff cutoff_;
    double mu_;
    double lead_{0.0};
    double dv_target_;
    double ve_;
    double thrust_;
    Eigen::Vector3d dir_eci_;
    AttitudeTarget target_;
    Epoch t_ignition_;
    double t_burn_;
    Phase phase_{Phase::kAlign};
    double m_ignition_{0.0};
    double achieved_{0.0};
    bool delayed_{false};
};

}  // namespace simcore

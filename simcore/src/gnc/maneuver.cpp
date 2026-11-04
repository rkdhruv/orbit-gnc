#include "simcore/gnc/maneuver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>

#include <Eigen/Geometry>

#include "simcore/check.hpp"
#include "simcore/integrators/six_dof.hpp"
#include "simcore/orbits/elements.hpp"
#include "simcore/orbits/kepler.hpp"

namespace simcore {

namespace {

constexpr double kCutoffResidual_mps = 1e-3;

void check_burn_inputs(double dv, double m0, double thrust, double ve) {
    SIMCORE_CHECK(std::isfinite(dv) && dv >= 0.0 && std::isfinite(m0) && m0 > 0.0 &&
                      std::isfinite(thrust) && thrust > 0.0 && std::isfinite(ve) && ve > 0.0,
                  "burn inputs must be finite, dv >= 0 and m0, thrust, v_e > 0");
}

}  // namespace

// Burn direction and attitude for a dv_rsw applied at `at`: body x along the dv, body z toward
// radial-out (the orbit normal for a radial burn).
void ManeuverExecutor::aim(const State& at) {
    const Eigen::Matrix3d M = rsw_to_eci(at);
    dir_eci_ = (M * node_.dv_rsw_mps).normalized();
    const Eigen::Vector3d& x = dir_eci_;
    Eigen::Vector3d ref = M.col(0);
    if (std::abs(ref.dot(x)) > 0.999) {
        ref = M.col(2);
    }
    const Eigen::Vector3d z = (ref - ref.dot(x) * x).normalized();
    Eigen::Matrix3d R;
    R.col(0) = x;
    R.col(1) = z.cross(x);
    R.col(2) = z;
    target_.q_body_to_eci = Eigen::Quaterniond(R).normalized();
}

Eigen::Matrix3d rsw_to_eci(const State& s) {
    const double rn = s.r_eci.norm();
    SIMCORE_CHECK(rn > 0.0 && s.r_eci.allFinite() && s.v_eci.allFinite(),
                  "state must be finite, r != 0");
    const Eigen::Vector3d h = s.r_eci.cross(s.v_eci);
    SIMCORE_CHECK(h.norm() > 1e-12 * rn * s.v_eci.norm(), "radial trajectory has no RSW frame");
    const Eigen::Vector3d R = s.r_eci / rn;
    const Eigen::Vector3d W = h.normalized();
    Eigen::Matrix3d M;
    M.col(0) = R;
    M.col(1) = W.cross(R);
    M.col(2) = W;
    return M;
}

State node_state(const State& now, const ManeuverNode& node, double mu) {
    return kepler_propagate(now, node.t, mu);
}

State post_burn_state(const State& now, const ManeuverNode& node, double mu) {
    SIMCORE_CHECK(node.dv_rsw_mps.allFinite(), "node dv must be finite");
    State s = node_state(now, node, mu);
    s.v_eci += rsw_to_eci(s) * node.dv_rsw_mps;
    return s;
}

double burn_duration_s(double dv_mps, double m0_kg, double thrust_n, double ve_mps) {
    check_burn_inputs(dv_mps, m0_kg, thrust_n, ve_mps);
    return (m0_kg * ve_mps / thrust_n) * -std::expm1(-dv_mps / ve_mps);
}

double burn_lead_s(double dv_mps, double m0_kg, double thrust_n, double ve_mps) {
    check_burn_inputs(dv_mps, m0_kg, thrust_n, ve_mps);
    return (m0_kg * ve_mps / thrust_n) * -std::expm1(-dv_mps / (2.0 * ve_mps));
}

ManeuverExecutor::ManeuverExecutor(const VehicleConfig& cfg, const VehicleState& now,
                                   const ManeuverNode& node, double hold_s, BurnCutoff cutoff,
                                   double mu)
    : node_(node), hold_s_(hold_s), cutoff_(cutoff), mu_(mu) {
    SIMCORE_CHECK(cutoff.kind == BurnCutoff::Kind::kDeltaV ||
                      (std::isfinite(cutoff.radius_m) && cutoff.radius_m > 0.0),
                  "an apsis cutoff needs a finite target radius > 0");
    SIMCORE_CHECK(node.dv_rsw_mps.allFinite() && node.dv_rsw_mps.norm() > 0.0,
                  "node dv must be finite and nonzero");
    SIMCORE_CHECK(std::isfinite(hold_s_) && hold_s_ > 0.0, "hold must be finite and > 0");
    SIMCORE_CHECK(!cfg.thrusters.empty(), "a maneuver needs a main engine (thruster 0)");
    validate_vehicle_state(cfg, now);
    const Eigen::Vector3d f = cfg.thrusters.front().force_body_n(1.0);
    thrust_ = f.norm();
    SIMCORE_CHECK((f / thrust_ - Eigen::Vector3d::UnitX()).norm() < 1e-9,
                  "the main engine must thrust along body +x");
    ve_ = cfg.thrusters.front().exhaust_velocity_mps();
    dv_target_ = node.dv_rsw_mps.norm();

    // Burn direction and attitude, from the node state; fixed in inertial space (w_target = 0).
    target_.w_eci_radps.setZero();
    aim(node_state(State{now.t, now.r_eci, now.v_eci}, node, mu));

    const double m0 = cfg.mass.mass_kg(now.propellant_kg);
    t_burn_ = simcore::burn_duration_s(dv_target_, m0, thrust_, ve_);
    lead_ = burn_lead_s(dv_target_, m0, thrust_, ve_);
    t_ignition_ = node.t - lead_;
    t_node_planned_ = node.t;
    last_ = State{now.t, now.r_eci, now.v_eci};

    // T = 2 pi sqrt(a^3 / mu) of the orbit now.
    late_limit_s_ = std::numeric_limits<double>::infinity();
    if (has_conic_elements(last_, mu)) {
        const KeplerianElements el = elements_from_state(last_, mu);
        if (el.e < 1.0) {
            late_limit_s_ =
                0.25 * 2.0 * std::numbers::pi * std::sqrt(el.a_m * el.a_m * el.a_m / mu);
        }
    }
}

double ManeuverExecutor::committed_dv_mps() const {
    return phase_ == Phase::kAlign || aborted_ ? 0.0 : ve_ * std::log(m_ignition_ / m_hold_end_);
}

ManeuverNode ManeuverExecutor::remaining_node() const {
    const double left = std::max(dv_target_ - committed_dv_mps(), 0.0);
    return ManeuverNode{last_.t, rsw_to_eci(last_).transpose() * (dir_eci_ * left)};
}

double ManeuverExecutor::apsis_radius(const State& o) const {
    // +infinity for the apoapsis of an open or non-conic orbit.
    const bool apo = cutoff_.kind == BurnCutoff::Kind::kApoapsis;
    if (!has_conic_elements(o, mu_)) {
        return apo ? std::numeric_limits<double>::infinity() : 0.0;
    }
    const KeplerianElements el = elements_from_state(o, mu_);
    if (apo) {
        return el.e < 1.0 ? el.a_m * (1.0 + el.e) : std::numeric_limits<double>::infinity();
    }
    return el.a_m * (1.0 - el.e);
}

VehicleCommand ManeuverExecutor::command(const VehicleConfig& cfg, const VehicleState& s,
                                         const AttitudeGains& gains) {
    const double m = cfg.mass.mass_kg(s.propellant_kg);
    last_ = State{s.t, s.r_eci, s.v_eci};
    if (phase_ == Phase::kAlign && s.t >= t_ignition_ && delayed_ &&
        (s.t + lead_) - t_node_planned_ > late_limit_s_) {
        // A quarter of an orbit late the geometry the node was planned for is gone (a burn meant
        // for periapsis would land near the semi-minor axis): give it up rather than fly it.
        phase_ = Phase::kDone;
        aborted_ = true;
    }
    if (phase_ == Phase::kAlign && s.t >= t_ignition_) {
        if (delayed_) {
            // Late: re-plan the burn as centred at now + lead, where the node's own inertial
            // direction would be stale by the orbit's rotation since. That target turns with the
            // orbit, at the RSW frame's rate (r x v) / r^2, which is fed to the attitude law.
            const State now{s.t, s.r_eci, s.v_eci};
            node_.t = s.t + lead_;
            aim(node_state(now, node_, mu_));
            const double rn = s.r_eci.norm();
            target_.w_eci_radps = s.r_eci.cross(s.v_eci) / (rn * rn);
        }
        // Aligned: attitude within tolerance and turning at the target's rate.
        const Eigen::Vector3d w_rel = s.w_body - s.q_body_to_eci.conjugate() * target_.w_eci_radps;
        const bool aligned = attitude_error_rad(s.q_body_to_eci, target_.q_body_to_eci).norm() <=
                                 kIgnitionAlign_rad &&
                             w_rel.norm() <= kIgnitionRate_radps;
        if (aligned) {
            phase_ = Phase::kBurn;
            m_ignition_ = m;
            if (cutoff_.kind != BurnCutoff::Kind::kDeltaV) {
                cutoff_side_ = cutoff_.radius_m - apsis_radius(last_) >= 0.0 ? 1.0 : -1.0;
            }
            target_.w_eci_radps.setZero();  // freeze the burn direction in inertial space
            t_ignition_ = s.t;
        } else {
            delayed_ = true;  // keep slewing; ignite as soon as aligned
        }
    }
    double throttle = 0.0;
    if (phase_ == Phase::kBurn) {
        achieved_ = ve_ * std::log(m_ignition_ / m);  // rocket equation: dv from propellant used
        const double remaining = remaining_dv(s);
        if (!(remaining > kCutoffResidual_mps) || s.propellant_kg <= 0.0) {
            phase_ = Phase::kDone;
        } else {
            // Full thrust, then just the remainder over the last hold: a = T / m.
            throttle = std::min(1.0, remaining / ((thrust_ / m) * hold_s_));
        }
        // Propellant used over the hold at this throttle, out of what is in the tank.
        m_hold_end_ =
            m - std::min(cfg.thrusters.front().mass_flow_kgps(throttle) * hold_s_, s.propellant_kg);
    }
    return attitude_command(cfg, s, target_, gains, throttle);
}

double ManeuverExecutor::remaining_dv(const VehicleState& s) const {
    if (cutoff_.kind == BurnCutoff::Kind::kDeltaV) {
        return dv_target_ - achieved_;
    }
    if (achieved_ >= 2.0 * dv_target_) {
        return 0.0;  // safety stop: something is badly off
    }
    const State now{s.t, s.r_eci, s.v_eci};
    const double f0 = apsis_radius(now);
    if (cutoff_side_ * (cutoff_.radius_m - f0) <= 0.0) {
        return 0.0;  // the apsis has reached (or crossed) the target from the side it started on
    }
    // Not there yet: one Newton step on f(dv) = apsis - target, with df/d(dv) by a 1 m/s
    // finite difference along the burn direction.
    State nudged = now;
    nudged.v_eci += dir_eci_;
    const double slope = apsis_radius(nudged) - f0;  // [m per m/s]
    const double remaining = (cutoff_.radius_m - f0) / slope;
    // A flat or wrong-way slope (e.g. near circular, where apoapsis and periapsis swap ends)
    // says nothing useful: fall back to the node's own dv (the 2x stop above still applies).
    return std::isfinite(remaining) && remaining > 0.0 ? remaining : dv_target_ - achieved_;
}

}  // namespace simcore

#include "simcore/gnc/momentum.hpp"

#include <algorithm>
#include <cmath>

#include <Eigen/QR>

#include "simcore/check.hpp"
#include "simcore/gnc/wheel_allocation.hpp"
#include "simcore/integrators/six_dof.hpp"

namespace simcore {

Eigen::Vector3d wheel_momentum_body(const VehicleConfig& cfg, const VehicleState& s) {
    validate_vehicle_state(cfg, s);
    Eigen::Vector3d h = Eigen::Vector3d::Zero();
    for (std::size_t j = 0; j < cfg.wheels.size(); ++j) {
        h += s.wheel_h_nms[j] * cfg.wheels[j].axis_body();
    }
    return h;
}

double max_wheel_saturation(const VehicleConfig& cfg, const VehicleState& s) {
    validate_vehicle_state(cfg, s);
    double sat = 0.0;
    for (std::size_t j = 0; j < cfg.wheels.size(); ++j) {
        const ReactionWheel& wh = cfg.wheels[j];
        sat = std::max(sat, std::abs(wh.relative_speed_radps(s.wheel_h_nms[j], s.w_body)) /
                                wh.max_speed_radps());
    }
    return sat;
}

bool desat_active(bool was_active, const VehicleConfig& cfg, const VehicleState& s,
                  const DesatSettings& settings) {
    SIMCORE_CHECK(settings.stop_fraction >= 0.0 &&
                      settings.stop_fraction < settings.start_fraction &&
                      settings.start_fraction <= 1.0,
                  "need 0 <= stop < start <= 1");
    const double sat = max_wheel_saturation(cfg, s);
    return was_active ? sat >= settings.stop_fraction : sat >= settings.start_fraction;
}

std::vector<double> rcs_dump_command(const VehicleConfig& cfg, const VehicleState& s,
                                     const DesatSettings& settings) {
    SIMCORE_CHECK(std::isfinite(settings.gain_per_s) && settings.gain_per_s > 0.0,
                  "desat gain must be finite and > 0");
    std::vector<double> u(cfg.rcs.size(), 0.0);
    if (cfg.rcs.empty()) {
        return u;
    }
    const Eigen::Vector3d tau = -settings.gain_per_s * wheel_momentum_body(cfg, s);
    // Columns: the torque each RCS axis makes at u = 1. Then tau = M u, u = M+ tau.
    const auto n = static_cast<Eigen::Index>(cfg.rcs.size());
    Eigen::Matrix<double, 3, Eigen::Dynamic> M(3, n);
    for (Eigen::Index j = 0; j < n; ++j) {
        M.col(j) = cfg.rcs[static_cast<std::size_t>(j)].torque_body_nm(1.0);
    }
    Eigen::VectorXd v = M.completeOrthogonalDecomposition().pseudoInverse() * tau;
    // Scale down together (keeping the torque's direction) to the RCS limits and to what the
    // wheels can counter: they must make -tau_rcs on top of the attitude torque.
    const double peak = v.cwiseAbs().maxCoeff();
    double scale = peak > 1.0 ? 1.0 / peak : 1.0;
    scale = std::min(scale, kDumpWheelShare * wheel_torque_headroom(cfg, -tau));
    v *= scale;
    for (std::size_t j = 0; j < u.size(); ++j) {
        u[j] = v(static_cast<Eigen::Index>(j));
    }
    return u;
}

VehicleCommand attitude_command_with_desat(const VehicleConfig& cfg, const VehicleState& s,
                                           const AttitudeTarget& target, const AttitudeGains& gains,
                                           bool dumping, const DesatSettings& settings,
                                           double throttle) {
    VehicleCommand c = attitude_command(cfg, s, target, gains, throttle);
    if (!dumping || cfg.rcs.empty()) {
        return c;
    }
    c.rcs = rcs_dump_command(cfg, s, settings);
    Eigen::Vector3d tau_rcs = Eigen::Vector3d::Zero();
    for (std::size_t j = 0; j < cfg.rcs.size(); ++j) {
        tau_rcs += cfg.rcs[j].torque_body_nm(c.rcs[j]);
    }
    // The wheels make up the rest, so the body sees exactly tau_att.
    c.wheel_torque_nm =
        wheel_torques_for_body_torque(cfg, attitude_torque_body(cfg, s, target, gains) - tau_rcs);
    return c;
}

}  // namespace simcore

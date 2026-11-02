#include "simcore/gnc/attitude_control.hpp"

#include <cmath>

#include "simcore/check.hpp"
#include "simcore/gnc/lqr.hpp"
#include "simcore/gnc/wheel_allocation.hpp"
#include "simcore/integrators/six_dof.hpp"
#include "simcore/vehicle/reaction_wheel.hpp"

namespace simcore {

AttitudeTarget hold_target(HoldMode mode, const State& orbit) {
    const Eigen::Vector3d& r = orbit.r_eci;
    const Eigen::Vector3d& v = orbit.v_eci;
    const double rn = r.norm();
    SIMCORE_CHECK(rn > 0.0 && r.allFinite() && v.allFinite(), "orbit state must be finite, r != 0");
    const Eigen::Vector3d h = r.cross(v);
    SIMCORE_CHECK(h.norm() > 1e-12 * rn * v.norm(), "radial trajectory has no orbit frame");
    const Eigen::Vector3d r_hat = r / rn;
    const Eigen::Vector3d v_hat = v.normalized();
    const Eigen::Vector3d h_hat = h.normalized();

    // Initialised (to prograde) although the switch below is exhaustive: GCC can't rule out an
    // out-of-range enum value and warns that x may be used uninitialised.
    Eigen::Vector3d x = v_hat;
    Eigen::Vector3d ref = r_hat;  // roll reference: body z toward radial-out
    switch (mode) {
        case HoldMode::kPrograde:
            x = v_hat;
            break;
        case HoldMode::kRetrograde:
            x = -v_hat;
            break;
        case HoldMode::kNormal:
            x = h_hat;
            break;
        case HoldMode::kAntiNormal:
            x = -h_hat;
            break;
        case HoldMode::kRadialOut:
            x = r_hat;
            ref = h_hat;
            break;
        case HoldMode::kRadialIn:
            x = -r_hat;
            ref = h_hat;
            break;
    }
    // Gram-Schmidt: z is ref with its x component removed, y completes a right-handed frame.
    const Eigen::Vector3d z = (ref - ref.dot(x) * x).normalized();
    const Eigen::Vector3d y = z.cross(x);
    Eigen::Matrix3d R;
    R.col(0) = x;
    R.col(1) = y;
    R.col(2) = z;
    return AttitudeTarget{Eigen::Quaterniond(R).normalized(), h / (rn * rn)};
}

Eigen::Vector3d attitude_error_rad(const Eigen::Quaterniond& q_body_to_eci,
                                   const Eigen::Quaterniond& q_target_body_to_eci) {
    const Eigen::Quaterniond qe = q_target_body_to_eci.conjugate() * q_body_to_eci;
    const Eigen::Vector3d v = qe.vec();
    const double vn = v.norm();
    if (vn == 0.0) {
        return Eigen::Vector3d::Zero();
    }
    // atan2 rather than acos(w): accurate for small and large angles alike.
    const double angle = 2.0 * std::atan2(vn, std::abs(qe.w()));
    return (qe.w() < 0.0 ? -angle : angle) * (v / vn);
}

AttitudeGains pd_gains(const Eigen::Matrix3d& inertia_kgm2, const PdGains& g) {
    SIMCORE_CHECK(std::isfinite(g.wn_radps) && g.wn_radps > 0.0, "wn must be finite and > 0");
    SIMCORE_CHECK(std::isfinite(g.zeta) && g.zeta >= 0.0, "zeta must be finite and >= 0");
    return AttitudeGains{g.wn_radps * g.wn_radps * inertia_kgm2,
                         2.0 * g.zeta * g.wn_radps * inertia_kgm2};
}

AttitudeGains lqr_attitude_gains(const Eigen::Matrix3d& inertia_kgm2,
                                 const Eigen::Matrix<double, 6, 6>& Q, const Eigen::Matrix3d& R) {
    SIMCORE_CHECK(inertia_kgm2.allFinite(), "inertia must be finite");
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(6, 6);
    A.topRightCorner(3, 3) = Eigen::Matrix3d::Identity();  // d_theta' = d_w
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(6, 3);
    B.bottomRows(3) = inertia_kgm2.inverse();  // I d_w' = tau
    const Eigen::MatrixXd K = lqr_gain(A, B, Q, R);
    return AttitudeGains{K.leftCols(3), K.rightCols(3)};
}

AttitudeGains lqr_attitude_gains_bryson(const Eigen::Matrix3d& inertia_kgm2,
                                        const BrysonLimits& limits) {
    SIMCORE_CHECK(std::isfinite(limits.angle_rad) && limits.angle_rad > 0.0 &&
                      std::isfinite(limits.rate_radps) && limits.rate_radps > 0.0 &&
                      std::isfinite(limits.torque_nm) && limits.torque_nm > 0.0,
                  "Bryson limits must be finite and > 0");
    Eigen::Matrix<double, 6, 6> Q = Eigen::Matrix<double, 6, 6>::Zero();
    Q.diagonal().head<3>().setConstant(1.0 / (limits.angle_rad * limits.angle_rad));
    Q.diagonal().tail<3>().setConstant(1.0 / (limits.rate_radps * limits.rate_radps));
    const Eigen::Matrix3d R = Eigen::Matrix3d::Identity() / (limits.torque_nm * limits.torque_nm);
    return lqr_attitude_gains(inertia_kgm2, Q, R);
}

Eigen::Vector3d attitude_torque_body(const VehicleConfig& cfg, const VehicleState& s,
                                     const AttitudeTarget& target, const AttitudeGains& gains) {
    validate_vehicle_state(cfg, s);
    const Eigen::Matrix3d I = cfg.mass.inertia_kgm2(s.propellant_kg);
    const Eigen::Vector3d d_theta = attitude_error_rad(s.q_body_to_eci, target.q_body_to_eci);
    const Eigen::Vector3d w_target_body = s.q_body_to_eci.conjugate() * target.w_eci_radps;
    const Eigen::Vector3d d_w = s.w_body - w_target_body;
    const Eigen::Vector3d H = total_angular_momentum_body(I, cfg.wheels, s.w_body, s.wheel_h_nms);
    return -(gains.Kp * d_theta) - gains.Kd * d_w + s.w_body.cross(H);
}

VehicleCommand attitude_command(const VehicleConfig& cfg, const VehicleState& s,
                                const AttitudeTarget& target, const AttitudeGains& gains,
                                double throttle) {
    SIMCORE_CHECK(throttle >= 0.0 && throttle <= 1.0, "throttle must be in [0, 1]");
    const Eigen::Vector3d tau_b = attitude_torque_body(cfg, s, target, gains);
    VehicleCommand c;
    c.throttle.assign(cfg.thrusters.size(), 0.0);
    if (!c.throttle.empty()) {
        c.throttle.front() = throttle;
    }
    c.wheel_torque_nm = wheel_torques_for_body_torque(cfg, tau_b);
    return c;
}

VehicleCommand attitude_command(const VehicleConfig& cfg, const VehicleState& s,
                                const AttitudeTarget& target, const PdGains& gains,
                                double throttle) {
    validate_vehicle_state(cfg, s);
    return attitude_command(cfg, s, target, pd_gains(cfg.mass.inertia_kgm2(s.propellant_kg), gains),
                            throttle);
}

}  // namespace simcore

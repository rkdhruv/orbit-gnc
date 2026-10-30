#include "simcore/integrators/six_dof.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "simcore/check.hpp"

namespace simcore {

namespace {

// The integrated state as plain arrays (q as Eigen coefficients x, y, z, w).
struct Y {
    Eigen::Vector3d r, v;
    Eigen::Vector4d q;
    Eigen::Vector3d w;
    double mp;
    std::vector<double> h;
};

// y + c * k, component by component.
Y axpy(const Y& y, double c, const Y& k) {
    Y out{y.r + c * k.r, y.v + c * k.v, y.q + c * k.q, y.w + c * k.w, y.mp + c * k.mp, y.h};
    for (std::size_t j = 0; j < out.h.size(); ++j) {
        out.h[j] += c * k.h[j];
    }
    return out;
}

void validate(const VehicleConfig& cfg, const VehicleCommand& cmd, const VehicleState& s) {
    SIMCORE_CHECK(cmd.throttle.size() == cfg.thrusters.size(), "one throttle per thruster");
    SIMCORE_CHECK(cmd.wheel_torque_nm.size() == cfg.wheels.size(),
                  "one wheel torque command per wheel");
    SIMCORE_CHECK(s.wheel_h_nms.size() == cfg.wheels.size(), "one wheel momentum per wheel");
    for (const double u : cmd.throttle) {
        SIMCORE_CHECK(u >= 0.0 && u <= 1.0, "throttle must be in [0, 1]");
    }
    (void)cfg.mass.mass_kg(s.propellant_kg);  // propellant in [0, capacity]
}

// y' at (t, y). Whether the thrusters fire is decided once per step (`burning`, from the propellant
// at the start of the step), not per RK stage: a stage evaluated exactly at burnout would otherwise
// see an empty tank and drop a sixth of the step's thrust. Stage values of m_p can dip just below 0
// in the step containing burnout; mass properties are evaluated at max(m_p, 0).
Y rate(const VehicleConfig& cfg, const ForceModel& env_accel, const TorqueModel& env_torque,
       const VehicleCommand& cmd, bool burning, const Epoch& t, const Y& y) {
    const double mp = std::clamp(y.mp, 0.0, cfg.mass.capacity_kg());
    const double m = cfg.mass.mass_kg(mp);
    const Eigen::Matrix3d I = cfg.mass.inertia_kgm2(mp);

    Eigen::Vector3d f_body = Eigen::Vector3d::Zero();
    Eigen::Vector3d tau_thrust = Eigen::Vector3d::Zero();
    double mdot = 0.0;
    for (std::size_t i = 0; i < cfg.thrusters.size(); ++i) {
        const double u = burning ? cmd.throttle[i] : 0.0;
        f_body += cfg.thrusters[i].force_body_n(u);
        tau_thrust += cfg.thrusters[i].torque_body_nm(u);
        mdot += cfg.thrusters[i].mass_flow_kgps(u);
    }

    const Eigen::Quaterniond q_raw(y.q);
    const Eigen::Quaterniond q_unit = q_raw.normalized();
    const Eigen::Vector3d accel =
        env_accel.acceleration_eci(State{t, y.r, y.v}) + q_unit.toRotationMatrix() * f_body / m;
    const Eigen::Vector3d tau_ext =
        env_torque.torque_body(AttitudeState{t, q_unit, y.w}) + tau_thrust;

    std::vector<double> tau_motor(cfg.wheels.size());
    for (std::size_t j = 0; j < cfg.wheels.size(); ++j) {
        const ReactionWheel& wh = cfg.wheels[j];
        tau_motor[j] = wh.applied_motor_torque_nm(cmd.wheel_torque_nm[j],
                                                  wh.relative_speed_radps(y.h[j], y.w));
    }

    return Y{y.v,
             accel,
             quaternion_rate(q_raw, y.w).coeffs(),
             angular_acceleration_with_wheels(I, cfg.wheels, y.w, y.h, tau_ext, tau_motor),
             -mdot,
             tau_motor};
}

}  // namespace

VehicleState six_dof_rk4_step(const VehicleConfig& cfg, const ForceModel& env_accel,
                              const TorqueModel& env_torque, const VehicleCommand& cmd,
                              const VehicleState& s, double h_s) {
    validate(cfg, cmd, s);
    const double half = 0.5 * h_s;
    const Y y{s.r_eci, s.v_eci, s.q_body_to_eci.coeffs(), s.w_body, s.propellant_kg, s.wheel_h_nms};
    const bool burning = s.propellant_kg > 0.0;  // thrusters fire for the whole step, or not at all

    const Y k1 = rate(cfg, env_accel, env_torque, cmd, burning, s.t, y);
    const Y k2 = rate(cfg, env_accel, env_torque, cmd, burning, s.t + half, axpy(y, half, k1));
    const Y k3 = rate(cfg, env_accel, env_torque, cmd, burning, s.t + half, axpy(y, half, k2));
    const Y k4 = rate(cfg, env_accel, env_torque, cmd, burning, s.t + h_s, axpy(y, h_s, k3));

    // y1 = y + h/6 (k1 + 2 k2 + 2 k3 + k4), written per component in the same order as rk4_step
    // so the uncoupled case reproduces the orbit propagator to roundoff.
    const double c = h_s / 6.0;
    VehicleState out;
    out.t = s.t + h_s;
    out.r_eci = y.r + c * (k1.r + 2.0 * k2.r + 2.0 * k3.r + k4.r);
    out.v_eci = y.v + c * (k1.v + 2.0 * k2.v + 2.0 * k3.v + k4.v);
    out.q_body_to_eci =
        Eigen::Quaterniond(Eigen::Vector4d(y.q + c * (k1.q + 2.0 * k2.q + 2.0 * k3.q + k4.q)))
            .normalized();
    out.w_body = y.w + c * (k1.w + 2.0 * k2.w + 2.0 * k3.w + k4.w);
    // Clamp: the step that contains burnout can overshoot the empty tank slightly.
    out.propellant_kg = std::max(0.0, y.mp + c * (k1.mp + 2.0 * k2.mp + 2.0 * k3.mp + k4.mp));
    out.wheel_h_nms = y.h;
    for (std::size_t j = 0; j < y.h.size(); ++j) {
        out.wheel_h_nms[j] += c * (k1.h[j] + 2.0 * k2.h[j] + 2.0 * k3.h[j] + k4.h[j]);
    }
    return out;
}

VehicleState propagate_six_dof(const VehicleConfig& cfg, const ForceModel& env_accel,
                               const TorqueModel& env_torque, const VehicleCommand& cmd,
                               const VehicleState& s0, const Epoch& t_end, double h_max_s) {
    SIMCORE_CHECK(std::isfinite(h_max_s) && h_max_s > 0.0, "h_max_s must be finite and > 0");
    validate(cfg, cmd, s0);
    const double span_s = t_end - s0.t;
    if (span_s == 0.0) {
        return s0;
    }
    const double steps = std::ceil(std::abs(span_s) / h_max_s);
    SIMCORE_CHECK(steps <= 1e12, "too many 6-DOF steps (span / h_max > 1e12)");
    const long n = std::max(1L, static_cast<long>(steps));
    const double h = span_s / static_cast<double>(n);

    VehicleState s = s0;
    for (long k = 0; k < n; ++k) {
        s.t = s0.t + static_cast<double>(k) * h;  // schedule from s0, not accumulated
        s = six_dof_rk4_step(cfg, env_accel, env_torque, cmd, s, h);
    }
    s.t = t_end;
    return s;
}

}  // namespace simcore

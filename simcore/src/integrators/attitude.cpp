#include "simcore/integrators/attitude.hpp"

#include <algorithm>
#include <cmath>

#include "simcore/check.hpp"

namespace simcore {

namespace {

struct Rate {
    Eigen::Vector4d q;  // q_dot coefficients (Eigen order x, y, z, w)
    Eigen::Vector3d w;  // w_dot
};

// y' at (t, q, w). The torque model sees a unit quaternion even at RK stages, where q has drifted
// slightly off the unit sphere, and the frozen orbit at the stage time.
Rate attitude_rate(const RigidBody& body, const TorqueModel& torque, const State& orbit,
                   const Epoch& t, const Eigen::Vector4d& q, const Eigen::Vector3d& w) {
    const Eigen::Quaterniond qq(q);  // from coefficients (x, y, z, w)
    const AttitudeState s{t, qq.normalized(), w};
    const State o{t, orbit.r_eci, orbit.v_eci};
    return {quaternion_rate(qq, w).coeffs(),
            body.angular_acceleration_body(w, torque.torque_body(o, s))};
}

}  // namespace

AttitudeState attitude_rk4_step(const RigidBody& body, const TorqueModel& torque,
                                const State& orbit, const AttitudeState& s, double h_s) {
    const double half = 0.5 * h_s;
    const Eigen::Vector4d q = s.q_body_to_eci.coeffs();
    const Eigen::Vector3d& w = s.w_body;

    const Rate k1 = attitude_rate(body, torque, orbit, s.t, q, w);
    const Rate k2 =
        attitude_rate(body, torque, orbit, s.t + half, q + half * k1.q, w + half * k1.w);
    const Rate k3 =
        attitude_rate(body, torque, orbit, s.t + half, q + half * k2.q, w + half * k2.w);
    const Rate k4 = attitude_rate(body, torque, orbit, s.t + h_s, q + h_s * k3.q, w + h_s * k3.w);

    const double c = h_s / 6.0;
    const Eigen::Vector4d q1 = q + c * (k1.q + 2.0 * k2.q + 2.0 * k3.q + k4.q);
    const Eigen::Vector3d w1 = w + c * (k1.w + 2.0 * k2.w + 2.0 * k3.w + k4.w);
    // Renormalise: q_dot = 1/2 Omega(w) q is linear with eigenvalues +-i |w| / 2. With
    // theta = |w| h / 2, |R(i theta)| = 1 - theta^6 / 144 + ..., so RK4 shrinks |q| by ~theta^6 /
    // 144 per step. (theta^5 / 120 is the phase error, which renormalising doesn't touch.)
    return AttitudeState{s.t + h_s, Eigen::Quaterniond(q1).normalized(), w1};
}

AttitudeState propagate_attitude(const RigidBody& body, const TorqueModel& torque,
                                 const State& orbit, const AttitudeState& s0, const Epoch& t_end,
                                 double h_max_s) {
    SIMCORE_CHECK(std::isfinite(h_max_s) && h_max_s > 0.0, "h_max_s must be finite and > 0");
    const double span_s = t_end - s0.t;
    if (span_s == 0.0) {
        return s0;
    }
    const double steps = std::ceil(std::abs(span_s) / h_max_s);
    SIMCORE_CHECK(steps <= 1e12, "too many attitude steps (span / h_max > 1e12)");
    const long n = std::max(1L, static_cast<long>(steps));
    const double h = span_s / static_cast<double>(n);

    AttitudeState s = s0;
    for (long k = 0; k < n; ++k) {
        s.t = s0.t + static_cast<double>(k) * h;  // schedule from s0, not accumulated
        s = attitude_rk4_step(body, torque, orbit, s, h);
    }
    s.t = t_end;
    return s;
}

}  // namespace simcore

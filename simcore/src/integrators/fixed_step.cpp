#include "simcore/integrators/fixed_step.hpp"

#include <algorithm>
#include <cmath>

#include "simcore/check.hpp"

namespace simcore {

State verlet_step(const ForceModel& f, const State& s, double h_s) {
    const double half = 0.5 * h_s;
    // Kick: half a step of velocity under the force at the start.
    const Eigen::Vector3d v_half = s.v_eci + half * f.acceleration_eci(s);
    // Drift: a full step of position at the mid-step velocity.
    State out{s.t + h_s, s.r_eci + h_s * v_half, v_half};
    // Kick: the other half step of velocity under the force at the new position.
    out.v_eci = v_half + half * f.acceleration_eci(out);
    return out;
}

State rk4_step(const ForceModel& f, const State& s, double h_s) {
    const double half = 0.5 * h_s;
    // Stages k_i = (dr/dt, dv/dt) = (v, a) evaluated at the classical RK4 points.
    const Eigen::Vector3d k1r = s.v_eci;
    const Eigen::Vector3d k1v = f.acceleration_eci(s);

    const State s2{s.t + half, s.r_eci + half * k1r, s.v_eci + half * k1v};
    const Eigen::Vector3d k2r = s2.v_eci;
    const Eigen::Vector3d k2v = f.acceleration_eci(s2);

    const State s3{s.t + half, s.r_eci + half * k2r, s.v_eci + half * k2v};
    const Eigen::Vector3d k3r = s3.v_eci;
    const Eigen::Vector3d k3v = f.acceleration_eci(s3);

    const State s4{s.t + h_s, s.r_eci + h_s * k3r, s.v_eci + h_s * k3v};
    const Eigen::Vector3d k4r = s4.v_eci;
    const Eigen::Vector3d k4v = f.acceleration_eci(s4);

    // y1 = y + h/6 (k1 + 2 k2 + 2 k3 + k4): Simpson's rule weights on the stage slopes.
    const double w = h_s / 6.0;
    return State{s.t + h_s, s.r_eci + w * (k1r + 2.0 * k2r + 2.0 * k3r + k4r),
                 s.v_eci + w * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)};
}

State propagate_fixed(const ForceModel& f, const State& s0, const Epoch& t_end, double h_max_s,
                      FixedStepMethod method) {
    SIMCORE_CHECK(std::isfinite(h_max_s) && h_max_s > 0.0, "h_max_s must be finite and > 0");
    const double span_s = t_end - s0.t;
    if (span_s == 0.0) {
        return s0;
    }
    const double steps = std::ceil(std::abs(span_s) / h_max_s);
    SIMCORE_CHECK(steps <= 1e12, "too many fixed steps (span / h_max > 1e12)");
    const long n = std::max(1L, static_cast<long>(steps));
    const double h = span_s / static_cast<double>(n);

    State s = s0;
    for (long k = 0; k < n; ++k) {
        s.t = s0.t + static_cast<double>(k) * h;  // schedule from s0, not accumulated
        s = (method == FixedStepMethod::kVelocityVerlet) ? verlet_step(f, s, h) : rk4_step(f, s, h);
    }
    s.t = t_end;
    return s;
}

}  // namespace simcore

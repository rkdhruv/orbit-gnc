#include "simcore/gnc/transfers.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>

#include <Eigen/Geometry>

#include "simcore/check.hpp"
#include "simcore/orbits/elements.hpp"
#include "simcore/orbits/kepler.hpp"

namespace simcore {

namespace {

constexpr double kTwoPi = 2.0 * std::numbers::pi;

KeplerianElements elliptic_elements(const State& s, double mu) {
    SIMCORE_CHECK(has_conic_elements(s, mu), "orbit has no conic elements");
    const KeplerianElements el = elements_from_state(s, mu);
    SIMCORE_CHECK(el.e < 1.0, "orbit must be elliptic");
    return el;
}

double wrap_2pi(double x) {
    x = std::fmod(x, kTwoPi);
    return x < 0.0 ? x + kTwoPi : x;
}

}  // namespace

double time_to_true_anomaly_s(const State& s, double nu_target_rad, double mu) {
    SIMCORE_CHECK(std::isfinite(nu_target_rad), "target anomaly must be finite");
    const KeplerianElements el = elliptic_elements(s, mu);
    const double n = std::sqrt(mu / (el.a_m * el.a_m * el.a_m));
    const double dM = wrap_2pi(mean_from_true_anomaly(nu_target_rad, el.e) -
                               mean_from_true_anomaly(el.nu_rad, el.e));
    return dM / n;
}

HohmannPlan hohmann_transfer(const State& now, const Epoch& t_burn, double r_target_m, double mu) {
    SIMCORE_CHECK(std::isfinite(r_target_m) && r_target_m > 0.0,
                  "target radius must be finite and > 0");
    const State at = kepler_propagate(now, t_burn, mu);
    const double r1 = at.r_eci.norm();
    const double v1 = at.v_eci.norm();
    SIMCORE_CHECK(std::abs(at.r_eci.dot(at.v_eci)) <= 1e-3 * r1 * v1,
                  "a Hohmann burn needs a horizontal velocity (circular orbit or an apsis)");
    const double r2 = r_target_m;
    const double a_t = 0.5 * (r1 + r2);
    HohmannPlan p{};
    p.dv1_mps = std::sqrt(mu * (2.0 / r1 - 1.0 / a_t)) - v1;
    p.dv2_mps = std::sqrt(mu / r2) - std::sqrt(mu * (2.0 / r2 - 1.0 / a_t));
    p.transfer_s = std::numbers::pi * std::sqrt(a_t * a_t * a_t / mu);
    p.first = ManeuverNode{t_burn, Eigen::Vector3d(0.0, p.dv1_mps, 0.0)};
    p.second = ManeuverNode{t_burn + p.transfer_s, Eigen::Vector3d(0.0, p.dv2_mps, 0.0)};
    return p;
}

ManeuverNode circularize_node(const State& now, bool at_apoapsis, double mu) {
    (void)elliptic_elements(now, mu);
    const double dt = time_to_true_anomaly_s(now, at_apoapsis ? std::numbers::pi : 0.0, mu);
    const Epoch t = now.t + dt;
    const State at = kepler_propagate(now, t, mu);
    const double r = at.r_eci.norm();
    return ManeuverNode{t, Eigen::Vector3d(0.0, std::sqrt(mu / r) - at.v_eci.norm(), 0.0)};
}

ManeuverNode plane_change_node(const State& now, double i_target_rad, double mu) {
    SIMCORE_CHECK(
        std::isfinite(i_target_rad) && i_target_rad >= 0.0 && i_target_rad <= std::numbers::pi,
        "target inclination must be in [0, pi]");
    const KeplerianElements el = elliptic_elements(now, mu);
    SIMCORE_CHECK(el.i_rad >= 1e-6 && el.i_rad <= std::numbers::pi - 1e-6,
                  "an equatorial orbit has no line of nodes");
    // Equator crossings: argument of latitude u = argp + nu = 0 (ascending) or pi (descending).
    const double t_asc = time_to_true_anomaly_s(now, wrap_2pi(-el.argp_rad), mu);
    const double t_desc = time_to_true_anomaly_s(now, wrap_2pi(std::numbers::pi - el.argp_rad), mu);
    const Epoch t = now.t + std::min(t_asc, t_desc);
    const State at = kepler_propagate(now, t, mu);
    const Eigen::Matrix3d M = rsw_to_eci(at);
    const double v = at.v_eci.norm();
    const double di = std::abs(i_target_rad - el.i_rad);
    // Both rotation senses about r_hat; keep the one whose inclination lands on the target.
    Eigen::Vector3d best = Eigen::Vector3d::Zero();
    double best_err = std::numeric_limits<double>::infinity();
    for (const double sign : {1.0, -1.0}) {
        const Eigen::Vector3d dv_rsw(0.0, v * (std::cos(di) - 1.0), sign * v * std::sin(di));
        State after = at;
        after.v_eci += M * dv_rsw;
        const double err = std::abs(elements_from_state(after, mu).i_rad - i_target_rad);
        if (err < best_err) {
            best_err = err;
            best = dv_rsw;
        }
    }
    return ManeuverNode{t, best};
}

}  // namespace simcore

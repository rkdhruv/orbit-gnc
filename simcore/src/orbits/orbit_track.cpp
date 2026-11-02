#include "simcore/orbits/orbit_track.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

#include <Eigen/Geometry>  // cross(): declared in Core, defined in Geometry

#include "simcore/check.hpp"

namespace simcore {

std::vector<Eigen::Vector3d> sample_orbit_eci(const State& s, int n_points, double r_max_m,
                                              double mu) {
    SIMCORE_CHECK(n_points >= 3, "need at least 3 points");
    SIMCORE_CHECK(std::isfinite(r_max_m) && r_max_m > 0.0, "r_max must be finite and > 0");
    const double r = s.r_eci.norm();
    SIMCORE_CHECK(r > 0.0, "position must be nonzero");
    const Eigen::Vector3d h = s.r_eci.cross(s.v_eci);
    SIMCORE_CHECK(h.norm() > 1e-12 * r * s.v_eci.norm(), "radial trajectory has no orbit plane");

    const Eigen::Vector3d r_hat = s.r_eci / r;
    const Eigen::Vector3d h_hat = h.normalized();
    const double p = h.squaredNorm() / mu;  // semi-latus rectum
    const Eigen::Vector3d e_vec = s.v_eci.cross(h) / mu - r_hat;
    const double e = e_vec.norm();
    const Eigen::Vector3d P = (e < 1e-12) ? r_hat : Eigen::Vector3d(e_vec / e);
    const Eigen::Vector3d Q = h_hat.cross(P);

    const auto point = [&](double nu) -> Eigen::Vector3d {
        return (p / (1.0 + e * std::cos(nu))) * (std::cos(nu) * P + std::sin(nu) * Q);
    };

    std::vector<Eigen::Vector3d> pts;
    pts.reserve(static_cast<std::size_t>(n_points));
    if (e < 1.0) {
        const double nu0 = std::atan2(r_hat.dot(Q), r_hat.dot(P));  // current true anomaly
        const double step = 2.0 * std::numbers::pi / n_points;
        for (int k = 0; k < n_points; ++k) {
            pts.push_back(point(nu0 + step * k));
        }
        return pts;
    }
    if (p / (1.0 + e) >= r_max_m) {
        return pts;  // the whole trajectory lies outside r_max
    }
    // Clamp guards roundoff at the edge; for e >= 1 the argument is always >= -1/e >= -1.
    const double nu_max = std::acos(std::clamp((p / r_max_m - 1.0) / e, -1.0, 1.0));
    const double step = 2.0 * nu_max / (n_points - 1);
    for (int k = 0; k < n_points; ++k) {
        pts.push_back(point(-nu_max + step * k));
    }
    return pts;
}

}  // namespace simcore

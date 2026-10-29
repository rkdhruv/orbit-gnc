#include "simcore/dynamics/j2_gravity.hpp"

#include <cmath>

#include "simcore/check.hpp"

namespace simcore {

J2Gravity::J2Gravity(double mu, double radius_m, double j2) {
    SIMCORE_CHECK(std::isfinite(mu) && mu > 0.0, "mu must be finite and > 0");
    SIMCORE_CHECK(std::isfinite(radius_m) && radius_m > 0.0, "radius must be finite and > 0");
    SIMCORE_CHECK(std::isfinite(j2), "j2 must be finite");
    k_ = 1.5 * j2 * mu * radius_m * radius_m;
}

Eigen::Vector3d J2Gravity::acceleration_eci(const State& s) const {
    const Eigen::Vector3d& r = s.r_eci;
    const double r2 = r.squaredNorm();
    SIMCORE_CHECK(std::isfinite(r2) && r2 > 0.0, "J2 undefined at r = 0 or for non-finite r");
    const double z2_r2 = r.z() * r.z() / r2;  // sin^2 of geocentric latitude
    const double r5 = r2 * r2 * std::sqrt(r2);
    const double c = -k_ / r5;
    // The z component gets 3 instead of 1: d/dz of the z^2 term in V_J2 adds 2 z / r^2.
    return {c * r.x() * (1.0 - 5.0 * z2_r2), c * r.y() * (1.0 - 5.0 * z2_r2),
            c * r.z() * (3.0 - 5.0 * z2_r2)};
}

}  // namespace simcore

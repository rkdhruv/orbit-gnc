#include "simcore/dynamics/point_mass_gravity.hpp"

#include <cmath>

#include "simcore/check.hpp"

namespace simcore {

PointMassGravity::PointMassGravity(double mu) : mu_(mu) {
    SIMCORE_CHECK(std::isfinite(mu) && mu > 0.0, "mu must be finite and positive");
}

// F = -G M m r_hat / r^2; dividing by m and writing mu = G M gives
// a = -mu r_hat / r^2 = -mu r / r^3, the negative gradient of U = -mu / r.
// Independent of time and velocity.
Eigen::Vector3d PointMassGravity::acceleration_eci(const State& s) const {
    const double r2 = s.r_eci.squaredNorm();
    SIMCORE_CHECK(std::isfinite(r2), "position must be finite");  // catches NaN/inf components
    const double k = mu_ / (r2 * std::sqrt(r2));                  // mu / |r|^3
    // k is +inf at r = 0 (mu/0) and when |r|^3 underflows. With k and r finite, every component
    // of k*r is bounded by k|r| = mu/|r|^2 <= max(k, mu), so the result is finite too.
    SIMCORE_CHECK(std::isfinite(k), "gravity undefined at r = 0 (or |r| too small)");
    return -k * s.r_eci;
}

}  // namespace simcore

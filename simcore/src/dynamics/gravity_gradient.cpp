#include "simcore/dynamics/gravity_gradient.hpp"

#include <cmath>

#include <Eigen/Geometry>

#include "simcore/check.hpp"

namespace simcore {

GravityGradientTorque::GravityGradientTorque(const Eigen::Matrix3d& inertia_kgm2, double mu)
    : inertia_(RigidBody(inertia_kgm2).inertia_body()), mu_(mu) {
    SIMCORE_CHECK(std::isfinite(mu_) && mu_ > 0.0, "mu must be finite and > 0");
}

Eigen::Vector3d GravityGradientTorque::torque_body(const State& orbit,
                                                   const AttitudeState& att) const {
    const double r = orbit.r_eci.norm();
    SIMCORE_CHECK(orbit.r_eci.allFinite() && r > 0.0, "position must be finite and nonzero");
    const Eigen::Vector3d r_hat_b = att.q_body_to_eci.conjugate() * (orbit.r_eci / r);
    return (3.0 * mu_ / (r * r * r)) * r_hat_b.cross(inertia_ * r_hat_b);
}

}  // namespace simcore

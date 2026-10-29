#include "simcore/dynamics/rigid_body.hpp"

#include <cmath>

#include <Eigen/Eigenvalues>

#include "simcore/check.hpp"

namespace simcore {

RigidBody::RigidBody(const Eigen::Matrix3d& inertia_body_kgm2) {
    SIMCORE_CHECK(inertia_body_kgm2.allFinite(), "inertia must be finite");
    const double scale = inertia_body_kgm2.norm();
    SIMCORE_CHECK((inertia_body_kgm2 - inertia_body_kgm2.transpose()).norm() <= 1e-12 * scale,
                  "inertia must be symmetric");
    // Symmetrise away roundoff (e.g. from R I R^T) so the stored tensor is exactly symmetric.
    inertia_ = 0.5 * (inertia_body_kgm2 + inertia_body_kgm2.transpose());

    // Principal moments = eigenvalues of the symmetric tensor, sorted ascending.
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(inertia_, Eigen::EigenvaluesOnly);
    const Eigen::Vector3d& p = eig.eigenvalues();
    SIMCORE_CHECK(p(0) > 0.0, "inertia must be positive definite");
    // I_max = integral of (r^2 - r_max_axis^2) dm <= sum of the other two moments.
    SIMCORE_CHECK(p(2) <= (p(0) + p(1)) * (1.0 + 1e-12),
                  "principal moments violate the triangle inequality");
    inertia_inv_ = inertia_.inverse();
}

RigidBody RigidBody::box(double mass_kg, double x_m, double y_m, double z_m) {
    SIMCORE_CHECK(std::isfinite(mass_kg) && mass_kg > 0.0, "mass must be finite and > 0");
    SIMCORE_CHECK(std::isfinite(x_m) && x_m > 0.0 && std::isfinite(y_m) && y_m > 0.0 &&
                      std::isfinite(z_m) && z_m > 0.0,
                  "box edges must be finite and > 0");
    const double k = mass_kg / 12.0;
    return RigidBody(Eigen::Vector3d(k * (y_m * y_m + z_m * z_m), k * (x_m * x_m + z_m * z_m),
                                     k * (x_m * x_m + y_m * y_m))
                         .asDiagonal());
}

Eigen::Vector3d RigidBody::angular_momentum_body(const Eigen::Vector3d& w_body) const {
    return inertia_ * w_body;
}

double RigidBody::rotational_kinetic_energy(const Eigen::Vector3d& w_body) const {
    return 0.5 * w_body.dot(inertia_ * w_body);
}

Eigen::Vector3d RigidBody::angular_acceleration_body(const Eigen::Vector3d& w_body,
                                                     const Eigen::Vector3d& torque_body_nm) const {
    return inertia_inv_ * (torque_body_nm - w_body.cross(inertia_ * w_body));
}

Eigen::Quaterniond quaternion_rate(const Eigen::Quaterniond& q_body_to_eci,
                                   const Eigen::Vector3d& w_body) {
    // Hamilton product with the pure quaternion [0, w], scaled by 1/2.
    const Eigen::Quaterniond p =
        q_body_to_eci * Eigen::Quaterniond(0.0, w_body.x(), w_body.y(), w_body.z());
    return Eigen::Quaterniond(0.5 * p.coeffs());
}

}  // namespace simcore

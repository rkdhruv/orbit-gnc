#include "simcore/vehicle/mass_properties.hpp"

#include <cmath>

#include <Eigen/Eigenvalues>

#include "simcore/check.hpp"

namespace simcore {

MassProperties::MassProperties(double dry_mass_kg, const Eigen::Matrix3d& dry_inertia_kgm2,
                               const PropellantTank& tank)
    : dry_mass_(dry_mass_kg),
      // RigidBody validates the dry tensor and returns it exactly symmetrised.
      dry_inertia_(RigidBody(dry_inertia_kgm2).inertia_body()),
      tank_(tank) {
    SIMCORE_CHECK(std::isfinite(dry_mass_) && dry_mass_ > 0.0, "dry mass must be finite and > 0");
    SIMCORE_CHECK(std::isfinite(tank_.capacity_kg) && tank_.capacity_kg > 0.0,
                  "tank capacity must be finite and > 0");
    const Eigen::Matrix3d& T = tank_.inertia_full_kgm2;
    SIMCORE_CHECK(T.allFinite(), "tank inertia must be finite");
    const double scale = T.norm();
    SIMCORE_CHECK((T - T.transpose()).norm() <= 1e-12 * scale, "tank inertia must be symmetric");
    // eval(): T aliases the destination, and the transpose must read the unmodified matrix.
    tank_.inertia_full_kgm2 = (0.5 * (T + T.transpose())).eval();
    // Physically realisable: I = integral (r^2 1 - r r^T) dm, so S = 1/2 tr(I) 1 - I is the second
    // moment integral r r^T dm, which is PSD. This implies I is PSD and each principal moment is at
    // most the sum of the other two. S is linear in I, so a physical tank plus a physical dry body
    // gives a physical I(m_p) at every fill level.
    const Eigen::Matrix3d& Ts = tank_.inertia_full_kgm2;
    const Eigen::Matrix3d S = 0.5 * Ts.trace() * Eigen::Matrix3d::Identity() - Ts;
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(S, Eigen::EigenvaluesOnly);
    SIMCORE_CHECK(eig.eigenvalues()(0) >= -1e-12 * scale,
                  "tank inertia must be physically realisable (1/2 tr(I) 1 - I PSD)");
}

void MassProperties::check_propellant(double propellant_kg) const {
    SIMCORE_CHECK(propellant_kg >= 0.0 && propellant_kg <= tank_.capacity_kg,
                  "propellant mass must be in [0, capacity]");  // NaN fails too
}

double MassProperties::mass_kg(double propellant_kg) const {
    check_propellant(propellant_kg);
    return dry_mass_ + propellant_kg;
}

Eigen::Matrix3d MassProperties::inertia_kgm2(double propellant_kg) const {
    check_propellant(propellant_kg);
    return dry_inertia_ + (propellant_kg / tank_.capacity_kg) * tank_.inertia_full_kgm2;
}

RigidBody MassProperties::rigid_body(double propellant_kg) const {
    return RigidBody(inertia_kgm2(propellant_kg));
}

}  // namespace simcore

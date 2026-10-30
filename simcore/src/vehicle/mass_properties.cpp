#include "simcore/vehicle/mass_properties.hpp"

#include <cmath>

#include <Eigen/Eigenvalues>

#include "simcore/check.hpp"

namespace simcore {

MassProperties::MassProperties(double dry_mass_kg, const Eigen::Matrix3d& dry_inertia_kgm2,
                               const PropellantTank& tank)
    : dry_mass_(dry_mass_kg), dry_inertia_(dry_inertia_kgm2), tank_(tank) {
    SIMCORE_CHECK(std::isfinite(dry_mass_) && dry_mass_ > 0.0, "dry mass must be finite and > 0");
    SIMCORE_CHECK(std::isfinite(tank_.capacity_kg) && tank_.capacity_kg > 0.0,
                  "tank capacity must be finite and > 0");
    (void)RigidBody(dry_inertia_);  // validates the dry tensor
    const Eigen::Matrix3d& T = tank_.inertia_full_kgm2;
    SIMCORE_CHECK(T.allFinite(), "tank inertia must be finite");
    const double scale = T.norm();
    SIMCORE_CHECK((T - T.transpose()).norm() <= 1e-12 * scale, "tank inertia must be symmetric");
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(T, Eigen::EigenvaluesOnly);
    SIMCORE_CHECK(eig.eigenvalues()(0) >= -1e-12 * scale,
                  "tank inertia must be positive semi-definite");
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

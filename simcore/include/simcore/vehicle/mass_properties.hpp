#pragma once

#include <Eigen/Core>

#include "simcore/dynamics/rigid_body.hpp"

namespace simcore {

// Propellant tank: capacity and the propellant's inertia about the vehicle CoM when full.
struct PropellantTank {
    double capacity_kg;
    Eigen::Matrix3d inertia_full_kgm2;  // body axes, about the vehicle CoM
};

// Mass properties as a function of the propellant left on board.
// Stated assumptions: the tank is centred on the vehicle's centre of mass (burning propellant
// never moves the CoM, so thruster torques are always about the same point), and the propellant's
// inertia scales with its mass (a uniformly emptying tank):
//   m(m_p) = m_dry + m_p,   I(m_p) = I_dry + (m_p / capacity) I_tank,full
class MassProperties {
public:
    // Throws std::invalid_argument unless dry mass and capacity are finite and > 0, the dry
    // inertia is a valid rigid-body tensor and the full-tank inertia is symmetric positive
    // semi-definite.
    MassProperties(double dry_mass_kg, const Eigen::Matrix3d& dry_inertia_kgm2,
                   const PropellantTank& tank);

    [[nodiscard]] double capacity_kg() const { return tank_.capacity_kg; }
    [[nodiscard]] double dry_mass_kg() const { return dry_mass_; }

    // Propellant mass must be in [0, capacity] (throws otherwise).
    [[nodiscard]] double mass_kg(double propellant_kg) const;
    [[nodiscard]] Eigen::Matrix3d inertia_kgm2(double propellant_kg) const;
    [[nodiscard]] RigidBody rigid_body(double propellant_kg) const;

private:
    void check_propellant(double propellant_kg) const;

    double dry_mass_;
    Eigen::Matrix3d dry_inertia_;
    PropellantTank tank_;
};

}  // namespace simcore

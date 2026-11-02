#pragma once

#include <Eigen/Core>

#include "simcore/constants.hpp"
#include "simcore/dynamics/rigid_body.hpp"

namespace simcore {

// Gravity-gradient torque: the side of the body nearer the Earth is pulled slightly harder than
// the far side. For a body small compared with its orbit radius the net torque about the centre
// of mass is (Hughes, "Spacecraft Attitude Dynamics", 8.1)
//   tau_b = (3 mu / r^3) r_hat_b x (I r_hat_b),   r_hat_b = R(q)^T r_hat_eci,
// zero when r_hat_b lies along a principal axis, and always perpendicular to r_hat_b. It drives
// the libration of an Earth-pointing body and loads the reaction wheels of a held one.
// Stated simplification: the inertia is fixed at construction (like drag's fixed mass, since a
// torque model doesn't see the propellant load). Not compatible with on-rails coasting, which
// requires zero environment torque.
class GravityGradientTorque final : public TorqueModel {
public:
    // Throws std::invalid_argument unless the inertia is a valid rigid-body tensor (RigidBody).
    explicit GravityGradientTorque(const Eigen::Matrix3d& inertia_kgm2, double mu = earth::mu);

    // Throws std::invalid_argument for r = 0 or a non-finite position.
    [[nodiscard]] Eigen::Vector3d torque_body(const State& orbit,
                                              const AttitudeState& att) const override;

private:
    Eigen::Matrix3d inertia_;
    double mu_;
};

}  // namespace simcore

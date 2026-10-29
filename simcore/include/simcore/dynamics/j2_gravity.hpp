#pragma once

#include "simcore/constants.hpp"
#include "simcore/dynamics/force_model.hpp"

namespace simcore {

// J2 (oblateness) perturbation of a body whose spin axis is ECI z (precession and nutation
// neglected). Returns the perturbation only; add PointMassGravity for the central term.
//   V_J2 = mu J2 R^2 / (2 r^3) (3 z^2 / r^2 - 1)   (potential energy per unit mass)
//   a = -grad V_J2 = -(3/2) J2 mu R^2 / r^5 [x (1 - 5 z^2/r^2), y (1 - 5 z^2/r^2),
//                                            z (3 - 5 z^2/r^2)]
// The oblate bulge pulls harder at the equator and less at the poles; its main secular effect is
// nodal regression, dRAAN/dt = -1.5 n J2 (R/p)^2 cos i.
class J2Gravity final : public ForceModel {
public:
    // Throws std::invalid_argument unless mu > 0, radius > 0 and j2 are finite.
    explicit J2Gravity(double mu = earth::mu, double radius_m = earth::radius,
                       double j2 = earth::j2);

    // Throws std::invalid_argument at r = 0 or for non-finite r.
    [[nodiscard]] Eigen::Vector3d acceleration_eci(const State& s) const override;

private:
    double k_;  // (3/2) J2 mu R^2 [m^5/s^2]
};

}  // namespace simcore

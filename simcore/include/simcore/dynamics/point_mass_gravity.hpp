#pragma once

#include "simcore/constants.hpp"
#include "simcore/dynamics/force_model.hpp"

namespace simcore {

// Newtonian gravity of a spherically symmetric body: a = -mu * r / |r|^3.
class PointMassGravity final : public ForceModel {
public:
    // Throws std::invalid_argument unless mu is finite and > 0.
    explicit PointMassGravity(double mu = earth::mu);

    [[nodiscard]] double mu() const { return mu_; }

    // Throws std::invalid_argument when |r|^2 or mu/|r|^3 is not finite: the singularity r = 0,
    // non-finite r, |r| < ~1e-103 m (mu/r^3 overflows) or |r| > ~1.3e154 m (r^2 overflows).
    [[nodiscard]] Eigen::Vector3d acceleration_eci(const State& s) const override;

private:
    double mu_;  // [m^3/s^2]
};

}  // namespace simcore

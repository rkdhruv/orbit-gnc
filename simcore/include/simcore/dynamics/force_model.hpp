#pragma once

#include <Eigen/Core>

#include "simcore/dynamics/state.hpp"

namespace simcore {

// A source of acceleration on a point mass. All physics enters the integrators through this.
// Models take the whole State so later ones (drag: Cd*A/m, thrust: m(t)) can read vehicle
// properties without another interface change.
class ForceModel {
public:
    virtual ~ForceModel() = default;

    // Acceleration in ECI [m/s^2] for the given state.
    [[nodiscard]] virtual Eigen::Vector3d acceleration_eci(const State& s) const = 0;

protected:
    // Protected so `ForceModel& a = g1; a = g2;` can't slice; derived classes stay copyable.
    ForceModel() = default;
    ForceModel(const ForceModel&) = default;
    ForceModel& operator=(const ForceModel&) = default;
    ForceModel(ForceModel&&) = default;
    ForceModel& operator=(ForceModel&&) = default;
};

}  // namespace simcore

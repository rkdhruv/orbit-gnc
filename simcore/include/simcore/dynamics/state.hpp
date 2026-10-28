#pragma once

#include <Eigen/Core>

#include "simcore/orbits/epoch.hpp"

namespace simcore {

// Translational state of a point mass in the Earth-centred inertial frame.
struct State {
    Epoch t;
    Eigen::Vector3d r_eci{Eigen::Vector3d::Zero()};  // position [m]
    Eigen::Vector3d v_eci{Eigen::Vector3d::Zero()};  // velocity [m/s]
};

}  // namespace simcore

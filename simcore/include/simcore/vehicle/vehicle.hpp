#pragma once

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "simcore/orbits/epoch.hpp"
#include "simcore/vehicle/mass_properties.hpp"
#include "simcore/vehicle/reaction_wheel.hpp"
#include "simcore/vehicle/thruster.hpp"

namespace simcore {

// What the vehicle is made of. Thruster positions are relative to the centre of mass, which the
// tank model keeps fixed. The dry inertia in `mass` includes every wheel's full (locked) inertia;
// the wheel equations subtract each wheel's spin-axis part (see reaction_wheel.hpp).
struct VehicleConfig {
    MassProperties mass;
    std::vector<Thruster> thrusters;
    std::vector<ReactionWheel> wheels;
};

// Full 6-DOF state: translation (ECI), attitude (body-to-ECI, body rates), propellant on board and
// each wheel's axial angular momentum (same order as VehicleConfig::wheels).
struct VehicleState {
    Epoch t;
    Eigen::Vector3d r_eci{Eigen::Vector3d::Zero()};  // [m]
    Eigen::Vector3d v_eci{Eigen::Vector3d::Zero()};  // [m/s]
    Eigen::Quaterniond q_body_to_eci{Eigen::Quaterniond::Identity()};
    Eigen::Vector3d w_body{Eigen::Vector3d::Zero()};  // [rad/s]
    double propellant_kg{0.0};
    std::vector<double> wheel_h_nms;
};

// Control inputs, held constant over a propagation call (zero-order hold): one throttle in [0, 1]
// per thruster and one commanded motor torque per wheel (saturated by the wheel model).
struct VehicleCommand {
    std::vector<double> throttle;
    std::vector<double> wheel_torque_nm;
};

}  // namespace simcore

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "simcore/dynamics/state.hpp"
#include "simcore/orbits/epoch.hpp"

namespace simcore {

// Rotational state of a rigid body. q is the body-to-inertial (ECI) rotation, Hamilton convention
// (v_eci = q v_body q^-1, i.e. R(q) v_body); w is the angular velocity in the body frame.
struct AttitudeState {
    Epoch t;
    Eigen::Quaterniond q_body_to_eci{Eigen::Quaterniond::Identity()};
    Eigen::Vector3d w_body{Eigen::Vector3d::Zero()};  // [rad/s]
};

// Mass distribution of a rigid body, described by its inertia tensor about the centre of mass in
// body axes.
class RigidBody {
public:
    // Throws std::invalid_argument unless the tensor is finite, symmetric (to 1e-12 relative),
    // positive definite, and its principal moments satisfy the triangle inequality
    // I_max <= I_mid + I_min (true of every real mass distribution).
    explicit RigidBody(const Eigen::Matrix3d& inertia_body_kgm2);

    // Solid box of uniform density, edges along body x, y, z:
    // I = m/12 diag(y^2 + z^2, x^2 + z^2, x^2 + y^2).
    static RigidBody box(double mass_kg, double x_m, double y_m, double z_m);

    [[nodiscard]] const Eigen::Matrix3d& inertia_body() const { return inertia_; }

    // H = I w (body frame) [kg m^2/s].
    [[nodiscard]] Eigen::Vector3d angular_momentum_body(const Eigen::Vector3d& w_body) const;
    // T = 1/2 w . I w [J].
    [[nodiscard]] double rotational_kinetic_energy(const Eigen::Vector3d& w_body) const;
    // Euler's equations: w_dot = I^-1 (tau - w x (I w)). The w x (I w) term is the gyroscopic
    // coupling that makes a spinning body's rates exchange energy between axes.
    [[nodiscard]] Eigen::Vector3d angular_acceleration_body(
        const Eigen::Vector3d& w_body, const Eigen::Vector3d& torque_body_nm) const;

private:
    Eigen::Matrix3d inertia_;
    Eigen::Matrix3d inertia_inv_;
};

// Quaternion kinematics, q_dot = 1/2 q (x) [0, w_body]. Body-frame rates compose on the right of
// a body-to-inertial q. The result is a rate, not a unit quaternion.
Eigen::Quaterniond quaternion_rate(const Eigen::Quaterniond& q_body_to_eci,
                                   const Eigen::Vector3d& w_body);

// A source of torque on a rigid body, in body axes. All torques enter attitude propagation
// through this (the rotational counterpart of ForceModel). It sees the translational state as
// well as the attitude, since the usual disturbance torques depend on position and velocity
// (gravity gradient 3 mu / r^3 r_b x I r_b, aerodynamic, magnetic). Both arguments carry the
// same t.
class TorqueModel {
public:
    virtual ~TorqueModel() = default;
    [[nodiscard]] virtual Eigen::Vector3d torque_body(const State& orbit,
                                                      const AttitudeState& att) const = 0;

protected:
    TorqueModel() = default;
    TorqueModel(const TorqueModel&) = default;
    TorqueModel& operator=(const TorqueModel&) = default;
    TorqueModel(TorqueModel&&) = default;
    TorqueModel& operator=(TorqueModel&&) = default;
};

// A torque fixed in body axes (zero for torque-free motion).
class ConstantBodyTorque final : public TorqueModel {
public:
    explicit ConstantBodyTorque(const Eigen::Vector3d& torque_body_nm) : tau_(torque_body_nm) {}
    [[nodiscard]] Eigen::Vector3d torque_body(const State& /*orbit*/,
                                              const AttitudeState& /*att*/) const override {
        return tau_;
    }

private:
    Eigen::Vector3d tau_;
};

}  // namespace simcore

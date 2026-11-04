#include "simcore/gnc/wheel_allocation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <Eigen/QR>

#include "simcore/check.hpp"

namespace simcore {

namespace {

// -A+ tau_b, before limits.
Eigen::VectorXd allocate(const VehicleConfig& cfg, const Eigen::Vector3d& tau_b) {
    const auto n = static_cast<Eigen::Index>(cfg.wheels.size());
    Eigen::Matrix<double, 3, Eigen::Dynamic> A(3, n);
    for (Eigen::Index j = 0; j < n; ++j) {
        A.col(j) = cfg.wheels[static_cast<std::size_t>(j)].axis_body();
    }
    const Eigen::Matrix<double, Eigen::Dynamic, 3> A_pinv =
        A.completeOrthogonalDecomposition().pseudoInverse();
    return -(A_pinv * tau_b);
}

// Largest factor s with every |s tau_m,j| within its wheel's torque limit: min_j limit_j /
// |tau_m,j| over the nonzero entries (+infinity if all are zero).
double max_scale(const VehicleConfig& cfg, const Eigen::VectorXd& tau_m) {
    double scale = std::numeric_limits<double>::infinity();
    for (Eigen::Index j = 0; j < tau_m.size(); ++j) {
        if (tau_m(j) != 0.0) {
            const double limit = cfg.wheels[static_cast<std::size_t>(j)].max_torque_nm();
            scale = std::min(scale, limit / std::abs(tau_m(j)));
        }
    }
    return scale;
}

}  // namespace

std::vector<double> wheel_torques_for_body_torque(const VehicleConfig& cfg,
                                                  const Eigen::Vector3d& tau_body_nm) {
    SIMCORE_CHECK(tau_body_nm.allFinite(), "body torque must be finite");
    std::vector<double> out(cfg.wheels.size(), 0.0);
    if (cfg.wheels.empty() || tau_body_nm == Eigen::Vector3d::Zero()) {
        return out;
    }
    Eigen::VectorXd tau_m = allocate(cfg, tau_body_nm);
    tau_m *= std::min(1.0, max_scale(cfg, tau_m));  // scale down together, keeping direction
    for (std::size_t j = 0; j < out.size(); ++j) {
        out[j] = tau_m(static_cast<Eigen::Index>(j));
    }
    return out;
}

Eigen::VectorXd wheel_torques_unlimited(const VehicleConfig& cfg,
                                        const Eigen::Vector3d& tau_body_nm) {
    SIMCORE_CHECK(tau_body_nm.allFinite(), "body torque must be finite");
    if (cfg.wheels.empty()) {
        return Eigen::VectorXd(0);
    }
    return allocate(cfg, tau_body_nm);
}

double wheel_torque_headroom(const VehicleConfig& cfg, const Eigen::Vector3d& tau_body_nm) {
    SIMCORE_CHECK(tau_body_nm.allFinite(), "body torque must be finite");
    if (tau_body_nm == Eigen::Vector3d::Zero()) {
        return std::numeric_limits<double>::infinity();
    }
    if (cfg.wheels.empty()) {
        return 0.0;
    }
    return max_scale(cfg, allocate(cfg, tau_body_nm));
}

double wheel_axis_authority_nm(const VehicleConfig& cfg, int axis) {
    SIMCORE_CHECK(axis >= 0 && axis < 3, "axis must be 0, 1 or 2");
    if (cfg.wheels.empty()) {
        return 0.0;
    }
    const double T = max_scale(cfg, allocate(cfg, Eigen::Vector3d::Unit(axis)));
    return std::isinf(T) ? 0.0 : T;
}

}  // namespace simcore

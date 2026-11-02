#include "simcore/sim/manual_control.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <Eigen/QR>

#include "simcore/check.hpp"
#include "simcore/integrators/six_dof.hpp"
#include "simcore/sim/simulation.hpp"

namespace simcore {

namespace {

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

VehicleCommand manual_command(const VehicleConfig& cfg, const ManualInput& in,
                              const VehicleState& s) {
    SIMCORE_CHECK(in.throttle >= 0.0 && in.throttle <= 1.0, "throttle must be in [0, 1]");
    SIMCORE_CHECK(in.rotate.allFinite() && in.rotate.cwiseAbs().maxCoeff() <= 1.0,
                  "rotate inputs must be in [-1, 1]");
    validate_vehicle_state(cfg, s);

    VehicleCommand c;
    c.throttle.assign(cfg.thrusters.size(), 0.0);
    if (!c.throttle.empty()) {
        c.throttle.front() = in.throttle;
    }
    c.wheel_torque_nm.assign(cfg.wheels.size(), 0.0);
    if (cfg.wheels.empty()) {
        return c;
    }

    // Wheel torques for a wanted body torque: tau_m = -A+ tau_b.
    const auto n = static_cast<Eigen::Index>(cfg.wheels.size());
    Eigen::Matrix<double, 3, Eigen::Dynamic> A(3, n);
    for (Eigen::Index j = 0; j < n; ++j) {
        A.col(j) = cfg.wheels[static_cast<std::size_t>(j)].axis_body();
    }
    const Eigen::Matrix<double, Eigen::Dynamic, 3> A_pinv =
        A.completeOrthogonalDecomposition().pseudoInverse();
    const auto allocate = [&A_pinv](const Eigen::Vector3d& tau_b) -> Eigen::VectorXd {
        return -(A_pinv * tau_b);
    };

    Eigen::Vector3d tau_b = Eigen::Vector3d::Zero();
    if (in.rotate != Eigen::Vector3d::Zero()) {
        for (int k = 0; k < 3; ++k) {
            if (in.rotate(k) == 0.0) {
                continue;
            }
            // T_k: scale a unit torque request about axis k until the first wheel hits its
            // limit (0 if no wheel can act about that axis).
            const double T_k = max_scale(cfg, allocate(Eigen::Vector3d::Unit(k)));
            tau_b(k) = in.rotate(k) * (std::isinf(T_k) ? 0.0 : T_k);
        }
    } else if (in.rate_damping && s.w_body.norm() > Simulation::kRailsMaxRate_radps) {
        tau_b = -kRateDampingGain_per_s * (cfg.mass.inertia_kgm2(s.propellant_kg) * s.w_body);
    } else {
        return c;
    }

    Eigen::VectorXd tau_m = allocate(tau_b);
    tau_m *= std::min(1.0, max_scale(cfg, tau_m));  // scale down together, keeping direction
    for (Eigen::Index j = 0; j < n; ++j) {
        c.wheel_torque_nm[static_cast<std::size_t>(j)] = tau_m(j);
    }
    return c;
}

}  // namespace simcore

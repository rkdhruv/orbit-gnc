#include "simcore/vehicle/rcs.hpp"

#include <cmath>

#include "simcore/check.hpp"
#include "simcore/constants.hpp"

namespace simcore {

namespace {

void check_command(double u) {
    SIMCORE_CHECK(u >= -1.0 && u <= 1.0, "RCS command must be in [-1, 1]");  // NaN fails too
}

}  // namespace

RcsAxis::RcsAxis(const Eigen::Vector3d& axis_body, double max_torque_nm, double lever_arm_m,
                 double isp_s)
    : max_torque_(max_torque_nm), lever_arm_(lever_arm_m), isp_(isp_s) {
    SIMCORE_CHECK(axis_body.allFinite() && axis_body.norm() > 0.0,
                  "RCS axis must be finite and nonzero");
    SIMCORE_CHECK(std::isfinite(max_torque_) && max_torque_ > 0.0,
                  "RCS torque must be finite and > 0");
    SIMCORE_CHECK(std::isfinite(lever_arm_) && lever_arm_ > 0.0,
                  "RCS lever arm must be finite and > 0");
    SIMCORE_CHECK(std::isfinite(isp_) && isp_ > 0.0, "RCS Isp must be finite and > 0");
    a_ = axis_body.normalized();
}

Eigen::Vector3d RcsAxis::torque_body_nm(double u) const {
    check_command(u);
    return (u * max_torque_) * a_;
}

double RcsAxis::mass_flow_kgps(double u) const {
    check_command(u);
    // Total thrust |u| tau_max / L, at exhaust speed Isp g0.
    return std::abs(u) * max_torque_ / (lever_arm_ * isp_ * g0);
}

}  // namespace simcore

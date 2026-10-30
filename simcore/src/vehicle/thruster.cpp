#include "simcore/vehicle/thruster.hpp"

#include <cmath>

#include <Eigen/Geometry>  // cross(): declared in Core, defined in Geometry

#include "simcore/check.hpp"
#include "simcore/constants.hpp"

namespace simcore {

namespace {

void check_throttle(double throttle) {
    SIMCORE_CHECK(throttle >= 0.0 && throttle <= 1.0, "throttle must be in [0, 1]");  // NaN fails
}

}  // namespace

Thruster::Thruster(double max_thrust_n, double isp_s, const Eigen::Vector3d& position_body_m,
                   const Eigen::Vector3d& direction_body)
    : max_thrust_n_(max_thrust_n), isp_s_(isp_s), r_(position_body_m) {
    SIMCORE_CHECK(std::isfinite(max_thrust_n_) && max_thrust_n_ > 0.0,
                  "thrust must be finite and > 0");
    SIMCORE_CHECK(std::isfinite(isp_s_) && isp_s_ > 0.0, "Isp must be finite and > 0");
    SIMCORE_CHECK(r_.allFinite(), "thruster position must be finite");
    SIMCORE_CHECK(direction_body.allFinite() && direction_body.norm() > 0.0,
                  "thrust direction must be finite and nonzero");
    d_ = direction_body.normalized();
}

Eigen::Vector3d Thruster::force_body_n(double throttle) const {
    check_throttle(throttle);
    return (throttle * max_thrust_n_) * d_;
}

Eigen::Vector3d Thruster::torque_body_nm(double throttle) const {
    return r_.cross(force_body_n(throttle));
}

double Thruster::mass_flow_kgps(double throttle) const {
    check_throttle(throttle);
    return throttle * max_thrust_n_ / exhaust_velocity_mps();
}

double Thruster::exhaust_velocity_mps() const {
    return isp_s_ * g0;
}

double rocket_delta_v_mps(double exhaust_velocity_mps, double m0_kg, double mf_kg) {
    SIMCORE_CHECK(std::isfinite(exhaust_velocity_mps) && exhaust_velocity_mps > 0.0,
                  "exhaust velocity must be finite and > 0");
    SIMCORE_CHECK(std::isfinite(m0_kg) && mf_kg > 0.0 && m0_kg >= mf_kg,
                  "masses must satisfy m0 >= mf > 0");
    return exhaust_velocity_mps * std::log(m0_kg / mf_kg);
}

}  // namespace simcore

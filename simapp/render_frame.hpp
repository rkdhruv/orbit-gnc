#pragma once

// ECI (double, metres) to render space (float, scaled units). Header-only and raylib-free, so the
// tests can check it.
//
// Axes: raylib is right-handed with y up, so render (x, y, z) = (x_eci, z_eci, -y_eci) puts ECI
// north on screen up. That is a -90 deg rotation about x (det +1): nothing is mirrored.
// Floating origin: positions are made relative to an origin (the camera's focus) in double before
// scaling and rounding to float. float's 24-bit mantissa spaces values near 7e6 m 0.5 m apart, so
// rounding a whole ECI position would shake a 2 m spacecraft apart; rounding the small offset
// keeps sub-micrometre detail near the origin.

#include <cmath>
#include <numbers>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "simcore/constants.hpp"
#include "simcore/orbits/epoch.hpp"

namespace simapp {

inline Eigen::Matrix3d render_from_eci() {
    Eigen::Matrix3d M;
    M << 1.0, 0.0, 0.0,  //
        0.0, 0.0, 1.0,   //
        0.0, -1.0, 0.0;
    return M;
}

// (r - origin) / metres_per_unit in render axes. The subtraction is in double.
inline Eigen::Vector3f to_render(const Eigen::Vector3d& r_eci, const Eigen::Vector3d& origin_eci,
                                 double metres_per_unit) {
    return (render_from_eci() * ((r_eci - origin_eci) / metres_per_unit)).cast<float>();
}

// Body-to-render rotation M R(q), for drawing the spacecraft with its attitude.
inline Eigen::Matrix3f render_rotation(const Eigen::Quaterniond& q_body_to_eci) {
    return (render_from_eci() * q_body_to_eci.toRotationMatrix()).cast<float>();
}

// Earth's display rotation, theta = w_E (t - J2000) mod 2 pi in [0, 2 pi). A stand-in for sidereal
// time (proper GMST comes with the frames work in M7); display only.
inline double earth_display_angle(const simcore::Epoch& t) {
    const double two_pi = 2.0 * std::numbers::pi;
    const double theta =
        std::fmod(simcore::earth::rotation_rate_radps * (t - simcore::Epoch::j2000()), two_pi);
    return theta < 0.0 ? theta + two_pi : theta;
}

}  // namespace simapp

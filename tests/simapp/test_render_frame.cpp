// M3.2: the front end's mapping from ECI (double, metres) to render space (float, scaled units).
//
// Axes: raylib is right-handed with y up. Render (x, y, z) = (x_eci, z_eci, -y_eci), so ECI north
// (z) is up on screen. That is a -90 deg rotation about x (det +1), so nothing is mirrored.
// Floating origin: subtract the origin (the camera's focus, usually the spacecraft) in double,
// then scale and round to float. Rounding happens on the small offset, not the 7,000 km position:
// float has 24 significant bits, so at 6.8e6 m its spacing is 0.5 m, while an offset of 1 cm keeps
// ~1e-9 m. Converting first and subtracting after loses everything below half a metre.
// Earth's display angle: theta = w_E (t - J2000) mod 2 pi, a stand-in for sidereal time (proper
// GMST comes with the frames work in M7). Display only; nothing in the sim depends on it.

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>

#include <Eigen/Geometry>

#include "render_frame.hpp"
#include "simcore/constants.hpp"
#include "simcore/orbits/epoch.hpp"

namespace {

constexpr double kPi = std::numbers::pi;

}  // namespace

TEST(RenderFrame, EciNorthIsScreenUpAndTheMappingIsARotation) {
    const Eigen::Matrix3d M = simapp::render_from_eci();
    EXPECT_EQ(M * Eigen::Vector3d::UnitX(), Eigen::Vector3d(1.0, 0.0, 0.0));
    EXPECT_EQ(M * Eigen::Vector3d::UnitY(), Eigen::Vector3d(0.0, 0.0, -1.0));
    EXPECT_EQ(M * Eigen::Vector3d::UnitZ(), Eigen::Vector3d(0.0, 1.0, 0.0));
    EXPECT_EQ(M * M.transpose(), Eigen::Matrix3d::Identity());
    EXPECT_EQ(M.determinant(), 1.0);
}

// A point 1 cm from a spacecraft at ~7,000 km. Floating origin: offset (0.01, 0, 0) m with error
// <= half a double ulp at 6.8e6 m (4.7e-10) plus float rounding of 0.01 (4.7e-10): bound 1e-9.
// Converting to float first: both coordinates round to the same multiple of 0.5 m (or adjacent
// ones), so the offset comes out as 0 or 0.5 m: error >= 0.009 m.
TEST(RenderFrame, FloatingOriginKeepsCentimetresAtOrbitalDistances) {
    const Eigen::Vector3d r_sc(6.8e6, 1.2e6, 3.4e5);
    const Eigen::Vector3d part = r_sc + Eigen::Vector3d(0.01, 0.0, 0.0);
    const Eigen::Vector3f rel = simapp::to_render(part, r_sc, 1.0);
    EXPECT_LE((rel - Eigen::Vector3f(0.01F, 0.0F, 0.0F)).cast<double>().norm(), 1e-9);
    const Eigen::Vector3f naive = part.cast<float>() - r_sc.cast<float>();
    EXPECT_GE(std::abs(static_cast<double>(naive.x()) - 0.01), 0.009);
    // Scale: 1 render unit = 1 km, and the axis mapping applies to offsets too.
    EXPECT_EQ(simapp::to_render(r_sc + Eigen::Vector3d(0.0, 0.0, 1000.0), r_sc, 1000.0),
              Eigen::Vector3f(0.0F, 1.0F, 0.0F));
}

// Attitude in render axes: M R(q). q = 90 deg about ECI z turns body x onto ECI y, which is
// render -z; body z stays ECI north, render up. Float rounding of cos/sin 90 deg: bound 1e-7.
TEST(RenderFrame, AttitudeRotatesBodyAxesIntoRenderAxes) {
    const Eigen::Quaterniond q(Eigen::AngleAxisd(kPi / 2.0, Eigen::Vector3d::UnitZ()));
    const Eigen::Matrix3f R = simapp::render_rotation(q);
    EXPECT_LE((R * Eigen::Vector3f::UnitX() - Eigen::Vector3f(0.0F, 0.0F, -1.0F)).norm(), 1e-7F);
    EXPECT_LE((R * Eigen::Vector3f::UnitZ() - Eigen::Vector3f(0.0F, 1.0F, 0.0F)).norm(), 1e-7F);
}

// theta = w_E (t - J2000) mod 2 pi: 0 at J2000, pi / 2 a quarter sidereal day later
// (2 pi / w_E = 86,164.10 s with the WGS-84 rate), and back to ~0 after a full one. At t - J2000
// = 8.3e8 s the angle before the mod is 6e4 rad, with spacing ~7e-12 rad: bounds 1e-9.
TEST(RenderFrame, EarthDisplayAngleTurnsOncePerSiderealDay) {
    const double w = simcore::earth::rotation_rate_radps;
    const double sidereal_day = 2.0 * kPi / w;
    EXPECT_NEAR(sidereal_day, 86164.10, 0.01);  // hand value
    const simcore::Epoch j2000 = simcore::Epoch::j2000();
    EXPECT_EQ(simapp::earth_display_angle(j2000), 0.0);
    EXPECT_NEAR(simapp::earth_display_angle(j2000 + 0.25 * sidereal_day), kPi / 2.0, 1e-9);
    const simcore::Epoch late = simcore::Epoch::from_tt_seconds_since_j2000(830'000'000);
    const double before = simapp::earth_display_angle(late);
    const double after = simapp::earth_display_angle(late + sidereal_day);
    EXPECT_NEAR(std::remainder(after - before, 2.0 * kPi), 0.0, 1e-9);
    EXPECT_GE(before, 0.0);
    EXPECT_LT(before, 2.0 * kPi);
}

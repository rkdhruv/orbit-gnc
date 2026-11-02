// M3.3: telemetry, the derived quantities the HUD shows. Computed in simcore so they are tested;
// the front end only formats them.
//
//   altitude = |r| - R (spherical Earth), speed = |v|
//   elements from elements_from_state (M1.4); for a closed orbit (e < 1)
//     periapsis / apoapsis altitude = a (1 -/+ e) - R,   period T = 2 pi sqrt(a^3 / mu)
//   and for an open one (e >= 1) apoapsis and period are +infinity.
//   mass m = m_dry + m_p; delta-v remaining (rocket equation, main engine = thruster 0):
//     dv = v_e ln(m / m_dry),  v_e = Isp g0
//   thrust-axis-to-prograde angle: between body +x in ECI, R(q) x_b, and v_hat (where the main
//   engine is pointing relative to the direction of flight; 0 = prograde, 180 = retrograde)
//   wheel saturation: the wheel's speed relative to the body, h / J - a.w, as a signed fraction
//   of its limit (1 = saturated)
//
// Tolerances: elements_from_state is good to ~1e-15 relative (M1.4 tests); the derived values
// add a few flops: bounds 1e-9 relative unless stated.

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>

#include <Eigen/Geometry>

#include "../orbits/elements_fixtures.hpp"
#include "simcore/constants.hpp"
#include "simcore/sim/telemetry.hpp"

namespace {

using elements_fixtures::kDeg;
using elements_fixtures::state_from;
using simcore::MassProperties;
using simcore::PropellantTank;
using simcore::ReactionWheel;
using simcore::Telemetry;
using simcore::Thruster;
using simcore::VehicleConfig;
using simcore::VehicleState;
namespace earth = simcore::earth;

constexpr double kRel = 1e-9;

// 400 kg dry + 400 kg tank, main engine Isp 300 s along +x, one z wheel (J = 0.1, 600 rad/s).
VehicleConfig test_vehicle() {
    return VehicleConfig{
        MassProperties(400.0, Eigen::Vector3d(100.0, 150.0, 200.0).asDiagonal(),
                       PropellantTank{400.0, Eigen::Matrix3d::Zero()}),
        {Thruster(1000.0, 300.0, Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitX())},
        {ReactionWheel(Eigen::Vector3d::UnitZ(), 0.1, 0.2, 600.0)}};
}

VehicleState vehicle_at(const simcore::State& orbit, double propellant_kg) {
    return VehicleState{orbit.t,
                        orbit.r_eci,
                        orbit.v_eci,
                        Eigen::Quaterniond::Identity(),
                        Eigen::Vector3d::Zero(),
                        propellant_kg,
                        {0.0}};
}

}  // namespace

// Circular 420 km at 51.6 deg (CLAUDE.md reference): v = sqrt(mu / r) = 7657.269 m/s,
// T = 2 pi sqrt(r^3 / mu) = 5578.223 s (92.97 min), periapsis = apoapsis = 420 km.
TEST(Telemetry, CircularOrbit) {
    const double r = earth::radius + 420.0e3;
    const VehicleConfig cfg = test_vehicle();
    const Telemetry tm = simcore::telemetry(
        cfg, vehicle_at(state_from(r, 0.0, 51.6 * kDeg, 30.0 * kDeg, 0.0, 1.0), 400.0));
    EXPECT_NEAR(tm.altitude_m / 420.0e3, 1.0, kRel);
    EXPECT_NEAR(tm.speed_mps, 7657.269, 1e-3);
    EXPECT_NEAR(tm.elements.a_m / r, 1.0, kRel);
    EXPECT_NEAR(tm.elements.e, 0.0, 1e-12);
    EXPECT_NEAR(tm.elements.i_rad, 51.6 * kDeg, 1e-12);
    EXPECT_NEAR(tm.periapsis_altitude_m / 420.0e3, 1.0, kRel);
    EXPECT_NEAR(tm.apoapsis_altitude_m / 420.0e3, 1.0, kRel);
    EXPECT_NEAR(tm.period_s, 5578.223, 1e-3);
}

// Geostationary transfer orbit 300 x 35,786 km: r_p = 6678.137 km, r_a = 42164.137 km,
// a = (r_p + r_a) / 2 = 24421.137 km, e = (r_a - r_p) / (r_a + r_p) = 0.726543,
// T = 2 pi sqrt(a^3 / mu) = 37980.42 s (10.55 h). Started at nu = 2 rad, away from either apsis.
TEST(Telemetry, EllipticalTransferOrbit) {
    const double rp = earth::radius + 300.0e3;
    const double ra = earth::radius + 35786.0e3;
    const double a = 0.5 * (rp + ra);
    const double e = (ra - rp) / (ra + rp);
    EXPECT_NEAR(e, 0.726543, 1e-6);  // hand value
    const Telemetry tm = simcore::telemetry(
        test_vehicle(), vehicle_at(state_from(a, e, 27.0 * kDeg, 0.0, 180.0 * kDeg, 2.0), 400.0));
    // a (1 - e) - R subtracts two ~6,700 km numbers: 1e-9 of 6,700 km is 2.2e-8 of 300 km.
    EXPECT_NEAR(tm.periapsis_altitude_m / 300.0e3, 1.0, 1e-7);
    EXPECT_NEAR(tm.apoapsis_altitude_m / 35786.0e3, 1.0, kRel);
    EXPECT_NEAR(tm.period_s, 37980.42, 0.01);
    EXPECT_NEAR(tm.elements.e, e, 1e-12);
}

// Hyperbolic flyby (e = 1.5): no apoapsis and no period, both reported as +infinity; periapsis
// altitude is still a (1 - e) - R with a < 0.
TEST(Telemetry, OpenOrbitHasNoApoapsisOrPeriod) {
    const double rp = earth::radius + 500.0e3;
    const double e = 1.5;
    const double a = rp / (1.0 - e);
    const Telemetry tm =
        simcore::telemetry(test_vehicle(), vehicle_at(state_from(a, e, 0.3, 0.0, 0.0, 0.4), 400.0));
    EXPECT_TRUE(std::isinf(tm.apoapsis_altitude_m) && tm.apoapsis_altitude_m > 0.0);
    EXPECT_TRUE(std::isinf(tm.period_s) && tm.period_s > 0.0);
    EXPECT_NEAR(tm.periapsis_altitude_m / 500.0e3, 1.0, 1e-7);
}

// Rocket equation, Isp 300 s (v_e = 2941.995 m/s), 400 kg dry:
//   full (800 kg): v_e ln 2 = 2039.2355 m/s;  half (600 kg): v_e ln 1.5 = 1192.8763 m/s;
//   empty: 0 exactly. A vehicle with no thrusters has no delta-v.
TEST(Telemetry, DeltaVRemainingFollowsTheRocketEquation) {
    const VehicleConfig cfg = test_vehicle();
    const simcore::State orbit = state_from(earth::radius + 420.0e3, 0.0, 0.0, 0.0, 0.0, 0.0);
    const Telemetry full = simcore::telemetry(cfg, vehicle_at(orbit, 400.0));
    EXPECT_EQ(full.mass_kg, 800.0);
    EXPECT_NEAR(full.delta_v_remaining_mps, 2039.2355, 1e-4);
    EXPECT_NEAR(simcore::telemetry(cfg, vehicle_at(orbit, 200.0)).delta_v_remaining_mps, 1192.8763,
                1e-4);
    EXPECT_EQ(simcore::telemetry(cfg, vehicle_at(orbit, 0.0)).delta_v_remaining_mps, 0.0);
    VehicleConfig no_engine = cfg;
    no_engine.thrusters.clear();
    EXPECT_EQ(simcore::telemetry(no_engine, vehicle_at(orbit, 400.0)).delta_v_remaining_mps, 0.0);
}

// Equatorial orbit at nu = 0: r along ECI x, v along ECI y. Identity attitude puts body +x along
// ECI x: 90 deg from prograde. Yawed +90 deg about z, body +x is ECI y: prograde, 0 deg. Yawed
// -90 deg: retrograde, 180 deg. Bound 1e-12 rad (acos near 0 and pi loses precision, so the angle
// is taken with atan2).
TEST(Telemetry, ThrustAxisToProgradeAngle) {
    const simcore::State orbit = state_from(earth::radius + 420.0e3, 0.0, 0.0, 0.0, 0.0, 0.0);
    VehicleState s = vehicle_at(orbit, 400.0);
    const VehicleConfig cfg = test_vehicle();
    EXPECT_NEAR(simcore::telemetry(cfg, s).thrust_axis_to_prograde_rad, 0.5 * std::numbers::pi,
                1e-12);
    s.q_body_to_eci = Eigen::AngleAxisd(0.5 * std::numbers::pi, Eigen::Vector3d::UnitZ());
    EXPECT_NEAR(simcore::telemetry(cfg, s).thrust_axis_to_prograde_rad, 0.0, 1e-12);
    s.q_body_to_eci = Eigen::AngleAxisd(-0.5 * std::numbers::pi, Eigen::Vector3d::UnitZ());
    EXPECT_NEAR(simcore::telemetry(cfg, s).thrust_axis_to_prograde_rad, std::numbers::pi, 1e-12);
}

// Wheel speed relative to the body, h / J - a.w, over its 600 rad/s limit. Body at rest:
// h = 30 N m s -> 300 rad/s, 0.5; h = -60 -> -1 (saturated, spinning backwards). Body turning at
// w_z = 0.1 rad/s with h = 30: 300 - 0.1 = 299.9 rad/s, 0.49983.
TEST(Telemetry, WheelSaturationFraction) {
    const simcore::State orbit = state_from(earth::radius + 420.0e3, 0.0, 0.0, 0.0, 0.0, 0.0);
    VehicleState s = vehicle_at(orbit, 400.0);
    const VehicleConfig cfg = test_vehicle();
    s.wheel_h_nms = {30.0};
    const Telemetry half = simcore::telemetry(cfg, s);
    ASSERT_EQ(half.wheel_saturation.size(), 1u);
    EXPECT_NEAR(half.wheel_saturation[0], 0.5, 1e-15);
    s.wheel_h_nms = {-60.0};
    EXPECT_NEAR(simcore::telemetry(cfg, s).wheel_saturation[0], -1.0, 1e-15);
    s.wheel_h_nms = {30.0};
    s.w_body = Eigen::Vector3d(0.0, 0.0, 0.1);
    EXPECT_NEAR(simcore::telemetry(cfg, s).wheel_saturation[0], 299.9 / 600.0, 1e-15);
}

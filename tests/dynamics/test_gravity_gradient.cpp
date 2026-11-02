// M4.3: gravity-gradient torque.
//
// The near side of a body is pulled slightly harder than the far side; for a body small compared
// with its orbit the net torque about the centre of mass is (Hughes, "Spacecraft Attitude
// Dynamics", 8.1)
//   tau_b = (3 mu / r^3) r_hat_b x (I r_hat_b),   r_hat_b = R(q)^T r_hat_eci (position direction
// in body axes). It vanishes when r_hat_b is along a principal axis and is always perpendicular
// to r_hat_b. The inertia is a fixed constructor parameter (a stated simplification, like drag's
// fixed mass: the torque model doesn't see the propellant load).
// Hand values: 420 km circular, mu / r^3 = n^2 = 1.268727e-6 /s^2.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <vector>

#include <Eigen/Geometry>

#include "simcore/constants.hpp"
#include "simcore/dynamics/gravity_gradient.hpp"
#include "simcore/dynamics/point_mass_gravity.hpp"
#include "simcore/sim/simulation.hpp"

namespace {

using simcore::AttitudeState;
using simcore::GravityGradientTorque;
using simcore::State;
namespace earth = simcore::earth;

constexpr double kPi = std::numbers::pi;
const double kR = earth::radius + 420.0e3;
const simcore::Epoch kT0 = simcore::Epoch::from_tt_seconds_since_j2000(800'000'000);

State at_x() {
    return State{kT0, Eigen::Vector3d(kR, 0.0, 0.0), Eigen::Vector3d(0.0, 7.0e3, 0.0)};
}

AttitudeState att(const Eigen::Quaterniond& q) {
    return AttitudeState{kT0, q, Eigen::Vector3d::Zero()};
}

}  // namespace

// I = diag(100, 200, 300), body turned -45 deg about z so that r_hat_b = (c, s, 0), c = s = 1/sqrt
// 2:
//   I r_hat_b = (100 c, 200 s, 0),  r_hat_b x I r_hat_b = (0, 0, 200 c s - 100 s c) = (0, 0, 50)
//   tau = 3 n^2 * 50 = 1.90309e-4 N m about z. Bound 1e-12 relative.
TEST(GravityGradient, HandValueAtFortyFiveDegrees) {
    const GravityGradientTorque gg(Eigen::Vector3d(100.0, 200.0, 300.0).asDiagonal());
    const Eigen::Quaterniond q(Eigen::AngleAxisd(-kPi / 4.0, Eigen::Vector3d::UnitZ()));
    const Eigen::Vector3d tau = gg.torque_body(at_x(), att(q));
    const double n2 = earth::mu / (kR * kR * kR);
    EXPECT_NEAR(n2, 1.268727e-6, 1e-12);  // hand value
    EXPECT_NEAR(tau.z() / (3.0 * n2 * 50.0), 1.0, 1e-12);
    EXPECT_NEAR(tau.z(), 1.90309e-4, 1e-9);
    EXPECT_LE(tau.head<2>().norm(), 1e-15);
}

// No torque with r_hat along a principal axis (any of the three), or for a body with equal
// moments (I = k 1: r x k r = 0). For a general attitude and inertia the torque is perpendicular
// to r_hat_b. Bounds: roundoff on ~1e-4 N m torques, 1e-18 N m absolute.
TEST(GravityGradient, VanishesOnPrincipalAxesAndIsPerpendicularToR) {
    Eigen::Matrix3d I;
    I << 1000.0, 0.0, 500.0, 0.0, 2000.0, 0.0, 500.0, 0.0, 2500.0;
    const GravityGradientTorque diag(Eigen::Vector3d(100.0, 200.0, 300.0).asDiagonal());
    for (const Eigen::Quaterniond& q :
         {Eigen::Quaterniond::Identity(),
          Eigen::Quaterniond(Eigen::AngleAxisd(kPi / 2.0, Eigen::Vector3d::UnitZ())),
          Eigen::Quaterniond(Eigen::AngleAxisd(kPi / 2.0, Eigen::Vector3d::UnitY()))}) {
        EXPECT_LE(diag.torque_body(at_x(), att(q)).norm(), 1e-18);
    }
    const GravityGradientTorque sphere(400.0 * Eigen::Matrix3d::Identity());
    const Eigen::Quaterniond skew(Eigen::AngleAxisd(0.7, Eigen::Vector3d(1, 2, 3).normalized()));
    EXPECT_LE(sphere.torque_body(at_x(), att(skew)).norm(), 1e-18);
    const GravityGradientTorque full(I);
    const Eigen::Vector3d tau = full.torque_body(at_x(), att(skew));
    const Eigen::Vector3d r_b = skew.conjugate() * Eigen::Vector3d::UnitX();
    EXPECT_GT(tau.norm(), 1e-4);
    EXPECT_LE(std::abs(tau.dot(r_b)), 1e-15 * tau.norm());
}

// Pitch libration, the classic gravity-gradient check. Equatorial circular 420 km orbit; body x
// radial (I_x = 100, minimum), y along-track (I_y = 300), z the orbit normal (I_z = 350), turning
// with the orbit at n, pitched 2 deg off radial. Linearised (Hughes 9.2):
//   I_z theta'' = -3 n^2 (I_y - I_x) theta,  w_lib = n sqrt(3 (I_y - I_x) / I_z) = 1.30931 n,
// period 2 pi / w_lib = 4260.44 s (orbit period / 1.30931). Measured between successive upward
// zero crossings of theta (linear interpolation within a 0.5 s tick). Deviations: amplitude
// 2 deg adds (2 theta0)^2 / 16 = 3e-4 to the period (pendulum in 2 theta), RK4 at 0.5 s is
// negligible. Bound 0.2%.
TEST(GravityGradient, PitchLibrationPeriodMatchesTheLinearTheory) {
    const Eigen::Matrix3d I = Eigen::Vector3d(100.0, 300.0, 350.0).asDiagonal();
    const simcore::VehicleConfig cfg{
        simcore::MassProperties(500.0, I, simcore::PropellantTank{1.0, Eigen::Matrix3d::Zero()}),
        {},
        {}};
    const double vc = std::sqrt(earth::mu / kR);
    const double n = vc / kR;
    const double theta0 = 2.0 * kPi / 180.0;
    const simcore::VehicleState s0{
        kT0,
        Eigen::Vector3d(kR, 0.0, 0.0),
        Eigen::Vector3d(0.0, vc, 0.0),
        Eigen::Quaterniond(Eigen::AngleAxisd(theta0, Eigen::Vector3d::UnitZ())),
        Eigen::Vector3d(0.0, 0.0, n),
        0.0,
        {}};
    simcore::Simulation sim(
        simcore::SimConfig{cfg, std::make_shared<simcore::PointMassGravity>(),
                           std::make_shared<GravityGradientTorque>(I), 0.5, std::nullopt},
        s0);
    const auto pitch = [](const simcore::VehicleState& s) {
        const Eigen::Vector3d r_hat = s.r_eci.normalized();
        const Eigen::Vector3d x = s.q_body_to_eci * Eigen::Vector3d::UnitX();
        return std::atan2(r_hat.cross(x).z(), r_hat.dot(x));  // body x ahead of radial: > 0
    };
    std::vector<double> crossings;  // times of upward zero crossings
    double prev = pitch(s0);
    // Starting at +theta0, upward crossings come at 3/4 and 7/4 of a period (~7460 s).
    for (int k = 0; k < 16'000 && crossings.size() < 2; ++k) {  // up to 8000 s
        sim.step(1);
        const double now = pitch(sim.snapshot().state);
        if (prev < 0.0 && now >= 0.0) {
            const double t = sim.snapshot().state.t - s0.t;
            crossings.push_back(t - 0.5 * now / (now - prev));
        }
        prev = now;
    }
    ASSERT_EQ(crossings.size(), 2u);
    const double T_lib = 2.0 * kPi / (n * std::sqrt(3.0 * 200.0 / 350.0));
    EXPECT_NEAR(T_lib, 4260.44, 0.01);  // hand value
    std::printf("[libration] period %.2f s (linear theory %.2f s)\n", crossings[1] - crossings[0],
                T_lib);
    EXPECT_NEAR((crossings[1] - crossings[0]) / T_lib, 1.0, 0.002);
}

TEST(GravityGradient, RejectsInvalidInertiaAndState) {
    EXPECT_THROW((void)GravityGradientTorque(Eigen::Vector3d(-1.0, 1.0, 1.0).asDiagonal()),
                 std::invalid_argument);
    const GravityGradientTorque gg(Eigen::Vector3d(100.0, 200.0, 300.0).asDiagonal());
    State origin = at_x();
    origin.r_eci.setZero();
    EXPECT_THROW((void)gg.torque_body(origin, att(Eigen::Quaterniond::Identity())),
                 std::invalid_argument);
}

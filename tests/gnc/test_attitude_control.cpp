// M4.1: PD attitude control on the quaternion error, and orbit-relative hold modes.
//
// Error: q_e = q_t^-1 (x) q, the rotation from the target attitude to the current one, in body
// axes (both q are body-to-ECI). With q_e = [w, v], the shortest-path rotation vector is
//   d_theta = 2 atan2(|v|, |w|) sign(w) v / |v|
// (q and -q are the same attitude, so the sign of w is factored out: the error is at most pi).
// Rate error: d_w = w - R(q)^T w_t,eci, the body rate relative to the target frame's rate.
// Control law (body torque):
//   tau = -I (wn^2 d_theta + 2 zeta wn d_w) + w x H
// The w x H term (H = I w + sum h a, the total momentum) cancels the gyroscopic term in Euler's
// equations, so for rotation about a principal axis the closed loop is exactly the second-order
// system  theta'' + 2 zeta wn theta' + wn^2 theta = 0.
// Hold targets: body +x (the engine axis) along a direction fixed in the orbit frame, roll fixed by
// putting body +z as close as possible to radial-out (or to the orbit normal when x is radial):
//   x = d,  z = unit(ref - (ref . x) x),  y = z x x,  R = [x y z]  (body-to-ECI, det +1)
// The frame turns with the orbit at w_t = (r x v) / |r|^2 (exact for circular orbits; the target
// rate is fed forward so a PD hold has no steady-state lag).

#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <numbers>
#include <stdexcept>

#include <Eigen/Geometry>

#include "../orbits/elements_fixtures.hpp"
#include "simcore/constants.hpp"
#include "simcore/dynamics/force_sum.hpp"
#include "simcore/dynamics/point_mass_gravity.hpp"
#include "simcore/gnc/attitude_control.hpp"
#include "simcore/sim/simulation.hpp"

namespace {

using elements_fixtures::kDeg;
using elements_fixtures::state_from;
using simcore::AttitudeTarget;
using simcore::HoldMode;
using simcore::PdGains;
using simcore::Simulation;
using simcore::VehicleConfig;
using simcore::VehicleState;
namespace earth = simcore::earth;

constexpr double kPi = std::numbers::pi;

// I = diag(300, 400, 450) (wheels locked in), empty-tank inertia constant, three 2 N m wheels on
// the body axes (strong enough that the step responses below never saturate).
VehicleConfig test_vehicle() {
    return VehicleConfig{
        simcore::MassProperties(400.0, Eigen::Vector3d(300.0, 400.0, 450.0).asDiagonal(),
                                simcore::PropellantTank{400.0, Eigen::Matrix3d::Zero()}),
        {simcore::Thruster(1000.0, 300.0, Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitX())},
        {simcore::ReactionWheel(Eigen::Vector3d::UnitX(), 0.05, 2.0, 600.0),
         simcore::ReactionWheel(Eigen::Vector3d::UnitY(), 0.05, 2.0, 600.0),
         simcore::ReactionWheel(Eigen::Vector3d::UnitZ(), 0.05, 2.0, 600.0)}};
}

Eigen::Vector3d column(const Eigen::Quaterniond& q, int k) {
    return q.toRotationMatrix().col(k);
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Attitude error
// ---------------------------------------------------------------------------------------------

// 0.3 rad about a skew axis from the identity target: d_theta = 0.3 a exactly (to roundoff). -q
// gives the same error. 350 deg about z is the short way -10 deg about z. With a non-identity
// target, the error is in body axes: q = q_t (x) Ry(0.2) gives (0, 0.2, 0).
TEST(AttitudeControl, ErrorIsTheShortestRotationFromTargetInBodyAxes) {
    const Eigen::Vector3d a = Eigen::Vector3d(1.0, -2.0, 0.5).normalized();
    const Eigen::Quaterniond q(Eigen::AngleAxisd(0.3, a));
    const Eigen::Quaterniond id = Eigen::Quaterniond::Identity();
    EXPECT_LE((simcore::attitude_error_rad(q, id) - 0.3 * a).norm(), 1e-15);
    const Eigen::Quaterniond minus_q(-q.w(), -q.x(), -q.y(), -q.z());
    EXPECT_LE((simcore::attitude_error_rad(minus_q, id) - 0.3 * a).norm(), 1e-15);
    const Eigen::Quaterniond q350(Eigen::AngleAxisd(350.0 * kDeg, Eigen::Vector3d::UnitZ()));
    EXPECT_LE((simcore::attitude_error_rad(q350, id) - Eigen::Vector3d(0, 0, -10.0 * kDeg)).norm(),
              1e-14);
    const Eigen::Quaterniond qt(Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitX()));
    const Eigen::Quaterniond qb =
        qt * Eigen::Quaterniond(Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitY()));
    EXPECT_LE((simcore::attitude_error_rad(qb, qt) - Eigen::Vector3d(0, 0.2, 0)).norm(), 1e-15);
}

// ---------------------------------------------------------------------------------------------
// Hold targets
// ---------------------------------------------------------------------------------------------

// Equatorial circular orbit at nu = 0: r along ECI x, v along ECI y, h along ECI z, so the radial,
// along-track and normal directions are the ECI axes and every target frame can be written down:
//   prograde     x = +y,  z = +x (radial out),  y = z x x = +z
//   retrograde   x = -y,  z = +x,               y = -z
//   normal       x = +z,  z = +x,               y = -y
//   antinormal   x = -z,  z = +x,               y = +y
//   radial out   x = +x,  z = +z (normal),      y = +y
//   radial in    x = -x,  z = +z,               y = -y
// All rotate with the orbit at w_t = n z, n = v / r. Bounds 1e-15 (a few flops on unit vectors).
TEST(AttitudeControl, HoldTargetsOnAnEquatorialOrbit) {
    const double r = earth::radius + 420.0e3;
    const simcore::State o = state_from(r, 0.0, 0.0, 0.0, 0.0, 0.0);
    const Eigen::Vector3d X = Eigen::Vector3d::UnitX();
    const Eigen::Vector3d Y = Eigen::Vector3d::UnitY();
    const Eigen::Vector3d Z = Eigen::Vector3d::UnitZ();
    struct Case {
        HoldMode mode;
        Eigen::Vector3d x, y, z;
    };
    const Case cases[] = {{HoldMode::kPrograde, Y, Z, X},  {HoldMode::kRetrograde, -Y, -Z, X},
                          {HoldMode::kNormal, Z, -Y, X},   {HoldMode::kAntiNormal, -Z, Y, X},
                          {HoldMode::kRadialOut, X, Y, Z}, {HoldMode::kRadialIn, -X, -Y, Z}};
    const double n = o.v_eci.norm() / r;
    for (const Case& c : cases) {
        SCOPED_TRACE(static_cast<int>(c.mode));
        const AttitudeTarget t = simcore::hold_target(c.mode, o);
        EXPECT_LE((column(t.q_body_to_eci, 0) - c.x).norm(), 1e-15);
        EXPECT_LE((column(t.q_body_to_eci, 1) - c.y).norm(), 1e-15);
        EXPECT_LE((column(t.q_body_to_eci, 2) - c.z).norm(), 1e-15);
        EXPECT_LE((t.w_eci_radps - n * Z).norm(), 1e-15 * n);
    }
}

// On an inclined, eccentric orbit the frames are still proper rotations with body x along the
// wanted direction (v_hat for prograde, h_hat for normal, r_hat for radial out).
TEST(AttitudeControl, HoldTargetsOnAnInclinedEllipse) {
    const simcore::State o = state_from(1.2e7, 0.3, 63.4 * kDeg, 40.0 * kDeg, 270.0 * kDeg, 1.0);
    const Eigen::Vector3d h_hat = o.r_eci.cross(o.v_eci).normalized();
    EXPECT_LE((column(simcore::hold_target(HoldMode::kPrograde, o).q_body_to_eci, 0) -
               o.v_eci.normalized())
                  .norm(),
              1e-15);
    EXPECT_LE((column(simcore::hold_target(HoldMode::kNormal, o).q_body_to_eci, 0) - h_hat).norm(),
              1e-15);
    EXPECT_LE((column(simcore::hold_target(HoldMode::kRadialOut, o).q_body_to_eci, 0) -
               o.r_eci.normalized())
                  .norm(),
              1e-15);
    EXPECT_NEAR(simcore::hold_target(HoldMode::kRetrograde, o)
                    .q_body_to_eci.toRotationMatrix()
                    .determinant(),
                1.0, 1e-15);
}

// ---------------------------------------------------------------------------------------------
// Closed loop
// ---------------------------------------------------------------------------------------------

// Step response about a principal axis: 10 deg about body z (I_zz = 450) from rest, inertial
// target (identity, w_t = 0), wn = 0.1 rad/s, zeta = 0.7, command updated every 5 ticks (0.1 s).
// The closed loop is the second-order system, so (Ogata):
//   overshoot  M_p = exp(-zeta pi / sqrt(1 - zeta^2)) = 4.60%,
//   peak time  t_p = pi / (wn sqrt(1 - zeta^2)) = 43.99 s.
// Deviations: the command is held 0.1 s (a ~0.05 s delay: phase lag wn T / 2 = 0.005 rad, which
// lowers the effective damping by ~wn T / 4 = 0.0025 and raises M_p by ~0.1 points), and the
// controller uses I_zz = 450 while the body responds with I_s,zz = 449.95 (1e-4 relative).
// Bounds: M_p within 0.3 points, t_p within 0.5 s. Peak torque Kp theta0 = 450 * 0.01 * 0.1745
// = 0.79 N m, under the 2 N m limit (linear throughout). Envelope at 120 s:
// 10 deg * e^(-zeta wn t) / sqrt(1 - zeta^2) = 0.003 deg: bound 0.01 deg. Off-axis error stays 0.
TEST(AttitudeControl, PdStepResponseMatchesSecondOrderTheory) {
    const VehicleConfig cfg = test_vehicle();
    const double theta0 = 10.0 * kDeg;
    VehicleState s0{simcore::Epoch::from_tt_seconds_since_j2000(800'000'000),
                    Eigen::Vector3d::Zero(),
                    Eigen::Vector3d::Zero(),
                    Eigen::Quaterniond(Eigen::AngleAxisd(theta0, Eigen::Vector3d::UnitZ())),
                    Eigen::Vector3d::Zero(),
                    0.0,
                    {0.0, 0.0, 0.0}};
    Simulation sim(
        simcore::SimConfig{cfg, std::make_shared<simcore::ForceSum>(),
                           std::make_shared<simcore::ConstantBodyTorque>(Eigen::Vector3d::Zero()),
                           0.02, std::nullopt, 0.0},
        s0);
    const AttitudeTarget target{Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero()};
    const PdGains gains{0.1, 0.7};
    double min_theta = theta0;
    double t_min = 0.0;
    double max_off_axis = 0.0;
    for (int k = 0; k < 1200; ++k) {  // 1200 x 0.1 s = 120 s
        const VehicleState& s = sim.snapshot().state;
        sim.submit(simcore::attitude_command(cfg, s, target, gains));
        sim.step(5);
        const Eigen::Vector3d e =
            simcore::attitude_error_rad(sim.snapshot().state.q_body_to_eci, target.q_body_to_eci);
        if (e.z() < min_theta) {
            min_theta = e.z();
            t_min = sim.snapshot().state.t - s0.t;
        }
        max_off_axis = std::max(max_off_axis, e.head<2>().norm());
    }
    const double zeta = 0.7;
    const double Mp = std::exp(-zeta * kPi / std::sqrt(1.0 - zeta * zeta));
    const double tp = kPi / (0.1 * std::sqrt(1.0 - zeta * zeta));
    EXPECT_NEAR(Mp, 0.0460, 1e-4);  // hand values
    EXPECT_NEAR(tp, 43.99, 0.01);
    EXPECT_NEAR(-min_theta / theta0, Mp, 0.003);
    EXPECT_NEAR(t_min, tp, 0.5);
    EXPECT_LE(simcore::attitude_error_rad(sim.snapshot().state.q_body_to_eci, target.q_body_to_eci)
                  .norm(),
              0.01 * kDeg);
    EXPECT_LE(max_off_axis, 1e-12);
}

// Prograde hold on a 420 km circular orbit: starting on target (attitude and rate), the PD hold
// with the target rate fed forward keeps body x on the velocity. The target turns at n = 1.1e-3
// rad/s, i.e. 1.1e-4 rad per 0.1 s command hold; between updates the body coasts at the matched
// rate, so the tracking error stays at roundoff and hold level: bound 0.01 deg over 300 s.
TEST(AttitudeControl, ProgradeHoldTracksTheVelocity) {
    const VehicleConfig cfg = test_vehicle();
    const simcore::State o = state_from(earth::radius + 420.0e3, 0.0, 51.6 * kDeg, 0.0, 0.0, 0.0);
    const AttitudeTarget t0 = simcore::hold_target(HoldMode::kPrograde, o);
    VehicleState s0{o.t,
                    o.r_eci,
                    o.v_eci,
                    t0.q_body_to_eci,
                    t0.q_body_to_eci.conjugate() * t0.w_eci_radps,
                    0.0,
                    {0.0, 0.0, 0.0}};
    Simulation sim(
        simcore::SimConfig{cfg, std::make_shared<simcore::PointMassGravity>(),
                           std::make_shared<simcore::ConstantBodyTorque>(Eigen::Vector3d::Zero()),
                           0.02, std::nullopt},
        s0);
    const PdGains gains{0.1, 0.7};
    double max_err = 0.0;
    for (int k = 0; k < 3000; ++k) {  // 300 s
        const VehicleState& s = sim.snapshot().state;
        const AttitudeTarget t =
            simcore::hold_target(HoldMode::kPrograde, simcore::State{s.t, s.r_eci, s.v_eci});
        sim.submit(simcore::attitude_command(cfg, s, t, gains));
        sim.step(5);
        const VehicleState& n = sim.snapshot().state;
        const Eigen::Vector3d x = n.q_body_to_eci * Eigen::Vector3d::UnitX();
        max_err = std::max(max_err, std::acos(std::min(1.0, x.dot(n.v_eci.normalized()))));
    }
    EXPECT_LE(max_err, 0.01 * kDeg);
}

TEST(AttitudeControl, RejectsInvalidGains) {
    const VehicleConfig cfg = test_vehicle();
    const VehicleState s{simcore::Epoch::j2000(),
                         Eigen::Vector3d(7e6, 0, 0),
                         Eigen::Vector3d(0, 7.5e3, 0),
                         Eigen::Quaterniond::Identity(),
                         Eigen::Vector3d::Zero(),
                         0.0,
                         {0.0, 0.0, 0.0}};
    const AttitudeTarget t{Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero()};
    const double kNaN = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((void)simcore::attitude_command(cfg, s, t, PdGains{0.0, 0.7}),
                 std::invalid_argument);
    EXPECT_THROW((void)simcore::attitude_command(cfg, s, t, PdGains{0.1, -0.1}),
                 std::invalid_argument);
    EXPECT_THROW((void)simcore::attitude_command(cfg, s, t, PdGains{kNaN, 0.7}),
                 std::invalid_argument);
    simcore::State radial{s.t, s.r_eci, Eigen::Vector3d(1000.0, 0, 0)};
    EXPECT_THROW((void)simcore::hold_target(HoldMode::kPrograde, radial), std::invalid_argument);
}

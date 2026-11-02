// M4.3: reaction-wheel momentum management (desaturation with RCS).
//
// An external torque that the attitude loop resists (here gravity gradient) ends up as momentum
// in the wheels, h_w = sum h_j a_j, until they saturate. Dumping: the RCS applies
//   tau_rcs = -k h_w   (scaled down together to the RCS limits)
// while the wheels are commanded tau_att - tau_rcs. The body then feels exactly the attitude
// torque: tau_att - tau_rcs from the wheels plus tau_rcs from the RCS (plus the environment, as
// without dumping), so the attitude holds. The wheels take the opposite, h_w' = tau_rcs - tau_att;
// in a steady hold tau_att ~ -tau_env, so h_w' = -k h_w + tau_env: an exponential spin-down with
// time constant 1/k towards h* = tau_env / k.
// Hysteresis: dumping starts when any wheel's saturation |h/J - a.w| / Omega_max reaches 0.8 and
// stops when all are below 0.1.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <numbers>
#include <optional>
#include <vector>

#include <Eigen/Geometry>

#include "../orbits/elements_fixtures.hpp"
#include "simcore/constants.hpp"
#include "simcore/dynamics/gravity_gradient.hpp"
#include "simcore/dynamics/point_mass_gravity.hpp"
#include "simcore/gnc/attitude_control.hpp"
#include "simcore/gnc/momentum.hpp"
#include "simcore/sim/simulation.hpp"
#include "simcore/vehicle/rcs.hpp"

namespace {

using elements_fixtures::state_from;
using simcore::DesatSettings;
using simcore::RcsAxis;
using simcore::VehicleConfig;
using simcore::VehicleState;
namespace earth = simcore::earth;

constexpr double kPi = std::numbers::pi;

// 1000 kg dry with a product of inertia (I_xz = 500; principal moments 848.5, 2000, 2651.5),
// 20 kg of propellant in a zero-inertia tank (I constant), small wheels on the body axes
// (J = 0.005, 0.1 N m, 600 rad/s: J Omega_max = 3 N m s) and RCS couples on the body axes
// (0.5 N m, L = 1 m, Isp 220 s).
Eigen::Matrix3d inertia() {
    Eigen::Matrix3d I;
    I << 1000.0, 0.0, 500.0, 0.0, 2000.0, 0.0, 500.0, 0.0, 2500.0;
    return I;
}

VehicleConfig test_vehicle() {
    VehicleConfig cfg{
        simcore::MassProperties(1000.0, inertia(),
                                simcore::PropellantTank{50.0, Eigen::Matrix3d::Zero()}),
        {},
        {simcore::ReactionWheel(Eigen::Vector3d::UnitX(), 0.005, 0.1, 600.0),
         simcore::ReactionWheel(Eigen::Vector3d::UnitY(), 0.005, 0.1, 600.0),
         simcore::ReactionWheel(Eigen::Vector3d::UnitZ(), 0.005, 0.1, 600.0)}};
    cfg.rcs = {RcsAxis(Eigen::Vector3d::UnitX(), 0.5, 1.0, 220.0),
               RcsAxis(Eigen::Vector3d::UnitY(), 0.5, 1.0, 220.0),
               RcsAxis(Eigen::Vector3d::UnitZ(), 0.5, 1.0, 220.0)};
    return cfg;
}

VehicleState resting_with_wheels(const std::vector<double>& h) {
    return VehicleState{simcore::Epoch::from_tt_seconds_since_j2000(800'000'000),
                        Eigen::Vector3d(7.0e6, 0.0, 0.0),
                        Eigen::Vector3d(0.0, 7.5e3, 0.0),
                        Eigen::Quaterniond::Identity(),
                        Eigen::Vector3d::Zero(),
                        20.0,
                        h};
}

}  // namespace

// h_w = sum h_j a_j: with wheels on the body axes, just the h vector. Saturation of the busiest
// wheel: h = 2.4 N m s on y at rest is 2.4 / 0.005 = 480 rad/s, 0.8 of 600.
TEST(Momentum, WheelMomentumAndSaturation) {
    const VehicleConfig cfg = test_vehicle();
    const VehicleState s = resting_with_wheels({1.0, -2.4, 0.5});
    EXPECT_LE((simcore::wheel_momentum_body(cfg, s) - Eigen::Vector3d(1.0, -2.4, 0.5)).norm(),
              1e-15);
    EXPECT_NEAR(simcore::max_wheel_saturation(cfg, s), 0.8, 1e-15);
}

// Hysteresis: off below 0.8 (0.79 stays off), on at 0.8; once on it stays on down to 0.1 and
// switches off below it.
TEST(Momentum, DesatHysteresis) {
    const VehicleConfig cfg = test_vehicle();
    const DesatSettings d{};
    const auto sat = [](double f) { return resting_with_wheels({0.0, f * 3.0, 0.0}); };
    EXPECT_FALSE(simcore::desat_active(false, cfg, sat(0.79), d));
    EXPECT_TRUE(simcore::desat_active(false, cfg, sat(0.80), d));
    EXPECT_TRUE(simcore::desat_active(true, cfg, sat(0.50), d));
    EXPECT_TRUE(simcore::desat_active(true, cfg, sat(0.10), d));
    EXPECT_FALSE(simcore::desat_active(true, cfg, sat(0.09), d));
}

// Dump torque tau = -k h_w, k = 0.02 /s, as RCS commands u = tau / tau_max (axes on the body
// axes): h_w = (0, 2.4, 0) -> tau = (0, -0.048, 0), u = (0, -0.096, 0) (the wheels can counter
// 0.048 N m: under 80% of their 0.1 N m). A large h_w is scaled down together, to the tighter of
// the RCS limit and 80% of the wheels' authority (they must counter it to hold attitude):
// (30, 0, 40) wants (-0.6, 0, -0.8) N m; the RCS would allow 1/1.6 of it, the wheels only
// 0.8 * 0.1 / 0.8 = 0.1 of it, so tau = (-0.06, 0, -0.08) and u = (-0.12, 0, -0.16). Bounds 1e-15.
TEST(Momentum, DumpCommandOpposesTheWheelMomentum) {
    const VehicleConfig cfg = test_vehicle();
    const DesatSettings d{};
    EXPECT_EQ(d.gain_per_s, 0.02);
    const std::vector<double> u =
        simcore::rcs_dump_command(cfg, resting_with_wheels({0.0, 2.4, 0.0}), d);
    EXPECT_NEAR(u[0], 0.0, 1e-15);
    EXPECT_NEAR(u[1], -0.096, 1e-15);
    EXPECT_NEAR(u[2], 0.0, 1e-15);
    const std::vector<double> big =
        simcore::rcs_dump_command(cfg, resting_with_wheels({30.0, 0.0, 40.0}), d);
    EXPECT_NEAR(big[0], -0.12, 1e-15);
    EXPECT_NEAR(big[2], -0.16, 1e-15);
}

// While dumping, the wheels make tau_att - tau_rcs, so wheels plus RCS together give the body
// exactly tau_att: -sum tau_m a + sum u tau_max a_rcs = attitude_torque_body (1e-15).
TEST(Momentum, DumpingLeavesTheAttitudeTorqueUnchanged) {
    const VehicleConfig cfg = test_vehicle();
    VehicleState s = resting_with_wheels({0.5, 2.4, -0.3});
    s.q_body_to_eci = Eigen::AngleAxisd(0.01, Eigen::Vector3d(1, 1, 0).normalized());
    const simcore::AttitudeTarget target{Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero()};
    const simcore::AttitudeGains g = simcore::pd_gains(inertia(), simcore::PdGains{0.05, 0.7});
    const simcore::VehicleCommand c =
        simcore::attitude_command_with_desat(cfg, s, target, g, true, DesatSettings{});
    Eigen::Vector3d net = Eigen::Vector3d::Zero();
    for (std::size_t j = 0; j < cfg.wheels.size(); ++j) {
        net -= c.wheel_torque_nm[j] * cfg.wheels[j].axis_body();
    }
    for (std::size_t j = 0; j < cfg.rcs.size(); ++j) {
        net += cfg.rcs[j].torque_body_nm(c.rcs[j]);
    }
    EXPECT_LE((net - simcore::attitude_torque_body(cfg, s, target, g)).norm(), 1e-15);
    EXPECT_NE(c.rcs[1], 0.0);
}

// Closed loop: prograde hold (PD, wn = 0.05, zeta = 0.7) on a 420 km equatorial circular orbit
// with gravity gradient, 4000 s at 0.5 s ticks. In the prograde frame r_hat_b = body z, so the
// gravity-gradient torque is constant in body axes:
//   tau = 3 n^2 z x (I z) = 3 n^2 (-I_yz, I_xz, 0) = 3 * 1.268727e-6 * (0, 500, 0)
//       = (0, 1.90309e-3, 0) N m,
// and the body turns at n about y, so w x H has no y part: the y wheel fills at exactly that rate.
// Predictions:
//   - first dump at saturation 0.8: (h/J - n) / 600 = 0.8 -> t1 = (480 + n) J / tau = 1261.11 s;
//   - a dump runs ~ln((2.4 - h*) / (0.3 - h*)) / k = 121.0 s (h* = tau / k = 0.095 N m s), then
//     recharging 0.3 -> 2.4 N m s takes 2.1 / tau = 1103.5 s: dumps start at ~1261, 2486, 3710 s,
//     so 3 within 4000 s;
//   - saturation never exceeds 0.8 by more than one tick's growth (tau dt / (J Omega) ~ 3e-4):
//     bound 0.81;
//   - the attitude error is the PD's steady offset tau / Kp = 1.9e-3 / (2000 * 0.0025) = 0.022 deg
//     (dumping doesn't disturb it): bound 0.05 deg;
//   - propellant = integral |tau_rcs| dt / (L Isp g0), exactly the commanded impulse; each dump
//     removes ~2.1 + tau * 121 = 2.33 N m s, so ~7.0 N m s and ~3.24e-3 kg in all (within 10%).
// Without dumping the y wheel saturates (at ~1577 s) and the uncountered torque swings the body
// off: error > 1 deg by 4000 s.
TEST(Momentum, DesatKeepsAGravityGradientHoldOutOfSaturation) {
    const VehicleConfig cfg = test_vehicle();
    const double r = earth::radius + 420.0e3;
    const simcore::State o = state_from(r, 0.0, 0.0, 0.0, 0.0, 0.0);
    const simcore::AttitudeTarget t0 = simcore::hold_target(simcore::HoldMode::kPrograde, o);
    const VehicleState s0{o.t,
                          o.r_eci,
                          o.v_eci,
                          t0.q_body_to_eci,
                          t0.q_body_to_eci.conjugate() * t0.w_eci_radps,
                          20.0,
                          {0.0, 0.0, 0.0}};
    const double n = std::sqrt(earth::mu / (r * r * r));
    const double tau = 3.0 * n * n * 500.0;
    EXPECT_NEAR(tau, 1.90309e-3, 1e-8);  // hand value
    const double t1_pred = (480.0 + n) * 0.005 / tau;
    EXPECT_NEAR(t1_pred, 1261.11, 0.01);

    const simcore::AttitudeGains g = simcore::pd_gains(inertia(), simcore::PdGains{0.05, 0.7});
    struct Run {
        int dumps = 0;
        double t_first = -1.0;
        double max_sat = 0.0;
        double max_err_deg = 0.0;
        double rcs_impulse_nms = 0.0;  // integral of |tau_rcs| dt
        double propellant_used = 0.0;
    };
    const auto fly = [&](bool manage) {
        simcore::Simulation sim(
            simcore::SimConfig{cfg, std::make_shared<simcore::PointMassGravity>(),
                               std::make_shared<simcore::GravityGradientTorque>(inertia()), 0.5,
                               std::nullopt},
            s0);
        Run run;
        bool dumping = false;
        for (int k = 0; k < 8000; ++k) {
            const VehicleState& s = sim.snapshot().state;
            const bool now = manage && simcore::desat_active(dumping, cfg, s, DesatSettings{});
            if (now && !dumping) {
                ++run.dumps;
                if (run.t_first < 0.0) {
                    run.t_first = s.t - s0.t;
                }
            }
            dumping = now;
            const simcore::AttitudeTarget t = simcore::hold_target(
                simcore::HoldMode::kPrograde, simcore::State{s.t, s.r_eci, s.v_eci});
            const simcore::VehicleCommand c =
                simcore::attitude_command_with_desat(cfg, s, t, g, dumping, DesatSettings{});
            for (std::size_t j = 0; j < cfg.rcs.size(); ++j) {
                run.rcs_impulse_nms += cfg.rcs[j].torque_body_nm(c.rcs[j]).norm() * 0.5;
            }
            sim.submit(c);
            sim.step(1);
            const VehicleState& after = sim.snapshot().state;
            run.max_sat = std::max(run.max_sat, simcore::max_wheel_saturation(cfg, after));
            const Eigen::Vector3d x = after.q_body_to_eci * Eigen::Vector3d::UnitX();
            run.max_err_deg =
                std::max(run.max_err_deg,
                         std::acos(std::min(1.0, x.dot(after.v_eci.normalized()))) * 180.0 / kPi);
        }
        run.propellant_used = s0.propellant_kg - sim.snapshot().state.propellant_kg;
        return run;
    };
    const Run managed = fly(true);
    const Run unmanaged = fly(false);
    std::printf(
        "[desat] dumps %d, first at %.1f s, max saturation %.4f, max error %.4f deg, "
        "RCS impulse %.3f N m s, propellant %.4e kg\n",
        managed.dumps, managed.t_first, managed.max_sat, managed.max_err_deg,
        managed.rcs_impulse_nms, managed.propellant_used);
    std::printf("[none ] max saturation %.4f, max error %.3f deg\n", unmanaged.max_sat,
                unmanaged.max_err_deg);
    EXPECT_EQ(managed.dumps, 3);
    EXPECT_NEAR(managed.t_first / t1_pred, 1.0, 0.01);
    EXPECT_LE(managed.max_sat, 0.81);
    EXPECT_LE(managed.max_err_deg, 0.05);
    EXPECT_NEAR(managed.propellant_used / (managed.rcs_impulse_nms / (1.0 * 220.0 * simcore::g0)),
                1.0, 1e-9);
    EXPECT_NEAR(managed.propellant_used / 3.24e-3, 1.0, 0.10);
    EXPECT_GE(unmanaged.max_sat, 0.999);
    EXPECT_GT(unmanaged.max_err_deg, 1.0);
}

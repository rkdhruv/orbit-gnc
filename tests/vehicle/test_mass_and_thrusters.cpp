// M2.3: vehicle mass properties and thrusters.
//
// Mass properties (stated assumption, see ROADMAP M2): the propellant tank is centred on the
// vehicle's centre of mass, so burning fuel never moves the CoM, and the propellant's inertia about
// the CoM scales with the propellant mass (a uniformly emptying tank):
//   m(m_p) = m_dry + m_p,   I(m_p) = I_dry + (m_p / m_p,full) I_tank,full
// Thruster: force F = throttle * T_max * d_hat (body), torque about the CoM tau = r x F, propellant
// consumption m_dot = throttle * T_max / (Isp g0). Exhaust speed v_e = Isp g0.
// Rocket equation: dv = v_e ln(m0 / mf). Isp 300 s, m0/mf = 2 -> 2941.995 * ln 2 = 2039.24 m/s
// (CLAUDE.md reference ~2,039 m/s).

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>

#include "simcore/constants.hpp"
#include "simcore/vehicle/mass_properties.hpp"
#include "simcore/vehicle/thruster.hpp"

namespace {

using simcore::MassProperties;
using simcore::PropellantTank;
using simcore::Thruster;

constexpr double kRel = 1e-14;
const double kNaN = std::numeric_limits<double>::quiet_NaN();

MassProperties example_vehicle() {
    // 500 kg dry, I_dry = diag(100, 150, 200); tank holds 300 kg with full-tank inertia diag(30,
    // 30, 20).
    return MassProperties(500.0, Eigen::Vector3d(100.0, 150.0, 200.0).asDiagonal(),
                          PropellantTank{300.0, Eigen::Vector3d(30.0, 30.0, 20.0).asDiagonal()});
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Mass properties
// ---------------------------------------------------------------------------------------------

TEST(MassProperties, MassAndInertiaScaleWithPropellant) {
    const MassProperties mp = example_vehicle();
    EXPECT_EQ(mp.capacity_kg(), 300.0);
    EXPECT_EQ(mp.mass_kg(300.0), 800.0);
    EXPECT_EQ(mp.mass_kg(0.0), 500.0);
    // Full: I_dry + I_tank = diag(130, 180, 220); half: diag(115, 165, 210); empty: I_dry.
    const Eigen::Matrix3d full = Eigen::Vector3d(130.0, 180.0, 220.0).asDiagonal();
    const Eigen::Matrix3d half = Eigen::Vector3d(115.0, 165.0, 210.0).asDiagonal();
    const Eigen::Matrix3d empty = Eigen::Vector3d(100.0, 150.0, 200.0).asDiagonal();
    EXPECT_LE((mp.inertia_kgm2(300.0) - full).norm(), kRel * 220.0);
    EXPECT_LE((mp.inertia_kgm2(150.0) - half).norm(), kRel * 220.0);
    EXPECT_LE((mp.inertia_kgm2(0.0) - empty).norm(), kRel * 220.0);
    // And it is a valid rigid body at every fill level.
    EXPECT_LE((mp.rigid_body(150.0).inertia_body() - half).norm(), kRel * 220.0);
}

TEST(MassProperties, RejectsInvalidInput) {
    const MassProperties mp = example_vehicle();
    EXPECT_THROW((void)mp.mass_kg(-1.0), std::invalid_argument);
    EXPECT_THROW((void)mp.mass_kg(300.1), std::invalid_argument);
    EXPECT_THROW((void)mp.inertia_kgm2(kNaN), std::invalid_argument);
    const Eigen::Matrix3d I = Eigen::Vector3d(100.0, 150.0, 200.0).asDiagonal();
    const PropellantTank tank{300.0, Eigen::Vector3d(30.0, 30.0, 20.0).asDiagonal()};
    EXPECT_THROW((void)MassProperties(0.0, I, tank), std::invalid_argument);
    EXPECT_THROW((void)MassProperties(500.0, I, PropellantTank{-1.0, tank.inertia_full_kgm2}),
                 std::invalid_argument);
    // Tank inertia must itself be physically realisable. For any mass distribution
    // I = integral (r^2 1 - r r^T) dm, so S = 1/2 tr(I) 1 - I = integral r r^T dm is positive
    // semi-definite; equivalently each principal moment is >= 0 and <= the sum of the other two.
    // diag(-1, 1, 1): negative moment. diag(0, 0, 1000): PSD, but 1000 > 0 + 0 (S has -500).
    EXPECT_THROW((void)MassProperties(
                     500.0, I, PropellantTank{300.0, Eigen::Vector3d(-1.0, 1.0, 1.0).asDiagonal()}),
                 std::invalid_argument);
    EXPECT_THROW(
        (void)MassProperties(500.0, I,
                             PropellantTank{300.0, Eigen::Vector3d(0.0, 0.0, 1000.0).asDiagonal()}),
        std::invalid_argument);
    // A thin rod along z, diag(10, 10, 0), is the limiting physical case (S = diag(0, 0, 10)).
    EXPECT_NO_THROW((void)MassProperties(
        500.0, I, PropellantTank{300.0, Eigen::Vector3d(10.0, 10.0, 0.0).asDiagonal()}));
    // An empty-capable tank with zero inertia is fine (a point mass at the CoM).
    EXPECT_NO_THROW((void)MassProperties(500.0, I, PropellantTank{300.0, Eigen::Matrix3d::Zero()}));
}

// Inputs within the 1e-12 symmetry tolerance are stored symmetrised, so I(m_p) is exactly symmetric
// (Cholesky-style solvers read one triangle while H = I w uses both).
TEST(MassProperties, StoresSymmetrisedTensors) {
    Eigen::Matrix3d dry = Eigen::Vector3d(100.0, 150.0, 200.0).asDiagonal();
    dry(0, 1) = 1.0;
    dry(1, 0) = 1.0 + 1e-13;
    Eigen::Matrix3d tank = Eigen::Vector3d(30.0, 30.0, 20.0).asDiagonal();
    tank(1, 2) = 0.5;
    tank(2, 1) = 0.5 - 1e-14;
    const MassProperties mp(500.0, dry, PropellantTank{300.0, tank});
    const Eigen::Matrix3d I = mp.inertia_kgm2(120.0);
    EXPECT_EQ(I, I.transpose());
}

// ---------------------------------------------------------------------------------------------
// Thrusters
// ---------------------------------------------------------------------------------------------

// 100 N, Isp 300 s, at r = (0.1, 0, -1) m pointing +z (direction given unnormalised as (0, 0, 2)).
//   F = (0, 0, 100) N at full throttle
//   tau = r x F = (0 * 100 - (-1) * 0, (-1) * 0 - 0.1 * 100, 0) = (0, -10, 0) N m
//   m_dot = 100 / (300 * 9.80665) = 0.0339905 kg/s, v_e = 2941.995 m/s
TEST(Thruster, ForceTorqueAndMassFlow) {
    const Thruster t(100.0, 300.0, Eigen::Vector3d(0.1, 0.0, -1.0), Eigen::Vector3d(0.0, 0.0, 2.0));
    EXPECT_EQ(t.force_body_n(1.0), Eigen::Vector3d(0.0, 0.0, 100.0));
    EXPECT_LE((t.torque_body_nm(1.0) - Eigen::Vector3d(0.0, -10.0, 0.0)).norm(), kRel * 10.0);
    EXPECT_NEAR(t.mass_flow_kgps(1.0), 0.0339905, 1e-7);
    EXPECT_NEAR(t.mass_flow_kgps(1.0), 100.0 / (300.0 * simcore::g0), kRel);
    EXPECT_NEAR(t.exhaust_velocity_mps(), 2941.995, 1e-9);
    // Throttle scales force, torque and flow linearly; zero throttle is exactly zero.
    EXPECT_EQ(t.force_body_n(0.25), Eigen::Vector3d(0.0, 0.0, 25.0));
    EXPECT_NEAR(t.mass_flow_kgps(0.25), 0.25 * t.mass_flow_kgps(1.0), kRel);
    EXPECT_EQ(t.force_body_n(0.0), Eigen::Vector3d::Zero());
    EXPECT_EQ(t.mass_flow_kgps(0.0), 0.0);
}

// A thruster through the CoM along its own line of action produces no torque.
TEST(Thruster, ThrustThroughCentreOfMassHasNoTorque) {
    const Thruster t(50.0, 220.0, Eigen::Vector3d(0.0, 0.0, -1.5), Eigen::Vector3d(0.0, 0.0, 1.0));
    EXPECT_EQ(t.torque_body_nm(1.0), Eigen::Vector3d::Zero());
}

TEST(Thruster, RejectsInvalidInput) {
    const Eigen::Vector3d r = Eigen::Vector3d::Zero(), d = Eigen::Vector3d::UnitZ();
    EXPECT_THROW((void)Thruster(0.0, 300.0, r, d), std::invalid_argument);
    EXPECT_THROW((void)Thruster(100.0, 0.0, r, d), std::invalid_argument);
    EXPECT_THROW((void)Thruster(100.0, 300.0, r, Eigen::Vector3d::Zero()), std::invalid_argument);
    EXPECT_THROW((void)Thruster(100.0, 300.0, Eigen::Vector3d(kNaN, 0.0, 0.0), d),
                 std::invalid_argument);
    const Thruster t(100.0, 300.0, r, d);
    EXPECT_THROW((void)t.force_body_n(-0.1), std::invalid_argument);
    EXPECT_THROW((void)t.force_body_n(1.1), std::invalid_argument);
    EXPECT_THROW((void)t.mass_flow_kgps(kNaN), std::invalid_argument);
}

// Tsiolkovsky: dv = v_e ln(m0 / mf). Isp 300 s, mass ratio 2 -> 2039.24 m/s (CLAUDE.md ~2,039).
TEST(RocketEquation, IdealDeltaV) {
    const double ve = 300.0 * simcore::g0;
    EXPECT_NEAR(simcore::rocket_delta_v_mps(ve, 800.0, 400.0), 2039.24, 0.01);
    EXPECT_NEAR(simcore::rocket_delta_v_mps(ve, 800.0, 400.0), ve * std::log(2.0), kRel * 2039.0);
    EXPECT_EQ(simcore::rocket_delta_v_mps(ve, 800.0, 800.0), 0.0);
    EXPECT_THROW((void)simcore::rocket_delta_v_mps(ve, 400.0, 800.0), std::invalid_argument);
    EXPECT_THROW((void)simcore::rocket_delta_v_mps(ve, 800.0, 0.0), std::invalid_argument);
}

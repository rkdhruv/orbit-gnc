// M4.6: the autopilot flies transfers on its own, and the M4 done-when check, LEO -> GEO.
//
// Orbit-targeted cutoff: near periapsis the apoapsis is very sensitive to dv,
// dr_a/dv ~ 4 a^2 v / mu (60.8 km per m/s on a GTO), so a burn cut off on its impulsive dv lands
// short by the finite-burn loss (~15 m/s would be ~900 km). A burn with an apoapsis (or
// periapsis) cutoff fires until the osculating apsis reaches the target; the remaining dv is
// estimated from the apsis's sensitivity to dv along the burn direction (a finite difference),
// which drives the same throttle-down as a dv cutoff.
// Autopilot: a queue of steps (nodes, and circularise steps computed only when their turn comes
// from the orbit actually reached), flown one after another by ManeuverExecutor. Between burns it
// coasts with a zero command until coast_margin before ignition, so the sim can go on rails
// (warp-to-node).
//
// GEO-mission vehicle (an upper stage): 1000 kg dry + 3000 kg propellant, 20 kN at Isp 320 s
// (v_e = 3138.128 m/s, dv capacity v_e ln 4 = 4350 m/s), 1 N m / 120 N m s wheels.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <memory>
#include <numbers>
#include <optional>
#include <utility>
#include <vector>

#include "../orbits/elements_fixtures.hpp"
#include "simcore/constants.hpp"
#include "simcore/dynamics/point_mass_gravity.hpp"
#include "simcore/gnc/autopilot.hpp"
#include "simcore/gnc/maneuver.hpp"
#include "simcore/gnc/transfers.hpp"
#include "simcore/orbits/elements.hpp"
#include "simcore/sim/manual_control.hpp"
#include "simcore/sim/simulation.hpp"

namespace {

using elements_fixtures::state_from;
using simcore::Autopilot;
using simcore::BurnCutoff;
using simcore::ManeuverNode;
using simcore::State;
using simcore::VehicleConfig;
using simcore::VehicleState;
namespace earth = simcore::earth;

constexpr double kPi = std::numbers::pi;
const double kR300 = earth::radius + 300.0e3;
const double kR1000 = earth::radius + 1000.0e3;
const double kRGeo = earth::radius + 35786.0e3;
const double kVe = 320.0 * simcore::g0;

VehicleConfig upper_stage() {
    return VehicleConfig{
        simcore::MassProperties(
            1000.0, Eigen::Vector3d(1500.0, 3000.0, 3000.0).asDiagonal(),
            simcore::PropellantTank{3000.0, Eigen::Vector3d(900.0, 1800.0, 1800.0).asDiagonal()}),
        {simcore::Thruster(20000.0, 320.0, Eigen::Vector3d(-2.0, 0.0, 0.0),
                           Eigen::Vector3d::UnitX())},
        {simcore::ReactionWheel(Eigen::Vector3d::UnitX(), 0.2, 1.0, 600.0),
         simcore::ReactionWheel(Eigen::Vector3d::UnitY(), 0.2, 1.0, 600.0),
         simcore::ReactionWheel(Eigen::Vector3d::UnitZ(), 0.2, 1.0, 600.0)}};
}

// 300 km circular equatorial (or a given orbit), at rest in attitude, full tank; two-body with
// rails, 0.1 s ticks.
struct Mission {
    Mission() = default;
    explicit Mission(const State& start, VehicleConfig vehicle = upper_stage())
        : cfg(std::move(vehicle)), orbit(start) {}
    VehicleConfig cfg = upper_stage();
    State orbit = state_from(kR300, 0.0, 0.0, 0.0, 0.0, 0.0);
    simcore::Simulation sim{
        simcore::SimConfig{cfg, std::make_shared<simcore::PointMassGravity>(),
                           std::make_shared<simcore::ConstantBodyTorque>(Eigen::Vector3d::Zero()),
                           0.1, earth::mu},
        VehicleState{orbit.t, orbit.r_eci, orbit.v_eci, Eigen::Quaterniond::Identity(),
                     Eigen::Vector3d::Zero(), 3000.0, std::vector<double>(cfg.wheels.size(), 0.0)}};
    bool saw_rails = false;
    bool saw_coast_end = false;   // coast_until() was reported while coasting
    bool coast_end_ahead = true;  // ... and never more than one 60 s call in the past

    // Fly the autopilot until it disengages (or a time limit): commands re-evaluated at least every
    // 0.5 s (5 ticks), in 60 s calls (a coast is held for a whole call).
    void fly(Autopilot& ap, double limit_s) {
        const simcore::ControlLaw law = [this, &ap](const VehicleState& s) {
            return ap.command(cfg, s,
                              simcore::pd_gains(cfg.mass.inertia_kgm2(s.propellant_kg),
                                                simcore::PdGains{0.05, 0.7}));
        };
        while (ap.engaged() && sim.snapshot().state.t - orbit.t < limit_s) {
            simcore::fly_with(sim, law, 600, 5);
            saw_rails = saw_rails || sim.snapshot().on_rails;
            if (const std::optional<simcore::Epoch> until = ap.coast_until()) {
                saw_coast_end = true;
                coast_end_ahead = coast_end_ahead && *until > sim.snapshot().state.t - 60.0;
            }
        }
    }
    [[nodiscard]] simcore::KeplerianElements elements() const {
        const VehicleState& s = sim.snapshot().state;
        return simcore::elements_from_state(State{s.t, s.r_eci, s.v_eci});
    }
};

}  // namespace

// Apoapsis cutoff: 300 km -> apoapsis 1000 km. The impulsive dv is the Hohmann dv1 = 190.034 m/s,
// but the executor fires until the osculating apoapsis reaches the target. Checks: apoapsis within
// 0.1 km of R + 1000 km (the 1 mm/s cutoff residual is ~4 m here: dr_a/dv ~ 4 a^2 v / mu
// ~ 3.9 km per m/s), and the achieved dv at least the impulsive one (finite-burn loss >= 0) but
// within 0.5% of it (a 37 s burn sweeps 2.5 deg).
TEST(Autopilot, ExecutorCutsOffOnTheTargetApoapsis) {
    Mission m;
    const ManeuverNode node{m.orbit.t + 600.0, Eigen::Vector3d(0.0, 190.034, 0.0)};
    simcore::ManeuverExecutor exec(m.cfg, m.sim.snapshot().state, node, 0.5,
                                   BurnCutoff{BurnCutoff::Kind::kApoapsis, kR1000});
    for (int k = 0; k < 4000 && exec.phase() != simcore::ManeuverExecutor::Phase::kDone; ++k) {
        const VehicleState& s = m.sim.snapshot().state;
        m.sim.submit(exec.command(m.cfg, s,
                                  simcore::pd_gains(m.cfg.mass.inertia_kgm2(s.propellant_kg),
                                                    simcore::PdGains{0.05, 0.7})));
        m.sim.step(5);
    }
    ASSERT_EQ(exec.phase(), simcore::ManeuverExecutor::Phase::kDone);
    const simcore::KeplerianElements el = m.elements();
    const double dv =
        kVe * std::log(4000.0 / m.cfg.mass.mass_kg(m.sim.snapshot().state.propellant_kg));
    std::printf("[apo cutoff] apoapsis %.4f km, dv %.3f m/s (impulsive 190.034)\n",
                (el.a_m * (1.0 + el.e) - earth::radius) / 1e3, dv);
    EXPECT_NEAR(el.a_m * (1.0 + el.e), kR1000, 100.0);
    EXPECT_GE(dv, 190.034);
    EXPECT_LE(dv, 190.034 * 1.005);
}

// Planning: add_hohmann queues the first burn (cut off on the target apoapsis) and a circularise
// step; next_node shows the first as a node. A deferred circularise step, shown before its turn,
// is circularize_node of the orbit it's given. An empty plan can't be engaged.
TEST(Autopilot, PlanQueue) {
    Autopilot ap(0.5);
    EXPECT_FALSE(ap.has_plan());
    ap.engage();
    EXPECT_FALSE(ap.engaged());
    const State leo = state_from(kR300, 0.0, 0.0, 0.0, 0.0, 0.0);
    ap.add_hohmann(leo, leo.t + 1200.0, kRGeo);
    EXPECT_EQ(ap.steps(), 2u);
    const std::optional<ManeuverNode> first = ap.next_node(leo);
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(first->t, leo.t + 1200.0);
    EXPECT_NEAR(first->dv_rsw_mps.y(), 2425.732, 1e-3);
    ap.clear();
    EXPECT_FALSE(ap.has_plan());
    const double a = 0.5 * (kR300 + kRGeo);
    const State gto = state_from(a, (kRGeo - kR300) / (kRGeo + kR300), 0.0, 0.0, 0.0, 2.0);
    ap.add_circularize(true);
    const ManeuverNode expect = simcore::circularize_node(gto, true);
    const std::optional<ManeuverNode> c = ap.next_node(gto);
    ASSERT_TRUE(c.has_value());
    EXPECT_EQ(c->t, expect.t);
    EXPECT_EQ(c->dv_rsw_mps, expect.dv_rsw_mps);
}

// A Hohmann 300 -> 1000 km flown by the autopilot: two burns, the transfer coast spent on rails,
// ending circular at 1000 km. Impulsive total 190.034 + 185.355 = 375.389 m/s. Checks: 2 burns;
// on rails at some point; a within 1 km of R + 1000 km and e < 1e-3 (the first burn is targeted
// to the apoapsis, the second is computed from the orbit actually reached); total dv within 1% of
// impulsive (two short burns).
TEST(Autopilot, FliesAHohmannWithAWarpedCoast) {
    Mission m;
    Autopilot ap(0.5);
    ap.add_hohmann(m.orbit, m.orbit.t + 1200.0, kR1000);
    ap.engage();
    m.fly(ap, 20000.0);
    EXPECT_FALSE(ap.engaged());
    ASSERT_EQ(ap.burns_dv_mps().size(), 2u);
    const double total = ap.burns_dv_mps()[0] + ap.burns_dv_mps()[1];
    const simcore::KeplerianElements el = m.elements();
    std::printf("[hohmann 1000] a %.3f km, e %.2e, dv %.3f + %.3f = %.3f m/s\n",
                (el.a_m - earth::radius) / 1e3, el.e, ap.burns_dv_mps()[0], ap.burns_dv_mps()[1],
                total);
    EXPECT_TRUE(m.saw_rails);
    EXPECT_TRUE(m.saw_coast_end);
    EXPECT_TRUE(m.coast_end_ahead);
    EXPECT_NEAR(el.a_m, kR1000, 1000.0);
    EXPECT_LT(el.e, 1e-3);
    EXPECT_NEAR(total / 375.389, 1.0, 0.01);
}

// The M4 done-when check: 300 km -> GEO (35,786 km), equatorial, flown by the autopilot alone.
// ROADMAP criteria: total dv within 2% of the impulsive 3892.557 m/s, with the finite-burn loss
// reported (total - impulsive); final altitude within 10 km of GEO; e < 1e-3. Also: the orbit
// stays equatorial (i < 0.01 deg) and the vehicle had the propellant (it ends with some left).
TEST(Autopilot, FliesLeoToGeo) {
    Mission m;
    Autopilot ap(0.5);
    ap.add_hohmann(m.orbit, m.orbit.t + 1200.0, kRGeo);
    ap.engage();
    m.fly(ap, 30000.0);
    EXPECT_FALSE(ap.engaged());
    ASSERT_EQ(ap.burns_dv_mps().size(), 2u);
    const double total = ap.burns_dv_mps()[0] + ap.burns_dv_mps()[1];
    const simcore::KeplerianElements el = m.elements();
    const double r_p = el.a_m * (1.0 - el.e);
    const double r_a = el.a_m * (1.0 + el.e);
    std::printf(
        "[LEO->GEO] burns %.2f + %.2f = %.2f m/s, finite-burn loss %.2f m/s (%.2f%%); "
        "final %.2f x %.2f km, e %.2e, i %.4f deg, propellant left %.1f kg\n",
        ap.burns_dv_mps()[0], ap.burns_dv_mps()[1], total, total - 3892.557,
        100.0 * (total / 3892.557 - 1.0), (r_p - earth::radius) / 1e3, (r_a - earth::radius) / 1e3,
        el.e, el.i_rad * 180.0 / kPi, m.sim.snapshot().state.propellant_kg);
    EXPECT_GE(total, 3892.557);
    EXPECT_LE(total, 3892.557 * 1.02);
    EXPECT_NEAR(r_p, kRGeo, 10.0e3);
    EXPECT_NEAR(r_a, kRGeo, 10.0e3);
    EXPECT_LT(el.e, 1e-3);
    EXPECT_LT(el.i_rad * 180.0 / kPi, 0.01);
    EXPECT_GT(m.sim.snapshot().state.propellant_kg, 0.0);
}

// ---------------------------------------------------------------------------------------------
// M4 review fixes
// ---------------------------------------------------------------------------------------------

// Hohmann from an ellipse (300 x 2000 km, a = 7528.137 km, e = 0.112910), as the HUD plans it:
// burn at periapsis when the target is above the mean radius, at apoapsis when below. The far
// apsis to target is chosen by the target radius against the radius at the burn, not by the
// sign of dv1 (M4 review: the sign is opposite in both cases below, so the cutoff watched the
// apsis at the burn point, saw zero slope and stopped at once).
//   A: at periapsis to 1500 km: dv1 = -112.350 m/s (lowering the apoapsis 2000 -> 1500 km, an
//      apoapsis cutoff), then circularise at apoapsis, dv2 = +299.501 m/s; total 411.851 m/s.
//   B: at apoapsis to 1000 km: dv1 = +178.590 m/s (raising the periapsis 300 -> 1000 km, a
//      periapsis cutoff: the lowering-side cutoff flown for the first time), then circularise at
//      periapsis, dv2 = -229.657 m/s; total 408.247 m/s.
// Checks: circular at the target (a within 1 km, e < 1e-3); total dv within 1% of impulsive.
TEST(Autopilot, FliesAHohmannFromAnEllipseEitherWay) {
    const double a = 0.5 * (kR300 + earth::radius + 2000.0e3);
    const double e = (2000.0e3 - 300.0e3) / (2.0 * earth::radius + 2300.0e3);
    struct Case {
        double nu0, nu_burn, target_alt_m, impulsive_total;
    };
    for (const Case c : {Case{3.0, 0.0, 1500.0e3, 411.851}, Case{0.3, kPi, 1000.0e3, 408.247}}) {
        SCOPED_TRACE(c.target_alt_m);
        Mission m(state_from(a, e, 0.0, 0.0, 0.0, c.nu0));
        Autopilot ap(0.5);
        const double t = simcore::time_to_true_anomaly_s(m.orbit, c.nu_burn);
        ap.add_hohmann(m.orbit, m.orbit.t + t, earth::radius + c.target_alt_m);
        ap.engage();
        m.fly(ap, 30000.0);
        ASSERT_FALSE(ap.engaged());
        ASSERT_EQ(ap.burns_dv_mps().size(), 2u);
        const double total = ap.burns_dv_mps()[0] + ap.burns_dv_mps()[1];
        const simcore::KeplerianElements el = m.elements();
        std::printf("[ellipse -> %.0f km] a %.3f km, e %.2e, dv %.3f m/s (impulsive %.3f)\n",
                    c.target_alt_m / 1e3, (el.a_m - earth::radius) / 1e3, el.e, total,
                    c.impulsive_total);
        EXPECT_NEAR(el.a_m, earth::radius + c.target_alt_m, 1000.0);
        EXPECT_LT(el.e, 1e-3);
        EXPECT_NEAR(total / c.impulsive_total, 1.0, 0.01);
    }
}

// Disengaging mid-burn and re-engaging finishes the node instead of repeating it (M4 review: a
// 100 m/s node delivered 152.94 m/s). The step keeps the dv already delivered; the rest is flown
// as a re-planned (late) burn. Checks: total delivered 100 m/s within 0.01 m/s (two 1 mm/s
// cutoff residuals plus the re-plan), reported as one burn of the same size.
TEST(Autopilot, ReEngagingAfterAMidBurnStopDoesNotRepeatTheDeliveredDv) {
    Mission m;
    Autopilot ap(0.5);
    ap.add_node(ManeuverNode{m.orbit.t + 1200.0, Eigen::Vector3d(0.0, 100.0, 0.0)});
    ap.engage();
    const simcore::ControlLaw law = [&m, &ap](const VehicleState& s) {
        return ap.command(m.cfg, s,
                          simcore::pd_gains(m.cfg.mass.inertia_kgm2(s.propellant_kg),
                                            simcore::PdGains{0.05, 0.7}));
    };
    while (ap.engaged() && (ap.executor() == nullptr || ap.executor()->achieved_dv_mps() < 50.0)) {
        simcore::fly_with(m.sim, law, 5, 5);
    }
    ap.disengage();
    simcore::fly_with(m.sim, law, 600, 5);  // coast a minute, disengaged
    ap.engage();
    m.fly(ap, 6000.0);
    ASSERT_FALSE(ap.engaged());
    const double delivered =
        kVe * std::log(4000.0 / m.cfg.mass.mass_kg(m.sim.snapshot().state.propellant_kg));
    ASSERT_EQ(ap.burns_dv_mps().size(), 1u);
    EXPECT_NEAR(delivered, 100.0, 0.01);
    EXPECT_NEAR(ap.burns_dv_mps()[0], delivered, 0.01);
}

// A node the vehicle can't align for (no wheels, so it can never turn onto the burn) is abandoned
// once it would be more than a quarter of an orbit late (M4 review: a late re-plan had no limit,
// and a cross-track burn far from its node changes RAAN instead of inclination). Checks: the
// autopilot disengages with the step flagged aborted, and no propellant was used.
TEST(Autopilot, AbandonsANodeItCannotAlignFor) {
    VehicleConfig no_wheels = upper_stage();
    no_wheels.wheels.clear();
    const State start = state_from(kR300, 0.0, 0.0, 0.0, 0.0, 0.0);
    Mission m(start, no_wheels);
    // The vehicle starts with body x radial (90 deg from the prograde burn) and no way to turn.
    Autopilot ap(0.5);
    ap.add_node(ManeuverNode{m.orbit.t + 600.0, Eigen::Vector3d(0.0, 50.0, 0.0)});
    ap.engage();
    m.fly(ap, 6000.0);
    EXPECT_FALSE(ap.engaged());
    EXPECT_TRUE(ap.aborted());
    EXPECT_EQ(m.sim.snapshot().state.propellant_kg, 3000.0);
    EXPECT_TRUE(ap.burns_dv_mps().empty());
}

// Circularising an orbit that is already circular is a step with nothing to do: it is skipped
// rather than failing (M4 review: the zero-dv executor threw and ended the session).
TEST(Autopilot, SkipsACircularizeWithNothingToDo) {
    Mission m;
    Autopilot ap(0.5);
    ap.add_circularize(true);
    ap.engage();
    EXPECT_NO_THROW(m.fly(ap, 600.0));
    EXPECT_FALSE(ap.engaged());
    EXPECT_FALSE(ap.has_plan());
    EXPECT_EQ(m.sim.snapshot().state.propellant_kg, 3000.0);
}

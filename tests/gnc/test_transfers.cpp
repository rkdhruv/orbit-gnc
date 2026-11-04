// M4.5: transfer calculators that create maneuver nodes.
//
// Timing: on an ellipse the time from true anomaly nu to nu' is (M' - M) / n, wrapped forward to
// [0, T), with M the mean anomaly (M1.4: mean_from_true_anomaly) and n = sqrt(mu / a^3). (For a
// circular orbit elements_from_state reports the argument of latitude as nu, so the same works.)
// Hohmann (coplanar, from a horizontal velocity at radius r1 to a circle at r2), transfer ellipse
// a_t = (r1 + r2) / 2 (Curtis 6.2):
//   dv1 = sqrt(mu (2 / r1 - 1 / a_t)) - v1        (prograde at r1)
//   dv2 = sqrt(mu / r2) - sqrt(mu (2 / r2 - 1 / a_t))   (prograde at r2, half an orbit later)
//   transfer time = pi sqrt(a_t^3 / mu)
// Circularise at an apsis r: dv = sqrt(mu / r) - v_apsis (along-track).
// Plane change at an equator crossing (the line of nodes): rotate v about r_hat by di, keeping
// its size: dv_rsw = (0, v (cos di - 1), +-v sin di), |dv| = 2 v sin(di / 2).

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <stdexcept>

#include "../orbits/elements_fixtures.hpp"
#include "simcore/constants.hpp"
#include "simcore/gnc/maneuver.hpp"
#include "simcore/gnc/transfers.hpp"
#include "simcore/orbits/elements.hpp"
#include "simcore/orbits/kepler.hpp"

namespace {

using elements_fixtures::kDeg;
using elements_fixtures::state_from;
using simcore::ManeuverNode;
using simcore::State;
namespace earth = simcore::earth;

constexpr double kPi = std::numbers::pi;
const double kR300 = earth::radius + 300.0e3;
const double kRGeo = earth::radius + 35786.0e3;

// The orbit after applying `node` (from `now`), as elements.
simcore::KeplerianElements after(const State& now, const ManeuverNode& node) {
    return simcore::elements_from_state(simcore::post_burn_state(now, node));
}

}  // namespace

// Circular 420 km (T = 5578.2227 s) at nu = 1 rad: a quarter orbit to nu = 1 + pi/2 is 1394.5557 s.
// Ellipse a = 20,000 km, e = 0.7 (T = 28148.546 s): periapsis to apoapsis is T/2 = 14074.273 s;
// from nu = 2 rad (M = 0.517304) to the next periapsis is (2 pi - M) / n = 25831.035 s. A target
// equal to the current anomaly is 0 s away. Bounds: 1e-6 s circular, 1e-4 s ellipse (Kepler
// roundoff on ~1e4 s).
TEST(Transfers, TimeToTrueAnomaly) {
    const State c = state_from(earth::radius + 420.0e3, 0.0, 0.3, 0.0, 0.0, 1.0);
    EXPECT_NEAR(simcore::time_to_true_anomaly_s(c, 1.0 + kPi / 2.0), 1394.5557, 1e-3);
    EXPECT_NEAR(simcore::time_to_true_anomaly_s(c, 1.0 + kPi / 2.0),
                0.25 * 2.0 * kPi * std::sqrt(std::pow(earth::radius + 420.0e3, 3) / earth::mu),
                1e-6);
    const State e0 = state_from(2.0e7, 0.7, 0.5, 0.0, 0.0, 0.0);
    EXPECT_NEAR(simcore::time_to_true_anomaly_s(e0, kPi), 14074.273, 1e-3);
    const State e2 = state_from(2.0e7, 0.7, 0.5, 0.0, 0.0, 2.0);
    EXPECT_NEAR(simcore::time_to_true_anomaly_s(e2, 0.0), 25831.035, 1e-3);
    EXPECT_NEAR(simcore::time_to_true_anomaly_s(e2, 2.0), 0.0, 1e-6);
    // And the state there really is at that anomaly (periapsis radius a (1 - e) = 6000 km).
    const double t = simcore::time_to_true_anomaly_s(e2, 0.0);
    EXPECT_NEAR(simcore::kepler_propagate(e2, e2.t + t).r_eci.norm() / 6.0e6, 1.0, 1e-9);
}

// Hohmann 300 km -> 35,786 km (GEO), the CLAUDE.md reference: dv1 = 2425.732 m/s,
// dv2 = 1466.824 m/s, total 3892.557 m/s (~3.89 km/s), transfer 18990.21 s (5.28 h). Flown
// impulsively the plan lands on the GEO circle: after node 1 the apoapsis is r_GEO, and after
// node 2 (half an orbit later) the orbit is circular at r_GEO. Bounds: dv 1e-3 m/s; orbit
// 1e-7 relative in radius, e < 1e-8 (Kepler over half an e = 0.73 orbit plus vis-viva roundoff).
TEST(Transfers, HohmannToGeoMatchesTheReference) {
    const State leo = state_from(kR300, 0.0, 0.0, 0.0, 0.0, 0.0);
    const simcore::HohmannPlan plan = simcore::hohmann_transfer(leo, leo.t + 100.0, kRGeo);
    EXPECT_NEAR(plan.dv1_mps, 2425.732, 1e-3);
    EXPECT_NEAR(plan.dv2_mps, 1466.824, 1e-3);
    EXPECT_NEAR(plan.dv1_mps + plan.dv2_mps, 3892.557, 1e-3);
    EXPECT_NEAR(plan.transfer_s, 18990.21, 0.01);
    EXPECT_EQ(plan.first.t, leo.t + 100.0);
    EXPECT_NEAR(plan.second.t - plan.first.t, plan.transfer_s, 1e-9);
    EXPECT_NEAR(plan.first.dv_rsw_mps.y(), plan.dv1_mps, 1e-12);
    EXPECT_EQ(plan.first.dv_rsw_mps.x(), 0.0);
    EXPECT_EQ(plan.first.dv_rsw_mps.z(), 0.0);

    const simcore::KeplerianElements t1 = after(leo, plan.first);
    EXPECT_NEAR(t1.a_m * (1.0 + t1.e) / kRGeo, 1.0, 1e-7);
    const State transfer = simcore::post_burn_state(leo, plan.first);
    const simcore::KeplerianElements geo = after(transfer, plan.second);
    EXPECT_NEAR(geo.a_m / kRGeo, 1.0, 1e-7);
    EXPECT_LT(geo.e, 1e-8);
}

// Lowering is the same with retrograde burns: GEO -> 300 km gives dv1 = -1466.824 and
// dv2 = -2425.732 m/s (the mirror image), ending circular at 300 km.
TEST(Transfers, HohmannDownIsTheMirrorImage) {
    const State geo = state_from(kRGeo, 0.0, 0.0, 0.0, 0.0, 0.0);
    const simcore::HohmannPlan plan = simcore::hohmann_transfer(geo, geo.t, kR300);
    EXPECT_NEAR(plan.dv1_mps, -1466.824, 1e-3);
    EXPECT_NEAR(plan.dv2_mps, -2425.732, 1e-3);
    const simcore::KeplerianElements low =
        after(simcore::post_burn_state(geo, plan.first), plan.second);
    EXPECT_NEAR(low.a_m / kR300, 1.0, 1e-7);
    EXPECT_LT(low.e, 1e-8);
}

// Circularise on the GTO (300 x 35,786 km) from nu = 2 rad: at the next apoapsis the burn is
// sqrt(mu / r_a) - v_a = +1466.824 m/s (the Hohmann dv2), at the next periapsis
// sqrt(mu / r_p) - v_p = -2425.732 m/s; each leaves a circle at that radius (e < 1e-8).
TEST(Transfers, CirculariseAtEitherApsis) {
    const double a = 0.5 * (kR300 + kRGeo);
    const double e = (kRGeo - kR300) / (kRGeo + kR300);
    const State gto = state_from(a, e, 0.0, 0.0, 0.0, 2.0);
    const ManeuverNode at_apo = simcore::circularize_node(gto, true);
    EXPECT_NEAR(at_apo.dv_rsw_mps.y(), 1466.824, 1e-3);
    EXPECT_NEAR(at_apo.t - gto.t, simcore::time_to_true_anomaly_s(gto, kPi), 1e-9);
    const simcore::KeplerianElements c1 = after(gto, at_apo);
    EXPECT_NEAR(c1.a_m / kRGeo, 1.0, 1e-7);
    EXPECT_LT(c1.e, 1e-8);
    const ManeuverNode at_peri = simcore::circularize_node(gto, false);
    EXPECT_NEAR(at_peri.dv_rsw_mps.y(), -2425.732, 1e-3);
    const simcore::KeplerianElements c2 = after(gto, at_peri);
    EXPECT_NEAR(c2.a_m / kR300, 1.0, 1e-7);
    EXPECT_LT(c2.e, 1e-8);
}

// Plane change 51.6 -> 41.6 deg on a 420 km circle, starting at argument of latitude 30 deg: the
// next equator crossing is the descending node at 180 deg, 150/360 of an orbit away = 2324.259 s;
// |dv| = 2 v sin(5 deg) = 1334.750 m/s. Rotating v about r_hat keeps the orbit circular, its size
// and its node line: i becomes 41.6 deg, e stays ~0, a and RAAN unchanged. Bounds: time 1e-6 s,
// dv 1e-3 m/s, i and RAAN 1e-9 rad, a 1e-9 relative, e 1e-9.
TEST(Transfers, PlaneChangeAtTheNextNode) {
    const double r = earth::radius + 420.0e3;
    const State s = state_from(r, 0.0, 51.6 * kDeg, 30.0 * kDeg, 0.0, 30.0 * kDeg);
    const ManeuverNode n = simcore::plane_change_node(s, 41.6 * kDeg);
    EXPECT_NEAR(n.t - s.t, 2324.259, 1e-3);
    EXPECT_NEAR(n.dv_rsw_mps.norm(), 1334.750, 1e-3);
    EXPECT_EQ(n.dv_rsw_mps.x(), 0.0);
    const simcore::KeplerianElements el = after(s, n);
    EXPECT_NEAR(el.i_rad, 41.6 * kDeg, 1e-9);
    EXPECT_NEAR(el.raan_rad, 30.0 * kDeg, 1e-9);
    EXPECT_NEAR(el.a_m / r, 1.0, 1e-9);
    EXPECT_LT(el.e, 1e-9);
}

// Plane change on eccentric orbits, at a node away from the apsides (M4 review: the burn used |v|
// instead of the horizontal speed, so it changed the orbit's size and missed the inclination).
// Rotating v about r_hat leaves its radial part alone: dv_rsw = (0, v_S (cos di - 1),
// +-v_S sin di), |dv| = 2 v_S sin(di / 2), with v_S the along-track speed.
//   GTO (300 x 35,786 km, a = 24421.137 km, e = 0.726543), i 28.5 -> 0 deg, argp 90 deg, from
//   periapsis: the next node is the descending one at nu = 90 deg, where r = p = 11530.089 km and
//   v_S = sqrt(mu / p) = 5879.665 m/s (v_R = e v_S = 4271.828 m/s): |dv| = 2894.598 m/s (|v|
//   would have given 3577.920).
//   a = 10,000 km, e = 0.3, i 20 -> 10 deg, argp 60 deg, from periapsis: node at nu = 120 deg,
//   v_S = 5625.578 m/s, |dv| = 980.603 m/s.
// Checks: inclination on target, and a, e and the speed unchanged (1e-9). Bounds: dv 1e-3 m/s.
TEST(Transfers, PlaneChangeOnEccentricOrbits) {
    const double a_gto = 0.5 * (kR300 + kRGeo);
    const double e_gto = (kRGeo - kR300) / (kRGeo + kR300);
    struct Case {
        double a, e, i0, argp, i1, dv;
    };
    for (const Case c : {Case{a_gto, e_gto, 28.5 * kDeg, 90.0 * kDeg, 0.0, 2894.598},
                         Case{1.0e7, 0.3, 20.0 * kDeg, 60.0 * kDeg, 10.0 * kDeg, 980.603}}) {
        SCOPED_TRACE(c.dv);
        const State s = state_from(c.a, c.e, c.i0, 30.0 * kDeg, c.argp, 0.0);
        const ManeuverNode n = simcore::plane_change_node(s, c.i1);
        EXPECT_NEAR(n.dv_rsw_mps.norm(), c.dv, 1e-3);
        const State at = simcore::node_state(s, n);
        const State post = simcore::post_burn_state(s, n);
        const simcore::KeplerianElements el = simcore::elements_from_state(post);
        EXPECT_NEAR(el.i_rad, c.i1, 1e-9);
        EXPECT_NEAR(el.a_m / c.a, 1.0, 1e-9);
        EXPECT_NEAR(el.e, c.e, 1e-9);
        EXPECT_NEAR(post.v_eci.norm() / at.v_eci.norm(), 1.0, 1e-12);
    }
}

TEST(Transfers, RejectsInvalidRequests) {
    const State hyper = state_from(-7.0e6, 2.0, 0.3, 0.0, 0.0, 0.4);
    EXPECT_THROW((void)simcore::time_to_true_anomaly_s(hyper, 1.0), std::invalid_argument);
    const State eq = state_from(kR300, 0.0, 0.0, 0.0, 0.0, 0.0);
    EXPECT_THROW((void)simcore::plane_change_node(eq, 10.0 * kDeg),
                 std::invalid_argument);  // no node line
    EXPECT_THROW((void)simcore::hohmann_transfer(eq, eq.t, 0.0), std::invalid_argument);
    // Hohmann needs a horizontal velocity at the first burn: not mid-way round an ellipse.
    const State el = state_from(2.0e7, 0.7, 0.5, 0.0, 0.0, 1.0);
    EXPECT_THROW((void)simcore::hohmann_transfer(el, el.t, kRGeo), std::invalid_argument);
}

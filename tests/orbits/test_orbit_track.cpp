// M3.2: points along the osculating two-body orbit, for drawing the orbit line.
//
// From the state alone (no element angles): h = r x v, p = |h|^2 / mu, eccentricity vector
// e = v x h / mu - r_hat. Perifocal axes: P = e_hat (or r_hat when the orbit is circular, e <
// 1e-12), Q = h_hat x P. The conic is
//   r(nu) = p / (1 + e cos nu) (cos nu P + sin nu Q)
// Closed orbits (e < 1): n points at equal steps of true anomaly from the current position,
// nu_k = nu0 + 2 pi k / n (equal-nu spacing puts more points near periapsis, where the curve
// bends most). The ring closes from the last point back to the first.
// Open orbits (e >= 1): n points with nu from -nu_max to +nu_max, where r(nu_max) = r_max:
// cos nu_max = (p / r_max - 1) / e. Empty when even periapsis is beyond r_max.
//
// Tolerances: each point is a few dozen flops from r and v (cross products, a sqrt, cos/sin), so
// it is exact to ~1e-15 relative; bounds 1e-12 relative.

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "elements_fixtures.hpp"
#include "simcore/constants.hpp"
#include "simcore/orbits/orbit_track.hpp"

namespace {

using elements_fixtures::kDeg;
using elements_fixtures::kPi;
using elements_fixtures::state_from;
using simcore::sample_orbit_eci;
using simcore::State;
namespace earth = simcore::earth;

constexpr double kRel = 1e-12;
constexpr double kNoLimit = 1e9;  // r_max far beyond any closed orbit here (ignored for e < 1)

}  // namespace

// Circular 420 km at i = 51.6 deg: every point at radius a, in the orbit plane, starting at the
// current position, equally spaced (chord 2 a sin(pi / n)) and ordered in the direction of motion.
TEST(OrbitTrack, CircularOrbitIsAnEvenRingThroughTheCurrentPosition) {
    const double a = earth::radius + 420.0e3;
    const State s = state_from(a, 0.0, 51.6 * kDeg, 30.0 * kDeg, 0.0, 1.0);
    const std::size_t n = 360;
    const std::vector<Eigen::Vector3d> pts = sample_orbit_eci(s, static_cast<int>(n), kNoLimit);
    ASSERT_EQ(pts.size(), n);
    const Eigen::Vector3d h_hat = s.r_eci.cross(s.v_eci).normalized();
    const double chord = 2.0 * a * std::sin(kPi / static_cast<double>(n));
    EXPECT_LE((pts[0] - s.r_eci).norm(), kRel * a);
    for (std::size_t k = 0; k < n; ++k) {
        SCOPED_TRACE(k);
        EXPECT_NEAR(pts[k].norm() / a, 1.0, kRel);
        EXPECT_LE(std::abs(pts[k].dot(h_hat)), kRel * a);
        EXPECT_NEAR((pts[(k + 1) % n] - pts[k]).norm() / chord, 1.0, 1e-9);
    }
    EXPECT_GT((pts[1] - pts[0]).dot(s.v_eci), 0.0);
}

// Ellipse a = 20,000 km, e = 0.7, started at periapsis: p = a (1 - e^2) = 10,200 km, and point k
// lies at nu_k = 2 pi k / n with |r_k| = p / (1 + e cos nu_k). Point 0 is periapsis (6,000 km) and
// point n/2 apoapsis (34,000 km). Started elsewhere (nu = 2 rad), point 0 is the current position.
TEST(OrbitTrack, EllipsePointsLieOnTheConicAtEqualTrueAnomalySteps) {
    const double a = 2.0e7;
    const double e = 0.7;
    const double p = a * (1.0 - e * e);
    const std::size_t n = 400;
    const State s = state_from(a, e, 30.0 * kDeg, 40.0 * kDeg, 60.0 * kDeg, 0.0);
    const std::vector<Eigen::Vector3d> pts = sample_orbit_eci(s, static_cast<int>(n), kNoLimit);
    ASSERT_EQ(pts.size(), n);
    EXPECT_NEAR(pts[0].norm() / 6.0e6, 1.0, kRel);
    EXPECT_NEAR(pts[n / 2].norm() / 3.4e7, 1.0, kRel);
    const Eigen::Vector3d periapsis_hat = pts[0].normalized();
    for (std::size_t k = 0; k < n; ++k) {
        SCOPED_TRACE(k);
        const double nu = 2.0 * kPi * static_cast<double>(k) / static_cast<double>(n);
        EXPECT_NEAR(pts[k].norm() / (p / (1.0 + e * std::cos(nu))), 1.0, kRel);
        EXPECT_NEAR(pts[k].normalized().dot(periapsis_hat), std::cos(nu), 1e-12);
    }
    const State s2 = state_from(a, e, 30.0 * kDeg, 40.0 * kDeg, 60.0 * kDeg, 2.0);
    EXPECT_LE((sample_orbit_eci(s2, static_cast<int>(n), kNoLimit)[0] - s2.r_eci).norm(),
              kRel * s2.r_eci.norm());
}

// Hyperbola e = 2, periapsis 7,000 km: a = r_p / (1 - e) = -7,000 km, p = a (1 - e^2) = 21,000 km.
// Clipped at r_max = 50,000 km: cos nu_max = (21 / 50 - 1) / 2 = -0.29, nu_max = 1.86502 rad. With
// n = 101 the endpoints sit at r_max, the middle point is periapsis, and nu increases along the
// list (the direction of motion). A limit inside periapsis gives no points.
TEST(OrbitTrack, HyperbolaIsClippedAtTheMaximumRadius) {
    const double e = 2.0;
    const double rp = 7.0e6;
    const double a = rp / (1.0 - e);
    const double p = a * (1.0 - e * e);
    const double r_max = 5.0e7;
    const double nu_max = std::acos((p / r_max - 1.0) / e);
    EXPECT_NEAR(nu_max, 1.86502, 1e-5);  // hand value
    const State s = state_from(a, e, 10.0 * kDeg, 0.0, 20.0 * kDeg, 0.5);
    const std::size_t n = 101;
    const std::vector<Eigen::Vector3d> pts = sample_orbit_eci(s, static_cast<int>(n), r_max);
    ASSERT_EQ(pts.size(), n);
    EXPECT_NEAR(pts.front().norm() / r_max, 1.0, kRel);
    EXPECT_NEAR(pts.back().norm() / r_max, 1.0, kRel);
    EXPECT_NEAR(pts[n / 2].norm() / rp, 1.0, kRel);
    const Eigen::Vector3d periapsis_hat = pts[n / 2].normalized();
    const Eigen::Vector3d h = s.r_eci.cross(s.v_eci);
    for (std::size_t k = 0; k < n; ++k) {
        SCOPED_TRACE(k);
        const double nu =
            -nu_max + 2.0 * nu_max * static_cast<double>(k) / static_cast<double>(n - 1);
        EXPECT_NEAR(pts[k].norm() / (p / (1.0 + e * std::cos(nu))), 1.0, kRel);
        EXPECT_NEAR(pts[k].normalized().dot(periapsis_hat), std::cos(nu), 1e-12);
        if (k > 0) {
            EXPECT_GT(pts[k - 1].cross(pts[k]).dot(h), 0.0);  // turning the same way as the orbit
        }
    }
    EXPECT_TRUE(sample_orbit_eci(s, static_cast<int>(n), 6.0e6).empty());
}

TEST(OrbitTrack, RejectsInvalidInput) {
    const double kNaN = std::numeric_limits<double>::quiet_NaN();
    const State s = state_from(earth::radius + 420.0e3, 0.0, 0.0, 0.0, 0.0, 0.0);
    EXPECT_THROW((void)sample_orbit_eci(s, 2, kNoLimit), std::invalid_argument);
    EXPECT_THROW((void)sample_orbit_eci(s, 100, 0.0), std::invalid_argument);
    EXPECT_THROW((void)sample_orbit_eci(s, 100, kNaN), std::invalid_argument);
    State radial = s;
    radial.v_eci = 1000.0 * s.r_eci.normalized();  // h = 0: no orbit plane
    EXPECT_THROW((void)sample_orbit_eci(radial, 100, kNoLimit), std::invalid_argument);
    State at_centre = s;
    at_centre.r_eci = Eigen::Vector3d::Zero();
    EXPECT_THROW((void)sample_orbit_eci(at_centre, 100, kNoLimit), std::invalid_argument);
}

// M2.1: atmospheric drag with a piecewise-exponential atmosphere.
//
// Density (Vallado, "Fundamentals of Astrodynamics", Table 8-4): in the band with base altitude
// h0 <= h < h1, rho(h) = rho0 exp(-(h - h0) / H), 28 bands from 0 to 1000 km; below 0 the first
// band and above 1000 km the last band are extrapolated. Altitude above a spherical Earth,
// h = |r| - R.
// Drag (Vallado eq. 8-29):  a = -1/2 rho (Cd A / m) |v_rel| v_rel,  v_rel = v - w x r,
// with the atmosphere co-rotating with Earth at w = 7.292115e-5 rad/s about z (configurable).
// Mass is a constructor parameter for now; M2.3 moves it into the vehicle state.
//
// Roundoff tolerance kRel = 1e-13 (a few dozen flops, exp and sqrt).

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "../orbits/elements_fixtures.hpp"
#include "simcore/constants.hpp"
#include "simcore/dynamics/exponential_drag.hpp"
#include "simcore/dynamics/force_sum.hpp"
#include "simcore/dynamics/point_mass_gravity.hpp"
#include "simcore/integrators/dormand_prince.hpp"
#include "simcore/orbits/elements.hpp"

namespace {

using elements_fixtures::kDeg;
using elements_fixtures::state_from;
using simcore::Epoch;
using simcore::ExponentialDrag;
using simcore::State;
namespace earth = simcore::earth;

constexpr double kRel = 1e-13;
const Epoch kT0 = Epoch::from_tt_seconds_since_j2000(800'000'000, 0.25);

// Vallado Table 8-4: base altitude [km], base density [kg/m^3], scale height [km].
struct Band {
    double h0_km, rho0, H_km;
};
const std::vector<Band> kTable{
    {0, 1.225, 7.249},         {25, 3.899e-2, 6.349},    {30, 1.774e-2, 6.682},
    {40, 3.972e-3, 7.554},     {50, 1.057e-3, 8.382},    {60, 3.206e-4, 7.714},
    {70, 8.770e-5, 6.549},     {80, 1.905e-5, 5.799},    {90, 3.396e-6, 5.382},
    {100, 5.297e-7, 5.877},    {110, 9.661e-8, 7.263},   {120, 2.438e-8, 9.473},
    {130, 8.484e-9, 12.636},   {140, 3.845e-9, 16.149},  {150, 2.070e-9, 22.523},
    {180, 5.464e-10, 29.740},  {200, 2.789e-10, 37.105}, {250, 7.248e-11, 45.546},
    {300, 2.418e-11, 53.628},  {350, 9.518e-12, 53.298}, {400, 3.725e-12, 58.515},
    {450, 1.585e-12, 60.828},  {500, 6.967e-13, 63.822}, {600, 1.454e-13, 71.835},
    {700, 3.614e-14, 88.667},  {800, 1.170e-14, 124.64}, {900, 5.245e-15, 181.05},
    {1000, 3.019e-15, 268.00},
};

}  // namespace

// ---------------------------------------------------------------------------------------------
// Density
// ---------------------------------------------------------------------------------------------

// At each band's base altitude the density is the tabulated rho0 exactly (exp(0) = 1).
TEST(ExponentialAtmosphere, DensityAtBandBasesIsTabulated) {
    for (const Band& b : kTable) {
        SCOPED_TRACE(b.h0_km);
        EXPECT_NEAR(ExponentialDrag::density_kgpm3(b.h0_km * 1e3) / b.rho0, 1.0, kRel);
    }
}

// Inside a band: rho(h0 + H) = rho0 / e, e.g. 400 km band: 3.725e-12 / e at 458.515 km.
TEST(ExponentialAtmosphere, DensityFallsByEPerScaleHeight) {
    for (const Band& b : kTable) {
        const double h = (b.h0_km + 0.5 * b.H_km) * 1e3;
        SCOPED_TRACE(b.h0_km);
        EXPECT_NEAR(ExponentialDrag::density_kgpm3(h) / (b.rho0 * std::exp(-0.5)), 1.0, kRel);
    }
}

// The table is built to be nearly continuous: extrapolating band k to band k+1's base gives
// band k+1's rho0 to within 0.14% (largest mismatch, at 25 km; the rest are < 0.02%). Checking it
// guards against a mistyped digit in the table. Bound 0.2%.
TEST(ExponentialAtmosphere, DensityIsNearlyContinuousAcrossBands) {
    for (std::size_t k = 1; k < kTable.size(); ++k) {
        SCOPED_TRACE(kTable[k].h0_km);
        const double below = ExponentialDrag::density_kgpm3(kTable[k].h0_km * 1e3 - 1e-6);
        const double above = ExponentialDrag::density_kgpm3(kTable[k].h0_km * 1e3);
        EXPECT_NEAR(below / above, 1.0, 2e-3);
    }
}

// Monotonic decrease from 0 to 1500 km (including the extrapolated region above 1000 km), and
// extrapolation below 0 km uses the first band.
TEST(ExponentialAtmosphere, DensityDecreasesMonotonicallyAndExtrapolates) {
    double prev = ExponentialDrag::density_kgpm3(0.0);
    for (double h = 1e3; h <= 1.5e6; h += 1e3) {
        const double rho = ExponentialDrag::density_kgpm3(h);
        ASSERT_LT(rho, prev) << "h = " << h;
        prev = rho;
    }
    EXPECT_NEAR(ExponentialDrag::density_kgpm3(-1e3) / (1.225 * std::exp(1.0 / 7.249)), 1.0, kRel);
    EXPECT_NEAR(ExponentialDrag::density_kgpm3(1.2e6) / (3.019e-15 * std::exp(-200.0 / 268.0)), 1.0,
                kRel);
}

// ---------------------------------------------------------------------------------------------
// Acceleration
// ---------------------------------------------------------------------------------------------

// Non-rotating atmosphere: a = -1/2 rho B |v| v with B = Cd A / m = 2.2 * 1 / 100 = 0.022 m^2/kg.
// At 400 km (rho = 3.725e-12) and v = 7668.56 m/s: |a| = 0.5 * 3.725e-12 * 0.022 * 7668.56^2
// = 2.4096e-6 m/s^2, opposite to v.
TEST(ExponentialDrag, OpposesVelocityWithQuadraticMagnitude) {
    const ExponentialDrag drag(2.2, 1.0, 100.0, 0.0);
    const double r = earth::radius + 400.0e3;
    const Eigen::Vector3d v(0.0, 3000.0, std::sqrt(earth::mu / r - 9.0e6));
    const State s{kT0, Eigen::Vector3d(r, 0.0, 0.0), v};
    const Eigen::Vector3d a = drag.acceleration_eci(s);
    const double expected = 0.5 * 3.725e-12 * 0.022 * v.squaredNorm();
    EXPECT_NEAR(a.norm() / expected, 1.0, kRel);
    EXPECT_NEAR(a.normalized().dot(v.normalized()), -1.0, 1e-15);
    const double v_circ = std::sqrt(earth::mu / r);
    EXPECT_NEAR(0.5 * 3.725e-12 * 0.022 * v_circ * v_circ, 2.4096e-6, 1e-10);  // hand value
}

// Co-rotating atmosphere: v_rel = v - w x r. For an equatorial prograde circular orbit at 400 km,
// w r = 7.292115e-5 * 6,778,137 = 494.27 m/s, so |v_rel| = 7668.56 - 494.27 = 7174.29 m/s and the
// drag shrinks by (7174.29 / 7668.56)^2 = 0.8752 relative to a non-rotating atmosphere.
TEST(ExponentialDrag, UsesVelocityRelativeToCoRotatingAtmosphere) {
    const double r = earth::radius + 400.0e3;
    const double v = std::sqrt(earth::mu / r);
    const State s{kT0, Eigen::Vector3d(r, 0.0, 0.0), Eigen::Vector3d(0.0, v, 0.0)};
    const Eigen::Vector3d a = ExponentialDrag(2.2, 1.0, 100.0).acceleration_eci(s);
    const double v_rel = v - earth::rotation_rate_radps * r;
    EXPECT_NEAR(v_rel, 7174.29, 1e-2);  // hand value
    EXPECT_NEAR(a.y() / (-0.5 * 3.725e-12 * 0.022 * v_rel * v_rel), 1.0, kRel);
    EXPECT_EQ(a.x(), 0.0);
    EXPECT_EQ(a.z(), 0.0);
}

// A body at rest relative to the atmosphere (v = w x r) feels no drag.
TEST(ExponentialDrag, IsZeroWhenCoRotatingWithAtmosphere) {
    const Eigen::Vector3d r(4.0e6, 5.0e6, 1.0e6);
    const Eigen::Vector3d v = Eigen::Vector3d(0.0, 0.0, earth::rotation_rate_radps).cross(r);
    EXPECT_EQ(ExponentialDrag(2.2, 1.0, 100.0).acceleration_eci(State{kT0, r, v}),
              Eigen::Vector3d::Zero());
}

TEST(ExponentialDrag, RejectsInvalidParameters) {
    const double kNaN = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((void)ExponentialDrag(0.0, 1.0, 100.0), std::invalid_argument);
    EXPECT_THROW((void)ExponentialDrag(2.2, -1.0, 100.0), std::invalid_argument);
    EXPECT_THROW((void)ExponentialDrag(2.2, 1.0, 0.0), std::invalid_argument);
    EXPECT_THROW((void)ExponentialDrag(2.2, 1.0, kNaN), std::invalid_argument);
    EXPECT_THROW((void)ExponentialDrag(2.2, 1.0, 100.0, kNaN), std::invalid_argument);
    EXPECT_THROW((void)ExponentialDrag::density_kgpm3(kNaN), std::invalid_argument);
}

// ---------------------------------------------------------------------------------------------
// Orbit decay
// ---------------------------------------------------------------------------------------------

// Circular orbit decay rate (drag only, non-rotating atmosphere): from dE/dt = v . a_drag and
// E = -mu / (2a), da/dt = -rho B sqrt(mu a). At 400 km with B = 0.022: -4.2596e-3 m/s,
// i.e. -368.0 m/day. Over one day the orbit drops ~0.37 km, so rho rises by exp(0.368 / 58.5) - 1
// = 0.63%; to first order the mean rate is 0.31% above the starting rate. We compare the
// osculating semi-major axis after one day with rate0 * 1.0031 * 86400 s. Tolerance 1%: covers
// neglected second-order density growth (< 0.01%), osculating-vs-mean a (drag keeps the orbit
// circular, so the short-period part is ~1e-9 relative) and integrator error (cm).
TEST(ExponentialDrag, CircularOrbitDecayMatchesAnalyticRate) {
    const double a0 = earth::radius + 400.0e3;
    const State s0 = state_from(a0, 0.0, 51.6 * kDeg, 30.0 * kDeg, 0.0, 0.0);
    simcore::ForceSum f;
    f.add(std::make_shared<simcore::PointMassGravity>())
        .add(std::make_shared<ExponentialDrag>(2.2, 1.0, 100.0, 0.0));
    simcore::DpOptions o;
    o.rtol = 1e-12;
    o.atol_pos_m = 1e-6;
    o.atol_vel_mps = 1e-9;
    simcore::DormandPrince54 dp(o);
    const State s1 = dp.propagate(f, s0, s0.t + 86400.0);
    const double rate0 = -3.725e-12 * 0.022 * std::sqrt(earth::mu * a0);
    EXPECT_NEAR(rate0 * 86400.0, -368.03, 0.01);  // hand value
    const double da = simcore::elements_from_state(s1).a_m - a0;
    EXPECT_NEAR(da / (rate0 * 1.0031 * 86400.0), 1.0, 0.01);
}

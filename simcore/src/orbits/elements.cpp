#include "simcore/orbits/elements.hpp"

#include <cmath>
#include <numbers>

#include <Eigen/Geometry>  // cross(): declared in Core, defined in Geometry

#include "simcore/check.hpp"

namespace simcore {

namespace {

constexpr double kTwoPi = 2.0 * std::numbers::pi;
constexpr double kSingular = 1e-11;  // e or sin i below this: circular / equatorial
constexpr double kParabolic = 1e-9;  // |1 - e| below this: a undefined

// Wrap an angle into [0, 2 pi). (x + 2 pi can round up to exactly 2 pi for tiny negative x.)
double wrap_2pi(double x) {
    x = std::fmod(x, kTwoPi);
    if (x < 0.0) {
        x += kTwoPi;
    }
    return x >= kTwoPi ? 0.0 : x;
}

// Signed angle from x to y, measured positive about the axis k_hat:
// atan2(k_hat . (x cross y), x . y). Unlike acos(x . y / |x||y|) this is well conditioned for all
// angles and gives the full [0, 2 pi) range, so retrograde orbits need no special cases.
double angle_about(const Eigen::Vector3d& k_hat, const Eigen::Vector3d& x,
                   const Eigen::Vector3d& y) {
    return wrap_2pi(std::atan2(k_hat.dot(x.cross(y)), x.dot(y)));
}

}  // namespace

KeplerianElements elements_from_state(const State& s, double mu) {
    SIMCORE_CHECK(std::isfinite(mu) && mu > 0.0, "mu must be finite and > 0");
    const Eigen::Vector3d& r = s.r_eci;
    const Eigen::Vector3d& v = s.v_eci;
    SIMCORE_CHECK(r.allFinite() && v.allFinite(), "state must be finite");
    const double rn = r.norm();
    const double vn = v.norm();
    SIMCORE_CHECK(rn > 0.0, "r must be nonzero");

    // Specific angular momentum h = r x v: normal to the orbit plane.
    const Eigen::Vector3d h = r.cross(v);
    const double hn = h.norm();
    SIMCORE_CHECK(hn > 1e-12 * rn * vn && hn > 0.0, "rectilinear orbit (h ~ 0) not supported");
    const Eigen::Vector3d h_hat = h / hn;

    // Eccentricity vector, pointing at periapsis: e = (v x h) / mu - r_hat (Laplace-Runge-Lenz
    // vector divided by mu). Its length is the eccentricity.
    const Eigen::Vector3d e_vec = v.cross(h) / mu - r / rn;
    const double e = e_vec.norm();
    SIMCORE_CHECK(std::abs(1.0 - e) >= kParabolic, "near-parabolic orbit: a is undefined");

    // Vis-viva: energy = v^2/2 - mu/r = -mu / (2 a).
    const double energy = 0.5 * vn * vn - mu / rn;
    KeplerianElements el{};
    el.a_m = -mu / (2.0 * energy);
    el.e = e;

    // Inclination from h: atan2(|h_xy|, h_z) rather than acos(h_z/|h|), which loses accuracy
    // near 0 and pi.
    const double h_xy = std::hypot(h.x(), h.y());
    el.i_rad = std::atan2(h_xy, h.z());

    // Node line n = z x h = (-h_y, h_x, 0) points at the ascending node.
    const Eigen::Vector3d node(-h.y(), h.x(), 0.0);
    const bool equatorial = h_xy < kSingular * hn;  // sin i < 1e-11
    const bool circular = e < kSingular;
    Eigen::Vector3d ref;  // reference direction in the plane that argp (or nu) is measured from
    if (equatorial) {
        el.raan_rad = 0.0;
        ref = Eigen::Vector3d::UnitX();
    } else {
        el.raan_rad = wrap_2pi(std::atan2(node.y(), node.x()));
        ref = node;
    }
    if (circular) {
        el.argp_rad = 0.0;
        el.nu_rad = angle_about(h_hat, ref, r);  // argument of latitude (or true longitude)
    } else {
        el.argp_rad = angle_about(h_hat, ref, e_vec);
        el.nu_rad = angle_about(h_hat, e_vec, r);
    }
    return el;
}

State state_from_elements(const KeplerianElements& el, const Epoch& t, double mu) {
    SIMCORE_CHECK(std::isfinite(mu) && mu > 0.0, "mu must be finite and > 0");
    SIMCORE_CHECK(std::isfinite(el.a_m) && std::isfinite(el.e) && std::isfinite(el.i_rad) &&
                      std::isfinite(el.raan_rad) && std::isfinite(el.argp_rad) &&
                      std::isfinite(el.nu_rad),
                  "elements must be finite");
    SIMCORE_CHECK(el.e >= 0.0, "e must be >= 0");
    SIMCORE_CHECK(std::abs(1.0 - el.e) >= kParabolic, "near-parabolic orbit not supported");
    SIMCORE_CHECK((el.e < 1.0) ? el.a_m > 0.0 : el.a_m < 0.0,
                  "a must be > 0 for an ellipse and < 0 for a hyperbola");

    // Semi-latus rectum p = a (1 - e^2) > 0 for both conics; orbit equation r = p / (1 + e cos nu).
    const double p = el.a_m * (1.0 - el.e * el.e);
    const double cn = std::cos(el.nu_rad);
    const double sn = std::sin(el.nu_rad);
    const double denom = 1.0 + el.e * cn;
    SIMCORE_CHECK(denom > 0.0, "true anomaly beyond the hyperbolic asymptote");
    const double r = p / denom;
    const double vk = std::sqrt(mu / p);

    // Perifocal basis P (to periapsis), Q (90 deg ahead) in ECI: columns 1 and 2 of
    // Rz(raan) Rx(i) Rz(argp).
    const double cO = std::cos(el.raan_rad), sO = std::sin(el.raan_rad);
    const double ci = std::cos(el.i_rad), si = std::sin(el.i_rad);
    const double cw = std::cos(el.argp_rad), sw = std::sin(el.argp_rad);
    const Eigen::Vector3d P(cO * cw - sO * sw * ci, sO * cw + cO * sw * ci, sw * si);
    const Eigen::Vector3d Q(-cO * sw - sO * cw * ci, -sO * sw + cO * cw * ci, cw * si);

    // r = r (cos nu P + sin nu Q);  v = sqrt(mu/p) (-sin nu P + (e + cos nu) Q).
    return State{t, (r * cn) * P + (r * sn) * Q, (-vk * sn) * P + (vk * (el.e + cn)) * Q};
}

}  // namespace simcore

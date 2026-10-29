#include "simcore/orbits/kepler.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>

#include "simcore/check.hpp"
#include "simcore/orbits/elements.hpp"

namespace simcore {

namespace {

constexpr double kTwoPi = 2.0 * std::numbers::pi;
constexpr double kEps = std::numeric_limits<double>::epsilon();
constexpr int kMaxNewton = 50;

// Newton has converged when the step is down to the noise floor. f is evaluated with an absolute
// rounding error of a few eps * scale (scale = the size of its largest term), so the attainable
// step is ~eps * scale / |f'|. When f' is small (e near 1 close to periapsis: f' = 1 - e cos E
// can be 0.04), that floor sits well above eps |x|, and a test on eps |x| alone never passes.
bool converged(double step, double scale, double fprime) {
    return std::abs(step) <= 4.0 * kEps * std::max(1.0, scale) / std::abs(fprime);
}

}  // namespace

double solve_kepler_elliptic(double M_rad, double e) {
    SIMCORE_CHECK(std::isfinite(M_rad), "M must be finite");
    SIMCORE_CHECK(e >= 0.0 && e < 1.0, "elliptic Kepler needs 0 <= e < 1");
    if (e == 0.0) {
        return M_rad;
    }
    // Solve on the principal revolution [-pi, pi], then add the whole revolutions back.
    const double Mr = std::remainder(M_rad, kTwoPi);
    const double offset = M_rad - Mr;
    // Danby (1987): E0 = M + 0.85 e sign(sin M) converges for every e < 1.
    const double s = std::sin(Mr);
    const double sign_s = (s > 0.0) ? 1.0 : ((s < 0.0) ? -1.0 : 0.0);
    double E = Mr + 0.85 * e * sign_s;
    for (int it = 0; it < kMaxNewton; ++it) {
        // Newton on f(E) = E - e sin E - M, f'(E) = 1 - e cos E.
        const double fp = 1.0 - e * std::cos(E);
        const double dE = (E - e * std::sin(E) - Mr) / fp;
        E -= dE;
        if (converged(dE, std::abs(E) + std::abs(Mr), fp)) {
            return offset + E;
        }
    }
    SIMCORE_CHECK(false, "elliptic Kepler solver did not converge");
    return E;  // unreachable
}

double solve_kepler_hyperbolic(double M_rad, double e) {
    SIMCORE_CHECK(std::isfinite(M_rad), "M must be finite");
    SIMCORE_CHECK(std::isfinite(e) && e > 1.0, "hyperbolic Kepler needs e > 1");
    if (M_rad == 0.0) {
        return 0.0;
    }
    // Start from e sinh H ~ M. f(H) = e sinh H - H - M is convex for H > 0 (concave for H < 0),
    // and this guess sits on the near side of the root, so Newton overshoots at most once and
    // then converges monotonically.
    double H = std::asinh(M_rad / e);
    for (int it = 0; it < kMaxNewton; ++it) {
        const double fp = e * std::cosh(H) - 1.0;
        const double dH = (e * std::sinh(H) - H - M_rad) / fp;
        H -= dH;
        if (converged(dH, e * std::cosh(H) * std::max(1.0, std::abs(H)), fp)) {
            return H;
        }
    }
    SIMCORE_CHECK(false, "hyperbolic Kepler solver did not converge");
    return H;  // unreachable
}

double mean_from_true_anomaly(double nu_rad, double e) {
    SIMCORE_CHECK(std::isfinite(nu_rad) && std::isfinite(e) && e >= 0.0, "invalid nu or e");
    SIMCORE_CHECK(std::abs(1.0 - e) >= 1e-9, "near-parabolic orbit not supported");
    const double nu = std::remainder(nu_rad, kTwoPi);  // [-pi, pi]
    if (e < 1.0) {
        // tan(E/2) = sqrt((1-e)/(1+e)) tan(nu/2), in atan2 form (cos(nu/2) >= 0 here).
        const double E = 2.0 * std::atan2(std::sqrt(1.0 - e) * std::sin(0.5 * nu),
                                          std::sqrt(1.0 + e) * std::cos(0.5 * nu));
        return E - e * std::sin(E);
    }
    SIMCORE_CHECK(1.0 + e * std::cos(nu) > 0.0, "true anomaly beyond the hyperbolic asymptote");
    // tanh(H/2) = sqrt((e-1)/(e+1)) tan(nu/2).
    const double H = 2.0 * std::atanh(std::sqrt((e - 1.0) / (e + 1.0)) * std::tan(0.5 * nu));
    return e * std::sinh(H) - H;
}

double true_from_mean_anomaly(double M_rad, double e) {
    SIMCORE_CHECK(std::isfinite(e) && e >= 0.0, "invalid e");
    SIMCORE_CHECK(std::abs(1.0 - e) >= 1e-9, "near-parabolic orbit not supported");
    if (e < 1.0) {
        const double E = solve_kepler_elliptic(M_rad, e);
        // tan(nu/2) = sqrt((1+e)/(1-e)) tan(E/2), in atan2 form, then wrapped to [-pi, pi].
        const double nu = 2.0 * std::atan2(std::sqrt(1.0 + e) * std::sin(0.5 * E),
                                           std::sqrt(1.0 - e) * std::cos(0.5 * E));
        return std::remainder(nu, kTwoPi);
    }
    const double H = solve_kepler_hyperbolic(M_rad, e);
    return 2.0 * std::atan(std::sqrt((e + 1.0) / (e - 1.0)) * std::tanh(0.5 * H));
}

State kepler_propagate(const State& s0, const Epoch& t, double mu) {
    SIMCORE_CHECK(std::isfinite(mu) && mu > 0.0, "mu must be finite and > 0");
    const double dt_s = t - s0.t;
    if (dt_s == 0.0) {
        return s0;
    }
    KeplerianElements el = elements_from_state(s0, mu);
    // Mean anomaly advances uniformly: M = M0 + n dt, with mean motion n = sqrt(mu / |a|^3).
    const double a_abs = std::abs(el.a_m);
    const double n = std::sqrt(mu / (a_abs * a_abs * a_abs));
    const double M = mean_from_true_anomaly(el.nu_rad, el.e) + n * dt_s;
    el.nu_rad = true_from_mean_anomaly(M, el.e);
    return state_from_elements(el, t, mu);
}

}  // namespace simcore

#include "simcore/events/brent.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "simcore/check.hpp"

namespace simcore {

double brent_root(const std::function<double(double)>& g, double a, double b, double xtol,
                  int max_iter) {
    SIMCORE_CHECK(std::isfinite(a) && std::isfinite(b), "bracket must be finite");
    SIMCORE_CHECK(std::isfinite(xtol) && xtol > 0.0, "xtol must be finite and > 0");
    SIMCORE_CHECK(max_iter > 0, "max_iter must be > 0");
    double fa = g(a);
    double fb = g(b);
    SIMCORE_CHECK(std::isfinite(fa) && std::isfinite(fb), "g must be finite at the bracket");
    if (fa == 0.0) {
        return a;
    }
    if (fb == 0.0) {
        return b;
    }
    SIMCORE_CHECK((fa > 0.0) != (fb > 0.0), "g(a) and g(b) must have opposite signs");

    constexpr double kEps = std::numeric_limits<double>::epsilon();
    // b = best estimate, a = previous b, c = the point keeping the sign change [b, c].
    double c = b;
    double fc = fb;
    double d = b - a;  // last step
    double e = d;      // step before last
    for (int iter = 0; iter < max_iter; ++iter) {
        if ((fb > 0.0) == (fc > 0.0)) {
            // b and c on the same side: the sign change is now [a, b]; restart from it.
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }
        if (std::abs(fc) < std::abs(fb)) {
            // Keep b as the endpoint with the smaller |g| (the better estimate).
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }
        // Stop when the bracket half-width is within tolerance: then |b - x*| <= |c - b| <= 2 tol.
        const double tol = std::max(0.5 * xtol, 2.0 * kEps * std::abs(b));
        const double xm = 0.5 * (c - b);
        if (std::abs(xm) <= tol || fb == 0.0) {
            return b;
        }
        if (std::abs(e) >= tol && std::abs(fa) > std::abs(fb)) {
            // Interpolation step p/q: secant if only two distinct points (a == c), otherwise
            // inverse quadratic interpolation x(g) through (a, fa), (b, fb), (c, fc) at g = 0.
            const double s = fb / fa;
            double p = 0.0;
            double q = 0.0;
            if (a == c) {
                p = 2.0 * xm * s;
                q = 1.0 - s;
            } else {
                const double qa = fa / fc;
                const double r = fb / fc;
                p = s * (2.0 * xm * qa * (qa - r) - (b - a) * (r - 1.0));
                q = (qa - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if (p > 0.0) {
                q = -q;
            }
            p = std::abs(p);
            // Accept only if the step stays inside the bracket and is less than half the step
            // before last (guaranteed progress); otherwise bisect.
            const double min1 = 3.0 * xm * q - std::abs(tol * q);
            const double min2 = std::abs(e * q);
            if (2.0 * p < std::min(min1, min2)) {
                e = d;
                d = p / q;
            } else {
                d = xm;
                e = d;
            }
        } else {
            d = xm;
            e = d;
        }
        a = b;
        fa = fb;
        // Never step by less than tol, so each iteration makes progress.
        b += (std::abs(d) > tol) ? d : std::copysign(tol, xm);
        fb = g(b);
        SIMCORE_CHECK(std::isfinite(fb), "g returned a non-finite value");
    }
    SIMCORE_CHECK(false, "brent_root: max_iter exceeded");
    return b;  // unreachable
}

}  // namespace simcore

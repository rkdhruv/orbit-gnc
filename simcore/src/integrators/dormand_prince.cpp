#include "simcore/integrators/dormand_prince.hpp"

#include <algorithm>
#include <array>
#include <cmath>

#include "simcore/check.hpp"

namespace simcore {

namespace {

using Vec6 = DenseSegment::Vec6;

// Dormand & Prince (1980) tableau. Nodes c_i, stage weights a_ij, and the 5th-order weights,
// which equal the last row (a_7j), so stage 7 is f(t + h, y1).
constexpr double c2 = 1.0 / 5.0, c3 = 3.0 / 10.0, c4 = 4.0 / 5.0, c5 = 8.0 / 9.0;
constexpr double a21 = 1.0 / 5.0;
constexpr double a31 = 3.0 / 40.0, a32 = 9.0 / 40.0;
constexpr double a41 = 44.0 / 45.0, a42 = -56.0 / 15.0, a43 = 32.0 / 9.0;
constexpr double a51 = 19372.0 / 6561.0, a52 = -25360.0 / 2187.0, a53 = 64448.0 / 6561.0,
                 a54 = -212.0 / 729.0;
constexpr double a61 = 9017.0 / 3168.0, a62 = -355.0 / 33.0, a63 = 46732.0 / 5247.0,
                 a64 = 49.0 / 176.0, a65 = -5103.0 / 18656.0;
constexpr double a71 = 35.0 / 384.0, a73 = 500.0 / 1113.0, a74 = 125.0 / 192.0,
                 a75 = -2187.0 / 6784.0, a76 = 11.0 / 84.0;
// Error weights e_j = b_j (5th order) - b*_j (embedded 4th order).
constexpr double e1 = 71.0 / 57600.0, e3 = -71.0 / 16695.0, e4 = 71.0 / 1920.0,
                 e5 = -17253.0 / 339200.0, e6 = 22.0 / 525.0, e7 = -1.0 / 40.0;
// Hairer's dense-output weights for the 4th-order continuous extension.
constexpr double d1 = -12715105075.0 / 11282082432.0, d3 = 87487479700.0 / 32700410799.0,
                 d4 = -10690763975.0 / 1880347072.0, d5 = 701980252875.0 / 199316789632.0,
                 d6 = -1453857185.0 / 822651844.0, d7 = 69997945.0 / 29380423.0;

// Step-size controller: h_new = h * clamp(kSafety * err^(-1/5), kFacMin, kFacMax). The exponent
// is -1/5 because the error estimate's leading term scales as h^5 (it is the 4th-order error).
// This is a plain integral (I) controller; Hairer's DOPRI5 code defaults to a PI controller
// (Lund stabilisation, beta = 0.04) that damps step-size oscillation on stiff-ish stretches.
constexpr double kSafety = 0.9, kFacMin = 0.2, kFacMax = 5.0;
// Stub avoidance (as in Hairer's code): a step that would reach within 1% of t_end is stretched
// to land on it, instead of leaving a tiny final step.
constexpr double kStretch = 1.01;

Vec6 pack(const Eigen::Vector3d& r, const Eigen::Vector3d& v) {
    Vec6 y;
    y << r, v;
    return y;
}

// y' = [v; a(t, r, v)].
Vec6 deriv(const ForceModel& f, const Epoch& t, const Vec6& y, long& evals) {
    ++evals;
    const State s{t, y.head<3>(), y.tail<3>()};
    return pack(s.v_eci, f.acceleration_eci(s));
}

double atol_of(const DpOptions& o, int i) {
    return i < 3 ? o.atol_pos_m : o.atol_vel_mps;
}

// One term (x_i / sc_i)^2 of an RMS norm. sc_i = 0 is possible with atol = 0 and a component
// that is exactly 0: then x_i = 0 contributes nothing and any nonzero x_i is infinitely large.
double scaled_sq(double x, double sc) {
    if (sc > 0.0) {
        const double q = x / sc;
        return q * q;
    }
    return x == 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
}

// Hairer's error norm: sqrt(mean_i (est_i / sc_i)^2), sc_i = atol_i + rtol max(|y0_i|, |y1_i|).
double error_norm(const Vec6& est, const Vec6& y0, const Vec6& y1, const DpOptions& o) {
    double sum = 0.0;
    for (int i = 0; i < 6; ++i) {
        const double sc = atol_of(o, i) + o.rtol * std::max(std::abs(y0(i)), std::abs(y1(i)));
        sum += scaled_sq(est(i), sc);
    }
    return std::sqrt(sum / 6.0);
}

// RMS norm of x scaled by sk_i = atol_i + rtol |y0_i| (used only for the starting step).
double start_norm(const Vec6& x, const Vec6& y0, const DpOptions& o) {
    double sum = 0.0;
    for (int i = 0; i < 6; ++i) {
        sum += scaled_sq(x(i), atol_of(o, i) + o.rtol * std::abs(y0(i)));
    }
    return std::sqrt(sum / 6.0);
}

// Hairer, Norsett & Wanner, "Solving ODEs I", II.4: guess h from the sizes of y and y', take one
// explicit Euler trial step to estimate y'', and pick h so that h^5 * max(|y'|, |y''|) ~ 0.01.
// The trial step is clipped to the span so the force is never evaluated past t_end.
double initial_step(const ForceModel& f, const Epoch& t0, const Vec6& y0, const Vec6& f0,
                    double span_s, const DpOptions& o, long& evals) {
    const double dir = span_s > 0.0 ? 1.0 : -1.0;
    const double d0 = start_norm(y0, y0, o);
    const double d1n = start_norm(f0, y0, o);
    double h = (d0 < 1e-10 || d1n < 1e-10) ? 1e-6 : 0.01 * d0 / d1n;
    h = std::min({h, o.h_max_s, std::abs(span_s)});
    const Vec6 f1 = deriv(f, t0 + dir * h, y0 + (dir * h) * f0, evals);
    const double d2 = start_norm(f1 - f0, y0, o) / h;
    const double der12 = std::max(d2, d1n);
    const double h1 = der12 <= 1e-15 ? std::max(1e-6, h * 1e-3) : std::pow(0.01 / der12, 1.0 / 5.0);
    return std::min({100.0 * h, h1, o.h_max_s});
}

}  // namespace

DormandPrince54::DormandPrince54(DpOptions opts) : opts_(opts) {
    SIMCORE_CHECK(std::isfinite(opts_.rtol) && opts_.rtol > 0.0, "rtol must be finite and > 0");
    SIMCORE_CHECK(std::isfinite(opts_.atol_pos_m) && opts_.atol_pos_m >= 0.0,
                  "atol_pos_m must be finite and >= 0");
    SIMCORE_CHECK(std::isfinite(opts_.atol_vel_mps) && opts_.atol_vel_mps >= 0.0,
                  "atol_vel_mps must be finite and >= 0");
    SIMCORE_CHECK(std::isfinite(opts_.h_min_s) && opts_.h_min_s > 0.0,
                  "h_min_s must be finite and > 0");
    SIMCORE_CHECK(opts_.h_max_s >= opts_.h_min_s, "h_max_s must be >= h_min_s");  // NaN fails
    SIMCORE_CHECK(std::isfinite(opts_.h_initial_s) && opts_.h_initial_s >= 0.0,
                  "h_initial_s must be finite and >= 0");
    SIMCORE_CHECK(opts_.max_steps > 0, "max_steps must be > 0");
}

State DormandPrince54::propagate(const ForceModel& f, const State& s0, const Epoch& t_end,
                                 const std::function<StepAction(const DenseSegment&)>& on_step) {
    stats_ = {};
    const double span_s = t_end - s0.t;
    if (span_s == 0.0) {
        return s0;
    }
    const double dir = span_s > 0.0 ? 1.0 : -1.0;

    Epoch t = s0.t;
    Vec6 y = pack(s0.r_eci, s0.v_eci);
    Vec6 k1 = deriv(f, t, y, stats_.force_evals);

    double h_abs = opts_.h_initial_s > 0.0
                       ? opts_.h_initial_s
                       : initial_step(f, t, y, k1, span_s, opts_, stats_.force_evals);
    h_abs = std::min(h_abs, opts_.h_max_s);
    bool prev_rejected = false;

    while (true) {
        const double remaining_s = t_end - t;
        if (remaining_s * dir <= 0.0) {
            break;  // defensive: already at (or, by Epoch rounding, past) t_end
        }
        // Clip (or stretch by up to 1%) the final step to land on t_end; only a step the
        // controller chose can signal collapse.
        const bool last = kStretch * h_abs >= std::abs(remaining_s);
        const double h = last ? remaining_s : dir * h_abs;
        SIMCORE_CHECK(last || h_abs >= opts_.h_min_s, "step size collapsed below h_min_s");
        SIMCORE_CHECK(stats_.accepted + stats_.rejected < opts_.max_steps, "max_steps exceeded");

        // Stage times are computed fresh from t; on the clipped last step the c = 1 stages use
        // t_end itself, since t + h can round one ulp past it.
        const Epoch t1 = last ? t_end : t + h;
        long& ev = stats_.force_evals;
        const Vec6 k2 = deriv(f, t + c2 * h, y + h * (a21 * k1), ev);
        const Vec6 k3 = deriv(f, t + c3 * h, y + h * (a31 * k1 + a32 * k2), ev);
        const Vec6 k4 = deriv(f, t + c4 * h, y + h * (a41 * k1 + a42 * k2 + a43 * k3), ev);
        const Vec6 k5 =
            deriv(f, t + c5 * h, y + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4), ev);
        const Vec6 k6 =
            deriv(f, t1, y + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5), ev);
        const Vec6 y1 = y + h * (a71 * k1 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6);
        const Vec6 k7 = deriv(f, t1, y1, ev);

        const Vec6 est = h * (e1 * k1 + e3 * k3 + e4 * k4 + e5 * k5 + e6 * k6 + e7 * k7);
        const double err = error_norm(est, y, y1, opts_);

        if (err <= 1.0) {
            ++stats_.accepted;
            // err = 0 gives err^(-1/5) = inf, so the clamp takes kFacMax.
            double fac = std::min(kFacMax, std::max(kFacMin, kSafety * std::pow(err, -0.2)));
            if (prev_rejected) {
                fac = std::min(fac, 1.0);  // don't grow straight after a rejection
            }
            prev_rejected = false;

            StepAction action = StepAction::kContinue;
            if (on_step) {
                const Vec6 ydiff = y1 - y;
                const Vec6 bspl = h * k1 - ydiff;
                const std::array<Vec6, 5> c{
                    y, ydiff, bspl, ydiff - h * k7 - bspl,
                    h * (d1 * k1 + d3 * k3 + d4 * k4 + d5 * k5 + d6 * k6 + d7 * k7)};
                action = on_step(DenseSegment(t, t1, h, c));
            }
            t = t1;
            y = y1;
            k1 = k7;  // FSAL
            if (last || action == StepAction::kStop) {
                break;
            }
            h_abs = std::min(h_abs * fac, opts_.h_max_s);
        } else {
            ++stats_.rejected;
            // NaN err (e.g. a NaN force) fails err <= 1 and std::max picks kFacMin: the step
            // keeps shrinking until the collapse check throws.
            h_abs = std::abs(h) * std::max(kFacMin, kSafety * std::pow(err, -0.2));
            prev_rejected = true;
        }
    }
    return State{t, y.head<3>(), y.tail<3>()};
}

}  // namespace simcore
